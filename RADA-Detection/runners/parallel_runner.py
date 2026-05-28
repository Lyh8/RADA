from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def _build_pair_inputs_vectorized(self, obs, role_embed, eye_t, pair_mask, use_role_input):
        """
        Vectorized construction of buffer inputs for all agent-pairs.
        Replaces the original N x N Python for loop.

        Args:
            obs:             [B_active, N, obs_dim] tensor (GPU)
            role_embed:      [B_active, N, role_dim] tensor (GPU) or None
            eye_t:           [N, N] identity tensor (GPU)
            pair_mask:       [N, N] bool tensor, True where i != j
            use_role_input:  bool, controls whether role_j is stored into the buffer

        Returns:
            all_inputs_np: [B_active, N*(N-1)*single_dim] numpy (CPU)
            This is directly the final_buffer_input that can be pushed into the buffer.
        """
        B, N, _ = obs.shape

        obs_i = obs.unsqueeze(2).expand(-1, N, N, -1)
        id_i = eye_t.unsqueeze(0).unsqueeze(2).expand(B, N, N, -1)
        id_j = eye_t.unsqueeze(0).unsqueeze(1).expand(B, N, N, -1)

        if use_role_input and role_embed is not None:
            role_j = role_embed.unsqueeze(1).expand(-1, N, N, -1)
            all_pairs = th.cat([obs_i, role_j, id_i, id_j], dim=-1)
        else:
            all_pairs = th.cat([obs_i, id_i, id_j], dim=-1)

        pairs_no_diag = all_pairs[:, pair_mask]

        pairs_np = pairs_no_diag.cpu().numpy()

        return pairs_np.reshape(B, -1)

    def _build_target_actions_vectorized(self, cpu_actions_active, pair_mask_np):
        """
        Vectorized construction of target actions for all agent-pairs.

        Args:
            cpu_actions_active: [B_active, N] numpy
            pair_mask_np:       [N, N] bool numpy, True where i != j

        Returns:
            [B_active, N*(N-1)] numpy
        """
        B, N = cpu_actions_active.shape
        acts_j = np.tile(cpu_actions_active[:, np.newaxis, :], (1, N, 1))
        return acts_j[:, pair_mask_np]

    def run(self, test_mode=False, tracker=None):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        tracker_train = (tracker is not None and getattr(tracker.args, 'tracker_train', False))
        if tracker_train:
            N = self.env_info["n_agents"]
            n_act = self.env_info["n_actions"]
            dev = tracker.device

            if tracker.use_recl:
                t_recl_h = tracker.recl_net.agent_embedding_net.fc1.weight.new(
                    self.batch_size, N, tracker.args.agent_embedding_dim).zero_()

            t_last_a = th.zeros(self.batch_size, N, n_act, dtype=th.float32, device=dev)

            eye_t = th.eye(N, device=dev)
            pair_mask = ~th.eye(N, dtype=th.bool, device=dev)
            pair_mask_np = ~np.eye(N, dtype=bool)

            t_ep_data = [[] for _ in range(self.batch_size)]

        while True:

            if tracker_train:
                cur_obs_all = self.batch["obs"][:, self.t].float().to(dev)

                if tracker.use_recl:
                    with th.no_grad():
                        t_agent_e, t_recl_h = tracker.recl_net.agent_embedding_net(
                            cur_obs_all, t_last_a, t_recl_h)
                        t_role_e = tracker.recl_net.role_embedding_net(t_agent_e)
                else:
                    t_role_e = None

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            env_actions_full = np.zeros((self.batch_size, N), dtype=np.int64) if tracker_train else None
            if tracker_train:
                act_idx = 0
                active_env_indices = []
                for env_idx in range(self.batch_size):
                    if env_idx in envs_not_terminated and not terminated[env_idx]:
                        env_actions_full[env_idx] = cpu_actions[act_idx]
                        active_env_indices.append(env_idx)
                        act_idx += 1

                if len(active_env_indices) > 0:
                    active_idx_t = th.tensor(active_env_indices, dtype=th.long, device=dev)
                    B_active = len(active_env_indices)

                    obs_active = cur_obs_all[active_idx_t]
                    role_active = t_role_e[active_idx_t] if t_role_e is not None else None

                    buf_inputs_np = self._build_pair_inputs_vectorized(
                        obs_active, role_active, eye_t, pair_mask, tracker.use_role_input
                    )

                    cpu_actions_active = env_actions_full[active_env_indices]
                    tgt_actions_np = self._build_target_actions_vectorized(
                        cpu_actions_active, pair_mask_np
                    )

                    last_a_np = t_last_a[active_idx_t].cpu().numpy()
                    role_active_np = role_active.detach().cpu().numpy() if role_active is not None else None

                    for k, env_idx in enumerate(active_env_indices):
                        t_ep_data[env_idx].append({
                            'buf_input': buf_inputs_np[k],
                            'tgt_action': tgt_actions_np[k],
                            'last_a': last_a_np[k],
                            'role_e_np': role_active_np[k] if role_active_np is not None else None,
                            'reward': None,
                        })

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            next_obs_collector = {} if tracker_train else None

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

                    if tracker_train and len(t_ep_data[idx]) > 0 and t_ep_data[idx][-1]['reward'] is None:
                        entry = t_ep_data[idx][-1]
                        entry['reward'] = data["reward"]
                        entry['terminated'] = data["terminated"]
                        entry['truncated'] = (data["terminated"] and
                                              data["info"].get("episode_limit", False))
                        next_obs_collector[idx] = np.array(data["obs"])

            if tracker_train and next_obs_collector:
                env_indices_with_next = sorted(next_obs_collector.keys())
                B_next = len(env_indices_with_next)

                next_obs_batch = np.stack([next_obs_collector[idx] for idx in env_indices_with_next])
                next_obs_t = th.tensor(next_obs_batch, dtype=th.float32, device=dev)

                if tracker.use_recl and t_role_e is not None:
                    next_role_t = t_role_e[th.tensor(env_indices_with_next, dtype=th.long, device=dev)]
                else:
                    next_role_t = None

                next_buf_np = self._build_pair_inputs_vectorized(
                    next_obs_t, next_role_t, eye_t, pair_mask, tracker.use_role_input
                )

                for k, env_idx in enumerate(env_indices_with_next):
                    t_ep_data[env_idx][-1]['next_buf_input'] = next_buf_np[k]

            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            if tracker_train:
                for env_idx in active_env_indices:
                    acts = env_actions_full[env_idx]
                    onehot = np.eye(n_act)[acts]
                    t_last_a[env_idx] = th.tensor(onehot, dtype=th.float32, device=dev)

            self.t += 1

            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        if tracker_train:
            for env_idx in range(self.batch_size):
                for t_data in t_ep_data[env_idx]:
                    if t_data.get('reward') is None or 'next_buf_input' not in t_data:
                        continue
                    tracker.buffer.push(
                        t_data['buf_input'],
                        t_data['tgt_action'],
                        t_data['reward'],
                        t_data['next_buf_input'],
                        t_data['terminated'],
                        t_data['truncated'],
                        t_data['last_a']
                    )

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
