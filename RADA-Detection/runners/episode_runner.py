from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import copy
import time


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -1000000

        self.adv_active = self.args.attack_active
        self.victim_idx = 0
        self.episode_result = 0
        self.episode_len = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        if self.args.attack_start_t == -1:
            self.attack_start_t = np.random.randint(0, high=self.args.attack_max_start)
        else:
            self.attack_start_t = self.args.attack_start_t

        if getattr(self.args, "victim_id", -1) == -1:
            self.victim_idx = np.random.randint(0, self.args.n_agents)
        else:
            self.victim_idx = self.args.victim_id
        if self.victim_idx >= self.args.n_agents: self.victim_idx = 0

    def run(self, advagent=None, tracker=None, test_mode=False):
        self.reset()
        if self.args.save_replay: self.env.render()

        terminated = False
        self.mac.init_hidden(batch_size=self.batch_size)
        episode_return = 0

        adv_test_mode = False
        if advagent is not None:
            adv_test_mode = advagent.test_mode
            if self.adv_active:
                if hasattr(advagent, 'model') and hasattr(advagent.model, 'init_hidden'):
                    advagent.hidden = advagent.model.init_hidden()

        if tracker is not None:
            tracker.hidden = tracker.init_hidden()
            tracker.last_action = th.zeros(1, tracker.n_tracker_agents, self.args.n_actions, dtype=th.float32).to(
                self.args.device)
            tracker.last_reward = 0
            last_actions_for_buffer = th.zeros_like(tracker.last_action.squeeze(0))

        episode_scores = []
        episode_adv_rewards = []

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            victim_idx = self.victim_idx
            adv_state = np.array(pre_transition_data["obs"][0][victim_idx])
            adv_avail_actions = self.env.get_avail_agent_actions(victim_idx)
            adv_action = None

            if advagent is not None and self.args.attack_active:
                if self.args.attack_type == "OA":
                    if self.t >= self.attack_start_t:
                        adv_action = advagent.compute_action(adv_state, adv_avail_actions)
                        input_obs = copy.deepcopy(pre_transition_data["obs"])
                        batch_temp = copy.deepcopy(self.batch)

                        victim_perturbed_obs = self.jsma_perturb(
                            batch_temp, advagent.input_shape, adv_action,
                            input_obs, victim_idx=victim_idx, theta=0.5, max_iter=10
                        )
                        pre_transition_data["obs"][0][victim_idx] = victim_perturbed_obs
                        self.batch.update(pre_transition_data, ts=self.t)
                else:
                    if self.t >= self.attack_start_t:
                        adv_action = advagent.compute_action(adv_state, adv_avail_actions)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            victim_action_final = actions[0, self.victim_idx].item()

            if advagent is not None and self.adv_active:
                if self.t >= self.attack_start_t and self.args.attack_type != "OA":
                    if adv_action is not None:
                        actions[0, self.victim_idx] = adv_action
                        victim_action_final = int(adv_action)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            current_actions_idx = actions[0].cpu().numpy()
            last_onehot_a_n = np.eye(self.args.n_actions)[current_actions_idx]
            last_a_tensor = th.tensor(last_onehot_a_n, dtype=th.float32).to(self.args.device)

            if tracker is not None:
                current_obs_list = pre_transition_data["obs"][0]
                trackers_obs_tensor = th.tensor(np.array(current_obs_list), dtype=th.float32).to(self.args.device)

                current_avail_actions = pre_transition_data["avail_actions"][0]

                role_embed = None
                if tracker.use_recl:
                    with th.no_grad():
                        if self.t == 0:
                            tracker.recl_hidden = tracker.recl_net.agent_embedding_net.fc1.weight.new(
                                1, self.args.n_agents, self.args.agent_embedding_dim).zero_()

                        agent_embed, tracker.recl_hidden = tracker.recl_net.agent_embedding_net(
                            trackers_obs_tensor.unsqueeze(0), tracker.last_action, tracker.recl_hidden
                        )
                        role_embed = tracker.recl_net.role_embedding_net(agent_embed)

                if getattr(tracker.args, 'tracker_train', False):
                    buffer_inputs = []
                    target_actions = []

                    eye = th.eye(self.args.n_agents, device=self.args.device)
                    for i in range(self.args.n_agents):
                        for j in range(self.args.n_agents):
                            if i == j: continue

                            id_i_vec = eye[i]
                            id_j_vec = eye[j]

                            if tracker.use_role_input and role_embed is not None:
                                buffer_input = th.cat([
                                    trackers_obs_tensor[i],
                                    role_embed[0, j].view(-1),
                                    id_i_vec,
                                    id_j_vec
                                ], dim=-1)
                            else:
                                buffer_input = th.cat([
                                    trackers_obs_tensor[i],
                                    id_i_vec,
                                    id_j_vec
                                ], dim=-1)

                            target_action_j = th.tensor([current_actions_idx[j]])

                            buffer_inputs.append(buffer_input.cpu().numpy())
                            target_actions.append(target_action_j.cpu().numpy())

                    final_buffer_input = np.concatenate(buffer_inputs)
                    final_target_action = np.concatenate(target_actions)

                    next_obs_list = self.env.get_obs()
                    next_obs_tensor = th.tensor(np.array(next_obs_list), dtype=th.float32).to(self.args.device)

                    next_buffer_inputs = []
                    for i in range(self.args.n_agents):
                        for j in range(self.args.n_agents):
                            if i == j: continue
                            id_i_vec = eye[i]
                            id_j_vec = eye[j]
                            if tracker.use_role_input and role_embed is not None:
                                buffer_next_input = th.cat([
                                    next_obs_tensor[i],
                                    role_embed[0, j].view(-1),
                                    id_i_vec,
                                    id_j_vec
                                ], dim=-1)
                            else:
                                buffer_next_input = th.cat([
                                    next_obs_tensor[i],
                                    id_i_vec,
                                    id_j_vec
                                ], dim=-1)
                            next_buffer_inputs.append(buffer_next_input.cpu().numpy())

                    final_buffer_next_input = np.concatenate(next_buffer_inputs)

                    tracker.buffer.push(
                        final_buffer_input,
                        final_target_action,
                        reward,
                        final_buffer_next_input,
                        terminated,
                        env_info.get("episode_limit", False) and self.t == self.episode_limit - 1,
                        last_actions_for_buffer.cpu().numpy()
                    )

                else:
                    q_matrix, tracker.hidden = tracker.forward(
                        trackers_obs_tensor,
                        tracker.last_action.squeeze(0),
                        tracker.hidden,
                        self.t
                    )

                    tracker.output_statistics(q_matrix, current_actions_idx, current_avail_actions)

                tracker.last_reward = reward
                tracker.last_action = last_a_tensor.unsqueeze(0)
                last_actions_for_buffer = last_a_tensor

                if advagent is not None and not adv_test_mode:
                    adv_reward = -reward
                    scores_on_victim = []
                    for i in range(tracker.n_tracker_agents):
                        if i != self.victim_idx:
                            sc = tracker.normality_score[i][self.victim_idx]
                            scores_on_victim.append(sc)

                    if len(scores_on_victim) > 0:
                        episode_scores.append(np.mean(scores_on_victim))
                        adv_reward += advagent.constraint_reward(scores_on_victim)

                    episode_adv_rewards.append(adv_reward)

                    next_obs_list = self.env.get_obs()
                    adv_next_state = np.array(next_obs_list[self.victim_idx])
                    advagent.store_episode(adv_state, adv_action if adv_action is not None else victim_action_final,
                                           adv_reward, adv_next_state, terminated, victim_action_final)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        if advagent is not None and not adv_test_mode and self.adv_active:
            advagent.update_exploration_probability()
            adv_loss = advagent.train()
            if adv_loss is not None:
                self.logger.log_stat("adv_loss", adv_loss, self.t_env)

            if len(episode_adv_rewards) > 0:
                self.logger.log_stat("adv_episode_reward", np.sum(episode_adv_rewards), self.t_env)
            if len(episode_scores) > 0:
                self.logger.log_stat("adv_mean_score_on_victim", np.mean(episode_scores), self.t_env)
            self.logger.log_stat("adv_exploration_proba", advagent.exploration_proba, self.t_env)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        self.episode_result = cur_stats.get('battle_won', -1)
        self.episode_len = self.t

        if not test_mode: self.t_env += self.t
        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if tracker is not None:
            tracker.out_dict["actual_victim"].append(self.victim_idx)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes": self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    def jsma_perturb(self, batch, adv_n_features, target_action, input_obs, victim_idx, theta=0.5, max_iter=5,
                     norm_limit=10):
        if theta == -1:
            theta_min = 0.1
            theta_max = 0.9
            theta_step = (theta_max - theta_min) / (max_iter - 1)
            theta_vector = np.arange(theta_min, theta_max + theta_step, theta_step)
        else:
            theta_vector = np.ones(max_iter) * theta

        adv_obs = copy.deepcopy(input_obs[0][victim_idx])
        agents_obs = input_obs
        perturbed_obs = input_obs[0][victim_idx]

        it = 0
        t_ep = copy.deepcopy(self.t)
        t_env = copy.deepcopy(self.t_env)

        actions = self.mac.select_actions(batch, t_ep, t_env, test_mode=True, temp_mode=True)
        victim_action = actions[0][victim_idx]
        target_achieved = (victim_action == target_action)
        limit_reached = False

        while it < max_iter and not target_achieved and not limit_reached:
            agents_q_values = self.mac.q_values_calc(batch, t_ep, t_env, test_mode=True, temp_mode=True)
            victim_q_values = agents_q_values[0][victim_idx]
            target_delta_q = []
            non_target_delta_q = []

            for k in range(adv_n_features):
                temp_obs = copy.deepcopy(perturbed_obs)
                temp_obs[k] += 0.01
                agents_obs[0][victim_idx] = temp_obs
                batch.update({"obs": agents_obs}, ts=t_ep)
                perturbed_agents_q = self.mac.q_values_calc(batch, t_ep, t_env, test_mode=True, temp_mode=True)
                perturbed_q = perturbed_agents_q[0][victim_idx]
                feature_delta_q = perturbed_q[target_action] - victim_q_values[target_action]
                target_delta_q.append(feature_delta_q)
                valid_q = perturbed_q[perturbed_q != -float("inf")]
                valid_victim_q = victim_q_values[victim_q_values != -float("inf")]
                feature_delta_q_sum = (th.sum(valid_q) - th.sum(valid_victim_q))
                non_target_delta_q.append(feature_delta_q_sum - feature_delta_q)

            best_criterion = 0
            target_i = 0
            target_j = 0
            for i in range(adv_n_features - 1):
                target_i_term = target_delta_q[i]
                non_target_i_term = non_target_delta_q[i]
                for j in range(i + 1, adv_n_features):
                    target_j_term = target_delta_q[j]
                    non_target_j_term = non_target_delta_q[j]
                    criterion = (target_i_term + target_j_term) * (non_target_i_term + non_target_j_term)
                    if criterion < 0:
                        if -criterion > best_criterion:
                            best_criterion = -criterion
                            target_i = i
                            target_j = j
            if best_criterion > 0:
                sign = th.sign(target_delta_q[target_i] + target_delta_q[target_j])
                perturbed_obs[target_i] += theta_vector[it] * sign
                perturbed_obs[target_j] += theta_vector[it] * sign

            agents_obs[0][victim_idx] = perturbed_obs
            batch.update({"obs": agents_obs}, ts=t_ep)
            actions = self.mac.select_actions(batch, t_ep, t_env, test_mode=True, temp_mode=True)
            victim_action = actions[0][victim_idx]
            target_achieved = (victim_action == target_action)
            if np.linalg.norm(perturbed_obs - adv_obs, 1) > norm_limit: limit_reached = True
            it += 1

        return perturbed_obs