import numpy as np
import torch
import os
import json
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.utils.trans_tools import _t2n

from harl.algorithms.trackers.tracker import DecentralizedTracker, ABLATION_NO_ROLE



class OnPolicyMARunnerWithTracker(OnPolicyBaseRunner):

    def __init__(self, args, algo_args, env_args):
        self.use_tracker = algo_args["algo"].get("use_tracker", False)
        self.tracker_train = algo_args["algo"].get("tracker_train", False)

        super().__init__(args, algo_args, env_args)
        self.tracker_start_episode = algo_args["algo"].get("tracker_start_episode", 0)
        self.current_episode = 0
        self.collect_embeddings_mode = algo_args["algo"].get("collect_embeddings", False)
        self._collected_embeddings = []
        self.ablation_variant = algo_args["algo"].get("ablation_variant", "full")

        if self.use_tracker and not self.algo_args["render"]["use_render"]:
            obs_dim = self.envs.observation_space[0].shape[0]
            action_space = self.envs.action_space[0]
            action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n
            role_dim = algo_args["algo"].get("recl_role_embedding_dim", 32)

            self.tracker = DecentralizedTracker(
                num_agents=self.num_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                role_embedding_dim=role_dim,
                hidden_dim=algo_args["algo"].get("tracker_hidden_dim", 128),
                cluster_num=algo_args["algo"].get("recl_cluster_num", 2),
                device=self.device,
                tracker_lr=algo_args["algo"].get("tracker_lr", 5e-4),
                grad_norm_clip=algo_args["algo"].get("tracker_grad_norm_clip", 10.0),
                train_epochs=algo_args["algo"].get("tracker_train_epochs", 5),
                mini_batch_size=algo_args["algo"].get("tracker_mini_batch_size", 32),
                thresholds=algo_args["algo"].get("tracker_thresholds", [-9.5, -11.0, -12.5, -15.0]),
                windows=algo_args["algo"].get("tracker_windows", [-1, 5, 10, 20]),
                role_dev_threshold=algo_args["algo"].get("tracker_role_dev_threshold", 15.0),
                ablation_variant=self.ablation_variant,
            )

            self.tracker_train_freq = algo_args["algo"].get("tracker_train_freq", 1)
            self.tracker_train_step = 0

            if self.tracker_train:
                self._tracker_data_buffer = {
                    'obs': [], 'actions': [], 'role_embeddings': [], 'masks': [],
                }
                self._tracker_buffer_episodes = 0
                self._tracker_buffer_max = algo_args["algo"].get("tracker_buffer_episodes", 100)

            tracker_model_dir = algo_args["algo"].get("tracker_model_dir", None)
            if tracker_model_dir is not None:
                self._load_tracker_prerequisites(tracker_model_dir)

            print(f"[Tracker] Initialized: train={self.tracker_train}, "
                  f"epochs={self.tracker.train_epochs}, "
                  f"mini_batch={self.tracker.mini_batch_size}, "
                  f"buffer_max={self._tracker_buffer_max if self.tracker_train else 'N/A'}, "
                  f"ablation={self.ablation_variant}")

    def _load_tracker_prerequisites(self, model_dir):
        print(f"[Tracker] Skipping value_normalizer (it normalizes the value scalar, not obs)")

        if self.ablation_variant == ABLATION_NO_ROLE:
            print(f"[Tracker] Ablation V1 (no_role): Skipping cluster centers loading (Stage 1 disabled)")
        else:
            cluster_path = os.path.join(model_dir, "cluster_centers.npy")
            threshold_path = os.path.join(model_dir, "cluster_thresholds.npy")
            if os.path.exists(cluster_path):
                centers = np.load(cluster_path)
                thresholds = np.load(threshold_path) if os.path.exists(threshold_path) else None
                self.tracker.load_cluster_info(centers, thresholds)
            else:
                print(f"[Tracker] Warning: cluster_centers.npy not found in {model_dir}")

        tracker_subdir = os.path.join(model_dir, "tracker")
        if os.path.exists(os.path.join(tracker_subdir, "tracker_net.pt")):
            self.tracker.load(tracker_subdir)
        elif os.path.exists(os.path.join(model_dir, "tracker_net.pt")):
            self.tracker.load(model_dir)
        else:
            print(f"[Tracker] WARNING: tracker_net.pt not found in {model_dir} or {tracker_subdir}!")


    def train(self):
        self.current_episode += 1
        actor_train_infos = []
        critic_train_info = {}

        freeze_policy = (self.use_tracker and self.tracker_train)

        if freeze_policy:
            for _ in range(self.num_agents):
                actor_train_infos.append({})
            if self.current_episode == 1:
                print("=" * 60)
                print(f"[Phase 2] MAPPO actor/critic/ReCL frozen! Only training Tracker.")
                print(f"[Phase 2] Ablation variant: {self.ablation_variant}")
                print("=" * 60)
            elif self.current_episode % 100 == 0:
                print(f"[Phase 2] Episode {self.current_episode}, "
                      f"tracker_steps={getattr(self.tracker, 'training_steps'   , 0)}, "
                      f"ablation={self.ablation_variant}")
        else:
            if self.value_normalizer is not None:
                advantages = self.critic_buffer.returns[:-1] - \
                    self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
            else:
                advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

            if self.state_type == "FP":
                active_masks_collector = [
                    self.actor_buffer[i].active_masks for i in range(self.num_agents)
                ]
                active_masks_array = np.stack(active_masks_collector, axis=2)
                advantages_copy = advantages.copy()
                advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            if self.share_param:
                actor_train_info = self.actor[0].share_param_train(
                    self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
                )
                for _ in range(self.num_agents):
                    actor_train_infos.append(actor_train_info)
            else:
                for agent_id in range(self.num_agents):
                    if self.state_type == "EP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_id], advantages.copy(), "EP"
                        )
                    elif self.state_type == "FP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_id],
                            advantages[:, :, agent_id].copy(), "FP",
                        )
                    actor_train_infos.append(actor_train_info)

            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

            if self.use_recl:
                self.recl_train_step += 1
                if self.recl_train_step % self.recl_train_freq == 0:
                    batch_obs, batch_active = self._gather_obs_for_recl()
                    recl_info = self.recl.update(batch_obs, batch_active)
                    critic_train_info["recl_ae_loss"] = recl_info["recl_ae_loss"]
                    critic_train_info["recl_cl_loss"] = recl_info["recl_cl_loss"]

        if self.collect_embeddings_mode and self.use_recl:
            self._collect_embeddings_for_clustering()

        if self.use_tracker and self.tracker_train:
            if self.current_episode >= self.tracker_start_episode:
                self.tracker_train_step += 1

                if self.tracker_train_step == 1:
                    self._maybe_generate_cluster_centers()

                self._collect_tracker_training_data()

                if (self.tracker_train_step % self.tracker_train_freq == 0 and
                    self._tracker_buffer_episodes >= self._tracker_buffer_max):

                    buffer_size_before_train = self._tracker_buffer_episodes

                    tracker_info = self._train_tracker()

                    if tracker_info:
                        tracker_info['tracker_buffer_episodes'] = buffer_size_before_train

                        self._log_tracker_to_tensorboard(tracker_info)

                        updates = tracker_info.get('tracker_updates_this_call', 0)
                        print(f"[Tracker] step={self.tracker.training_steps}, "
                              f"nll={tracker_info['tracker_nll_loss']:.4f}, "
                              f"std={tracker_info['tracker_mean_std']:.4f}, "
                              f"grad={tracker_info['tracker_grad_norm']:.4f}, "
                              f"updates={updates}, buf={buffer_size_before_train}")
            else:
                if self.use_recl and self.current_episode % 10 == 0:
                    self._collect_embeddings_for_clustering()

        return actor_train_infos, critic_train_info

    def _log_tracker_to_tensorboard(self, tracker_info):
        if not hasattr(self, 'writter') or self.writter is None:
            return

        total_num_steps = (
            self.current_episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )

        for key, value in tracker_info.items():
            if isinstance(value, (int, float)):
                self.writter.add_scalar(f"tracker/{key}", value, total_num_steps)

    def _collect_embeddings_for_clustering(self):
        if not hasattr(self, 'recl') or self.recl is None:
            return
        obs_list = []
        for agent_id in range(self.num_agents):
            obs_list.append(self.actor_buffer[agent_id].obs)
        batch_obs = np.stack(obs_list, axis=2).transpose(1, 0, 2, 3)
        try:
            role_embeddings, _ = self.recl.get_role_embeddings(batch_obs)
            self._collected_embeddings.append(role_embeddings)
        except Exception as e:
            print(f"[Tracker] Warning: Failed to collect embeddings: {e}")

    def _maybe_generate_cluster_centers(self):
        if self.ablation_variant == ABLATION_NO_ROLE:
            print(f"[Tracker] Ablation V1: Skipping cluster centers generation (Stage 1 disabled)")
            return

        if self.tracker.cluster_centers is not None:
            print(f"[Tracker] Cluster centers already loaded.")
            return
        if len(self._collected_embeddings) > 0:
            try:
                from sklearn.cluster import KMeans
                all_embeds = np.concatenate(self._collected_embeddings, axis=0)
                flat_embeds = all_embeds.reshape(-1, all_embeds.shape[-1])
                n_clusters = self.tracker.cluster_num
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(flat_embeds)
                centers = kmeans.cluster_centers_
                thresholds = np.zeros(n_clusters)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.sum() > 0:
                        dists = np.linalg.norm(flat_embeds[mask] - centers[k], axis=1)
                        thresholds[k] = dists.max() * 1.2
                self.tracker.load_cluster_info(centers, thresholds)
                if hasattr(self, 'save_dir') and self.save_dir is not None:
                    np.save(os.path.join(str(self.save_dir), "cluster_centers.npy"), centers)
                    np.save(os.path.join(str(self.save_dir), "cluster_thresholds.npy"), thresholds)
                for k in range(n_clusters):
                    print(f"  Cluster {k}: size={(labels==k).sum()}, threshold={thresholds[k]:.4f}")
            except Exception as e:
                print(f"[Tracker] Auto cluster generation failed: {e}")
        else:
            print("[Tracker] Warning: No cluster centers available. Stage 1 detection disabled.")

    def _collect_tracker_training_data(self):
        obs_list, action_list, mask_list = [], [], []
        for agent_id in range(self.num_agents):
            obs_list.append(self.actor_buffer[agent_id].obs)
            action_list.append(self.actor_buffer[agent_id].actions)
            mask_list.append(self.actor_buffer[agent_id].active_masks)

        batch_obs = np.stack(obs_list, axis=2).transpose(1, 0, 2, 3)
        batch_actions = np.stack(action_list, axis=2).transpose(1, 0, 2, 3)
        batch_masks = np.stack(mask_list, axis=2).squeeze(-1).transpose(1, 0, 2)

        if hasattr(self, 'recl') and self.recl is not None:
            role_embeddings, _ = self.recl.get_role_embeddings(batch_obs)
        else:
            role_dim = self.tracker.role_embedding_dim
            B_r, T_r, N_r = batch_obs.shape[0], batch_obs.shape[1], batch_obs.shape[2]
            role_embeddings = np.zeros((B_r, T_r, N_r, role_dim), dtype=np.float32)
            print(f"[Tracker] Warning: ReCL not available, using zero role embeddings")

        self._tracker_data_buffer['obs'].append(batch_obs[:, :-1])
        self._tracker_data_buffer['actions'].append(batch_actions)
        self._tracker_data_buffer['role_embeddings'].append(role_embeddings[:, :-1])
        self._tracker_data_buffer['masks'].append(batch_masks[:, :-1])
        self._tracker_buffer_episodes += batch_obs.shape[0]

    def _train_tracker(self):
        if len(self._tracker_data_buffer['obs']) == 0:
            return {}
        all_obs = np.concatenate(self._tracker_data_buffer['obs'], axis=0)
        all_actions = np.concatenate(self._tracker_data_buffer['actions'], axis=0)
        all_role_embeds = np.concatenate(self._tracker_data_buffer['role_embeddings'], axis=0)
        all_masks = np.concatenate(self._tracker_data_buffer['masks'], axis=0)

        tracker_info = self.tracker.train_step(
            all_obs, all_actions, all_role_embeds, all_masks
        )

        self._tracker_data_buffer = {
            'obs': [], 'actions': [], 'role_embeddings': [], 'masks': [],
        }
        self._tracker_buffer_episodes = 0
        return tracker_info

    @torch.no_grad()
    def eval(self):
        if not self.use_tracker or self.tracker_train:
            super().eval()
            return

        self.logger.eval_init()
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        n_eval_threads = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_rnn_states = np.zeros((
            n_eval_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size,
        ), dtype=np.float32)
        eval_masks = np.ones((n_eval_threads, self.num_agents, 1), dtype=np.float32)

        self.tracker.init_hidden(n_eval_threads)
        if self.use_recl:
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((n_eval_threads, self.num_agents, self.tracker.action_dim))
        episode_rewards = [0.0] * n_eval_threads
        episode_steps = [0] * n_eval_threads

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id], eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            obs_tensor = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
            act_tensor = torch.tensor(last_actions, dtype=torch.float32, device=self.device)

            if self.use_recl:
                with torch.no_grad():
                    agent_embed = self.recl.embedding_net.agent_embed_forward(
                        obs_tensor.reshape(-1, self.tracker.obs_dim), detach=True)
                    role_embed = self.recl.embedding_net.role_embed_forward(
                        agent_embed, detach=True, ema=False)
                    role_embed = role_embed.reshape(n_eval_threads, self.num_agents, -1)
            else:
                role_embed = torch.zeros(
                    n_eval_threads, self.num_agents, self.tracker.role_embedding_dim,
                    dtype=torch.float32, device=self.device
                )

            mu, std, _ = self.tracker.forward(obs_tensor, act_tensor, role_embed)
            action_tensor = torch.tensor(eval_actions, dtype=torch.float32, device=self.device)
            scores = self.tracker.compute_scores(mu, std, action_tensor, eval_masks.squeeze(-1))
            self.tracker.update_detection_stats(scores[0])

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_available_actions) = self.eval_envs.step(eval_actions)

            self.logger.eval_per_step((
                eval_obs, eval_share_obs, eval_rewards,
                eval_dones, eval_infos, eval_available_actions))

            last_actions = eval_actions.copy()
            for t in range(n_eval_threads):
                episode_rewards[t] += eval_rewards[t].mean()
                episode_steps[t] += 1

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = 0
            eval_masks = np.ones((n_eval_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(n_eval_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)
                    self.tracker.end_episode(
                        reward=episode_rewards[eval_i],
                        episode_length=episode_steps[eval_i], attacked_agent=None)
                    episode_rewards[eval_i] = 0.0
                    episode_steps[eval_i] = 0
                    last_actions[eval_i] = 0.0
                    self.tracker.hidden[eval_i] = 0.0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                if self.use_tracker:
                    log_path = os.path.join(str(self.log_dir), "tracker_eval_logs.json")
                    self.tracker.save_episode_logs(log_path)
                    self._print_detection_summary()
                break

    def _print_detection_summary(self):
        print("\n" + "=" * 60)
        print(f"[Tracker] Detection Summary (ablation={self.ablation_variant})")
        print("=" * 60)
        for w_idx, window in enumerate(self.tracker.windows):
            for th_idx, threshold in enumerate(self.tracker.thresholds):
                result = self.tracker.get_detection_result(w_idx, th_idx)
                t_detect = result['t_detect']
                detected = (t_detect < 1e5).sum()
                if detected > 0:
                    avg_t = t_detect[t_detect < 1e5].mean()
                    total = self.num_agents * (self.num_agents - 1)
                    print(f"  W={window}, Th={threshold}: "
                          f"{detected}/{total} detected, avg_time={avg_t:.1f}")
        print("=" * 60 + "\n")

    def save(self):
        super().save()
        if self.use_tracker:
            self.tracker.save(os.path.join(str(self.save_dir), "tracker"))

    def restore(self):
        super().restore()
        if self.use_tracker and hasattr(self, 'tracker'):
            tracker_dir = os.path.join(str(self.algo_args["train"]["model_dir"]), "tracker")
            if os.path.exists(tracker_dir):
                self.tracker.load(tracker_dir)
                print(f"[Tracker] Restored from {tracker_dir}")

    def prep_rollout(self):
        super().prep_rollout()
        if hasattr(self, 'tracker') and self.use_tracker:
            self.tracker.prep_rollout()

    def prep_training(self):
        super().prep_training()
        if hasattr(self, 'tracker') and self.use_tracker:
            self.tracker.prep_training()
