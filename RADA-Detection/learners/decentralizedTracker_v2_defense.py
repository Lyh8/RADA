import copy
import torch
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from modules.roles.role_nets import RECL_NET
from modules.trackers.tracker_agent import TrackerAgent
from torch.optim import Adam
from utils.new_buffer_recurrent import MyRecurrentReplayBuffer


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class Tracker:
    def __init__(self, n_tracker_agents, input_shape, args):
        self.n_tracker_agents = n_tracker_agents
        self.input_shape = input_shape
        self.args = args
        self.device = get_device()

        self.cluster_num = getattr(args, 'cluster_num', 3)
        self.use_recl = getattr(args, 'use_recl', True)
        self.use_tf = getattr(args, 'use_tf', True)
        self.use_role_input = getattr(args, 'use_role_input', self.use_recl)
        if not self.use_recl:
            self.use_role_input = False
        self.multi_steps = getattr(args, 'multi_steps', 10)
        self.train_mode = getattr(args, 'tracker_train', False)

        self.use_stage1 = getattr(args, 'use_stage1', True)
        self.use_stage2 = getattr(args, 'use_stage2', True)

        if (self.use_stage1 or self.use_stage2) and not self.use_recl:
            if self.use_stage1:
                print("[Tracker] Warning: use_stage1=True requires use_recl=True (role embedding), enabled automatically")
                self.use_recl = True
        if not self.use_stage1 and not self.use_stage2:
            print("[Tracker] Warning: use_stage1 and use_stage2 cannot both be False, restored to the full scheme")
            self.use_stage1 = True
            self.use_stage2 = True

        self.default_threshold = getattr(args, 'role_dev_threshold', 15.0)

        self.defense_active = getattr(args, 'defense_active', False)
        self.defense_mode = getattr(args, 'defense_mode', 'action_replace')
        self.vote_mode = getattr(args, 'vote_mode', 'role_soft')
        self.w_same = getattr(args, 'w_same', 2.0)
        self.w_diff = getattr(args, 'w_diff', 1.0)
        self.softmax_temperature = getattr(args, 'softmax_temperature', 1.0)
        self.defense_delay = getattr(args, 'defense_delay', -1)
        self.defense_th = getattr(args, 'defense_th', -3.0)
        self.defense_window_idx = getattr(args, 'defense_window_idx', 0)

        self.detect_strategy = getattr(args, 'detect_strategy', 'same_role_only')
        self.detect_min_steps = getattr(args, 'detect_min_steps', 3)

        self.defense_stats = {
            "ep_summaries": [],
        }
        self._ep_corrected_count = 0
        self._ep_total_steps = 0
        self._ep_defense_steps = 0
        self._ep_consensus_values = []
        self._ep_detected_per_step = []
        self._ep_defense_log = []
        self._ep_accuracy_hits = []

        self.defense_calibrate = getattr(args, 'defense_calibrate', False)
        self._calibration_metrics = []

        self.centers = None
        self.cluster_thresholds = None

        self.current_labels = None
        self.current_deviations = None
        self.current_min_dists = None
        self.save_embeddings = getattr(args, 'save_embeddings', False)
        self.save_detailed_stats = getattr(args, 'save_detailed_stats', False)
        self.collected_embeddings = [] if self.save_embeddings else None

        if self.use_recl:
            self.recl_net = RECL_NET(args).to(self.device)
            self.recl_net.eval()

        predictor_input_shape = input_shape
        if self.use_tf:
            predictor_input_shape += args.n_actions
        if self.use_role_input:
            predictor_input_shape += args.role_embedding_dim
        predictor_input_shape += n_tracker_agents * 2

        target_dim = getattr(args, "tracker_hidden_dim", args.hidden_dim)

        print(f"[Tracker] init: input={predictor_input_shape}, hidden={target_dim} "
              f"(QMIX={args.hidden_dim}), use_recl={self.use_recl}, use_tf={self.use_tf}, "
              f"use_role_input={self.use_role_input}, "
              f"use_stage1={self.use_stage1}, use_stage2={self.use_stage2}, "
              f"defense_active={self.defense_active}, defense_mode={self.defense_mode}, "
              f"vote_mode={self.vote_mode}, detect_strategy={self.detect_strategy}, "
              f"defense_th={self.defense_th}, defense_window_idx={self.defense_window_idx}")

        self.tracker_net = TrackerAgent(
            predictor_input_shape,
            args,
            hidden_dim=target_dim
        ).to(self.device)
        self.tracker_net.eval()

        N = n_tracker_agents
        self.n_pairs = N * (N - 1)
        pair_j_list = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pair_j_list.append(j)
        self.pair_j_indices = torch.tensor(pair_j_list, dtype=torch.long)

        self.single_input_dim = input_shape + N * 2
        if self.use_role_input:
            self.single_input_dim += args.role_embedding_dim

        if self.train_mode:
            self.tracker_net.train()
            self.optimiser = Adam(self.tracker_net.parameters(), lr=args.tracker_lr)
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

            buffer_input_dim = self.single_input_dim * self.n_pairs
            buffer_a_dim = self.n_pairs

            self.buffer = MyRecurrentReplayBuffer(
                o_dim=buffer_input_dim,
                a_dim=buffer_a_dim,
                max_episode_len=args.episode_limit,
                capacity=getattr(args, 'tracker_buffer_size', 2000),
                batch_size=args.batch_size,
                n_agents=n_tracker_agents,
                action_dim=args.n_actions,
            )
            self.training_steps = 0

        self.hidden = self.init_hidden()
        self.recl_hidden = None

        self.windows = getattr(args, 'tracker_window', [-1, 10])
        if self.defense_window_idx >= len(self.windows):
            print(f"[Tracker] Warning: defense_window_idx={self.defense_window_idx} out of "
                  f"windows range (len={len(self.windows)}), falling back to 0")
            self.defense_window_idx = 0
        self.TH = args.thresholds if hasattr(args, 'thresholds') else [-3.0]

        self.normality_score = np.zeros((n_tracker_agents, n_tracker_agents))
        self.normality_metric = np.zeros((len(self.windows), self.n_tracker_agents, self.n_tracker_agents))
        self.t = 0
        self.t_detect = 1000 * np.ones((len(self.windows), len(self.TH), self.n_tracker_agents, self.n_tracker_agents))

        self.out_dict = {
            "attacked": [], "t_start": [], "window_size": self.windows,
            "threshold": self.TH, "t_detect": [], "battle_won": [],
            "ep_length": [], "actual_victim": [], "scores": [],
            "defense_stats": [],
        }
        self.episode_scores_history = []

    def init_hidden(self):
        h = self.tracker_net.init_hidden()
        return h.unsqueeze(1).unsqueeze(1).expand(1, self.n_tracker_agents, self.n_tracker_agents, -1).contiguous()

    def load_recl_weights(self, path):
        """Load only the RECL network weights"""
        if not self.use_recl:
            print(f"[Tracker] use_recl=False, skipping RECL weight loading.")
            return
        recl_path = os.path.join(path, "recl_net.th")
        if os.path.exists(recl_path):
            self.recl_net.load_state_dict(torch.load(recl_path, map_location=self.device))
            print(f"[Tracker] Loaded RECL weights from {recl_path}.")
        else:
            print(f"[Tracker] Warning: RECL weights not found in {path}")

    def load_tracker_weights(self, path):
        """Load only the Tracker network weights (and optimizer)"""
        p = os.path.join(path, "tracker_agent.th")
        if os.path.exists(p):
            self.tracker_net.load_state_dict(torch.load(p, map_location=self.device))
            print(f"[Tracker] Loaded Tracker weights from {p}.")
            if self.train_mode:
                opt_p = os.path.join(path, "tracker_opt.th")
                if os.path.exists(opt_p):
                    self.optimiser.load_state_dict(torch.load(opt_p, map_location=self.device))
                    print(f"[Tracker] Loaded Tracker optimizer state from {opt_p}.")
        else:
            print(f"[Tracker] Warning: Tracker weights not found in {path}")

    def load_clustering_info(self, path):
        """Load cluster centers and thresholds"""
        if not self.use_recl:
            print(f"[Tracker] use_recl=False, skipping clustering info loading.")
            return
        centers_path = os.path.join(path, "cluster_centers.npy")
        if os.path.exists(centers_path):
            centers_np = np.load(centers_path)
            self.centers = torch.tensor(centers_np, dtype=torch.float32).to(self.device)
            print(f"[Tracker] Loaded Cluster Centers from {centers_path}.")
        else:
            print(f"[Tracker] Warning: cluster_centers.npy not found in {path}")

        th_path = os.path.join(path, "cluster_thresholds.npy")
        if os.path.exists(th_path):
            th_np = np.load(th_path)
            self.cluster_thresholds = torch.tensor(th_np, dtype=torch.float32).to(self.device)
            print(f"[Tracker] Loaded Adaptive Thresholds from {th_path}.")
        else:
            print(f"[Tracker] Info: cluster_thresholds.npy not found in {path}. Using default.")

    def update_clustering_gpu(self, agent_embeddings, t):
        if t % self.multi_steps != 0 and t != 0:
            return

        if self.centers is None:
            return

        diff = agent_embeddings.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        dists = torch.norm(diff, dim=-1)
        min_dists, labels = torch.min(dists, dim=-1)

        self.current_labels = labels
        self.current_min_dists = min_dists

        if self.cluster_thresholds is not None:
            target_th = self.cluster_thresholds[labels]
            self.current_deviations = min_dists > target_th
        else:
            self.current_deviations = min_dists > self.default_threshold

    def forward(self, obs_tensor, last_a_tensor, hidden_states, t_ep, *args, **kwargs):
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
            last_a_tensor = last_a_tensor.unsqueeze(0)

        B, N, _ = obs_tensor.shape

        if self.use_recl:
            if t_ep == 0:
                self.recl_hidden = self.recl_net.agent_embedding_net.fc1.weight.new(
                    B, N, self.args.agent_embedding_dim).zero_()

            with torch.no_grad():
                agent_embed, self.recl_hidden = self.recl_net.agent_embedding_net(
                    obs_tensor, last_a_tensor, self.recl_hidden
                )
                role_embed = self.recl_net.role_embedding_net(agent_embed)

            if self.save_embeddings:
                self.collected_embeddings.append(agent_embed.detach().cpu().numpy())

            self.update_clustering_gpu(agent_embed, t_ep)

            if self.current_labels is None:
                return torch.zeros(B, N, N, self.args.n_actions).to(self.device), hidden_states
        else:
            self.current_labels = torch.zeros(B, N, dtype=torch.long, device=self.device)
            self.current_deviations = torch.zeros(B, N, dtype=torch.bool, device=self.device)

        if not self.use_stage2:
            q_out = torch.zeros(B, N, N, self.args.n_actions).to(self.device)
            return q_out, hidden_states

        obs_i = obs_tensor.unsqueeze(2).expand(B, N, N, -1)

        eye = torch.eye(N, device=self.device)
        id_i = eye.unsqueeze(0).unsqueeze(2).expand(B, N, N, -1)
        id_j = eye.unsqueeze(0).unsqueeze(1).expand(B, N, N, -1)

        parts = [obs_i]

        if self.use_tf:
            act_j = last_a_tensor.unsqueeze(1).expand(B, N, N, -1)
            parts.append(act_j)

        if self.use_role_input:
            role_j = role_embed.unsqueeze(1).expand(B, N, N, -1)
            parts.append(role_j)

        parts.extend([id_i, id_j])
        inputs = torch.cat(parts, dim=-1)

        inputs_flat = inputs.reshape(-1, inputs.shape[-1])
        if hidden_states.shape[0] != B: hidden_states = hidden_states.expand(B, -1, -1, -1)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        with torch.no_grad():
            q_flat, h_flat = self.tracker_net(inputs_flat, hidden_flat)

        q_out = q_flat.reshape(B, N, N, -1)
        h_out = h_flat.reshape(B, N, N, -1)

        return q_out, h_out


    def get_detected_victims(self, avail_actions=None, t_ep=0, attack_start_t=0):
        """
        Return the list of agents currently judged as abnormal, based on the
        accumulated detection metric.

        Key points:
        1. Uses self.defense_th (the defense-specific threshold), decoupled from
           self.TH of Part 1.
        2. Singleton-cluster fallback: if agent j has no alive same-role observer,
           fall back to using all observers.

        Detection strategies (controlled by self.detect_strategy):
          "same_role_only"  -- vote with same-role observers only; singletons fall back to all
          "any_observer"    -- trigger if any same-role observer flags an anomaly; singletons fall back to all
          "role_weighted"   -- role-weighted voting (legacy scheme)

        Returns:
            detected: list of int, indices of agents detected as abnormal
        """
        detected = []
        N = self.n_tracker_agents

        if self.defense_delay >= 0:
            if t_ep < attack_start_t + self.defense_delay:
                return []

        if self.t < self.detect_min_steps:
            return []

        labels = None
        if self.current_labels is not None:
            labels = self.current_labels.squeeze(0).cpu().numpy()

        alive_mask = np.ones(N, dtype=bool)
        if avail_actions is not None:
            for i in range(N):
                if np.sum(avail_actions[i]) <= 1:
                    alive_mask[i] = False

        for j in range(N):
            if not alive_mask[j]:
                continue

            same_role_alive = 0
            if labels is not None:
                for i in range(N):
                    if i != j and alive_mask[i] and labels[i] == labels[j]:
                        same_role_alive += 1

            is_singleton = (same_role_alive == 0)

            if self.detect_strategy == "same_role_only":
                abnormal_count = 0
                total_count = 0
                for i in range(N):
                    if i == j or not alive_mask[i]:
                        continue
                    if not is_singleton:
                        if labels is not None and labels[i] != labels[j]:
                            continue
                    total_count += 1
                    if self.normality_metric[self.defense_window_idx, i, j] < self.defense_th:
                        abnormal_count += 1
                if total_count > 0 and abnormal_count > total_count / 2.0:
                    detected.append(j)

            elif self.detect_strategy == "any_observer":
                for i in range(N):
                    if i == j or not alive_mask[i]:
                        continue
                    if not is_singleton:
                        if labels is not None and labels[i] != labels[j]:
                            continue
                    if self.normality_metric[self.defense_window_idx, i, j] < self.defense_th:
                        detected.append(j)
                        break

            elif self.detect_strategy == "role_weighted":
                abnormal_weight = 0.0
                total_weight = 0.0
                for i in range(N):
                    if i == j or not alive_mask[i]:
                        continue
                    if labels is not None and labels[i] == labels[j]:
                        w = self.w_same
                    else:
                        w = self.w_diff
                    total_weight += w
                    if self.normality_metric[self.defense_window_idx, i, j] < self.defense_th:
                        abnormal_weight += w
                if total_weight > 0 and abnormal_weight > total_weight / 2.0:
                    detected.append(j)

            else:
                abnormal_count = 0
                total_count = 0
                for i in range(N):
                    if i == j or not alive_mask[i]:
                        continue
                    if not is_singleton:
                        if labels is not None and labels[i] != labels[j]:
                            continue
                    total_count += 1
                    if self.normality_metric[self.defense_window_idx, i, j] < self.defense_th:
                        abnormal_count += 1
                if total_count > 0 and abnormal_count > total_count / 2.0:
                    detected.append(j)

        return detected

    def get_corrected_action(self, victim_id, q_matrix, avail_actions=None,
                             t_ep=0, attack_start_t=0):
        """
        Generate a corrective action for an agent detected as abnormal.

        Aggregates the action predictions of all normal observers for the victim
        through a voting mechanism.

        Args:
            victim_id: int, index of the agent detected as abnormal
            q_matrix: [B, N, N, n_actions] tensor or [N, N, n_actions] numpy array
            avail_actions: [N, n_actions] numpy array
            t_ep: current time step within the episode (passed to inner get_detected_victims)
            attack_start_t: time step when the attack starts (passed to inner get_detected_victims)

        Returns:
            corrected_action: int, the corrected action index
            consensus_rate: float, vote share of the top action (used for statistics)
        """
        N = self.n_tracker_agents

        if self.vote_mode == "ignore":
            if avail_actions is not None:
                valid = np.where(avail_actions[victim_id] > 0)[0]
                if len(valid) > 0:
                    return int(valid[-1]), 1.0
            return 0, 1.0
        elif self.vote_mode == "random":
            if avail_actions is not None:
                valid = np.where(avail_actions[victim_id] > 0)[0]
                if len(valid) > 0:
                    return int(np.random.choice(valid)), 0.0
            return 0, 0.0


        detected_set = set(self.get_detected_victims(
            avail_actions=avail_actions,
            t_ep=t_ep,
            attack_start_t=attack_start_t
        ))

        normal_ids = []
        for i in range(N):
            if i == victim_id:
                continue
            if i in detected_set:
                continue
            if avail_actions is not None and np.sum(avail_actions[i]) <= 1:
                continue
            normal_ids.append(i)

        if len(normal_ids) == 0:
            if avail_actions is not None:
                avail = avail_actions[victim_id]
                valid_actions = np.where(avail > 0)[0]
                if len(valid_actions) > 0:
                    return int(valid_actions[-1]), 0.0
            return 0, 0.0

        if isinstance(q_matrix, torch.Tensor):
            q_np = q_matrix.squeeze(0).cpu().numpy()
        else:
            q_np = q_matrix

        labels = None
        if self.current_labels is not None:
            labels = self.current_labels.squeeze(0).cpu().numpy()

        if self.vote_mode == "hard":
            action, consensus = self._hard_vote(q_np, victim_id, normal_ids, avail_actions)
        elif self.vote_mode == "role_hard":
            action, consensus = self._role_hard_vote(q_np, victim_id, normal_ids, labels, avail_actions)
        elif self.vote_mode == "soft":
            action, consensus = self._soft_vote(q_np, victim_id, normal_ids, avail_actions)
        elif self.vote_mode == "role_soft":
            action, consensus = self._role_soft_vote(q_np, victim_id, normal_ids, labels, avail_actions)
        else:
            action, consensus = self._role_soft_vote(q_np, victim_id, normal_ids, labels, avail_actions)

        if avail_actions is not None:
            if avail_actions[victim_id][action] == 0:
                valid_actions = np.where(avail_actions[victim_id] > 0)[0]
                if len(valid_actions) > 0:
                    action = int(valid_actions[0])

        return action, consensus

    def _hard_vote(self, q_np, victim_id, normal_ids, avail_actions=None):
        """Hard voting without role weighting"""
        n_actions = q_np.shape[-1]
        votes = np.zeros(n_actions)
        for i in normal_ids:
            q_ij = q_np[i, victim_id].copy()
            if avail_actions is not None:
                q_ij[avail_actions[victim_id] == 0] = -float('inf')
            best_a = np.argmax(q_ij)
            votes[best_a] += 1.0

        total = np.sum(votes)
        consensus = np.max(votes) / total if total > 0 else 0.0
        return int(np.argmax(votes)), consensus

    def _role_hard_vote(self, q_np, victim_id, normal_ids, labels, avail_actions=None):
        """Role-weighted hard voting"""
        n_actions = q_np.shape[-1]
        votes = np.zeros(n_actions)
        victim_label = labels[victim_id] if labels is not None else -1

        for i in normal_ids:
            q_ij = q_np[i, victim_id].copy()
            if avail_actions is not None:
                q_ij[avail_actions[victim_id] == 0] = -float('inf')
            best_a = np.argmax(q_ij)
            w = self.w_same if (labels is not None and labels[i] == victim_label) else self.w_diff
            votes[best_a] += w

        total = np.sum(votes)
        if total < 1e-12:
            votes = np.zeros(n_actions)
            for i in normal_ids:
                q_ij = q_np[i, victim_id].copy()
                if avail_actions is not None:
                    q_ij[avail_actions[victim_id] == 0] = -float('inf')
                best_a = np.argmax(q_ij)
                votes[best_a] += 1.0
            total = np.sum(votes)
        consensus = np.max(votes) / total if total > 0 else 0.0
        return int(np.argmax(votes)), consensus

    def _soft_vote(self, q_np, victim_id, normal_ids, avail_actions=None):
        """Soft voting without role weighting"""
        n_actions = q_np.shape[-1]
        agg = np.zeros(n_actions)
        tau = self.softmax_temperature

        for i in normal_ids:
            q_ij = q_np[i, victim_id].copy()
            if avail_actions is not None:
                q_ij[avail_actions[victim_id] == 0] = -1e10
            q_shifted = q_ij - np.max(q_ij)
            exp_q = np.exp(q_shifted / tau)
            prob = exp_q / (np.sum(exp_q) + 1e-8)
            agg += prob

        agg /= max(len(normal_ids), 1)
        consensus = np.max(agg)
        return int(np.argmax(agg)), consensus

    def _role_soft_vote(self, q_np, victim_id, normal_ids, labels, avail_actions=None):
        """Role-aware weighted soft voting (recommended main method D3)"""
        n_actions = q_np.shape[-1]
        agg = np.zeros(n_actions)
        total_w = 0.0
        tau = self.softmax_temperature
        victim_label = labels[victim_id] if labels is not None else -1

        for i in normal_ids:
            w = self.w_same if (labels is not None and labels[i] == victim_label) else self.w_diff
            q_ij = q_np[i, victim_id].copy()
            if avail_actions is not None:
                q_ij[avail_actions[victim_id] == 0] = -1e10
            q_shifted = q_ij - np.max(q_ij)
            exp_q = np.exp(q_shifted / tau)
            prob = exp_q / (np.sum(exp_q) + 1e-8)
            agg += w * prob
            total_w += w

        if total_w < 1e-12:
            agg = np.zeros(n_actions)
            for i in normal_ids:
                q_ij = q_np[i, victim_id].copy()
                if avail_actions is not None:
                    q_ij[avail_actions[victim_id] == 0] = -1e10
                q_shifted = q_ij - np.max(q_ij)
                exp_q = np.exp(q_shifted / tau)
                prob = exp_q / (np.sum(exp_q) + 1e-8)
                agg += prob
            agg /= max(len(normal_ids), 1)
        else:
            agg /= total_w
        consensus = np.max(agg)
        return int(np.argmax(agg)), consensus


    def reset_defense_stats(self):
        """Reset defense statistics at the start of each episode"""
        self._ep_corrected_count = 0
        self._ep_total_steps = 0
        self._ep_defense_steps = 0
        self._ep_consensus_values = []
        self._ep_detected_per_step = []
        self._ep_defense_log = []
        self._ep_accuracy_hits = []

    def log_defense_step(self, t_ep, detected_victims, corrections):
        """
        Log the details of a single defense step.

        Args:
            t_ep: int, current time step
            detected_victims: list of int, abnormal agents detected at this step
            corrections: list of dict, details of each corrective action
                         [{"victim": int, "corrected_action": int, "consensus": float}, ...]
        """
        self._ep_detected_per_step.append(len(detected_victims))

        for c in corrections:
            self._ep_consensus_values.append(c["consensus"])

        if self.save_detailed_stats:
            self._ep_defense_log.append({
                "t": t_ep,
                "detected": detected_victims,
                "corrections": corrections,
            })

    def record_defense_episode(self):
        """Summarize defense statistics at the end of each episode"""
        if self._ep_total_steps > 0:
            activation_ratio = self._ep_defense_steps / self._ep_total_steps
        else:
            activation_ratio = 0.0

        avg_consensus = (np.mean(self._ep_consensus_values)
                         if len(self._ep_consensus_values) > 0 else 0.0)

        avg_detected = (np.mean(self._ep_detected_per_step)
                        if len(self._ep_detected_per_step) > 0 else 0.0)

        avg_accuracy = (np.mean(self._ep_accuracy_hits)
                        if len(self._ep_accuracy_hits) > 0 else 0.0)

        summary = {
            "corrected_count": self._ep_corrected_count,
            "defense_steps": self._ep_defense_steps,
            "total_steps": self._ep_total_steps,
            "activation_ratio": round(activation_ratio, 4),
            "avg_consensus": round(float(avg_consensus), 4),
            "avg_accuracy": round(float(avg_accuracy), 4),
            "avg_detected_per_step": round(float(avg_detected), 4),
        }
        if self.save_detailed_stats:
            summary["defense_log"] = self._ep_defense_log
        self.defense_stats["ep_summaries"].append(summary)


    def output_statistics(self, q_matrix, all_real_actions, avail_actions, reset=False):
        if reset:
            if self.save_detailed_stats:
                self.out_dict["t_detect"].append(self.t_detect.copy())
                self.out_dict["scores"].append(copy.deepcopy(self.episode_scores_history))
            if self.defense_active:
                self.record_defense_episode()
                ep_summaries = self.defense_stats["ep_summaries"]
                self.out_dict["defense_stats"].append(
                    copy.deepcopy(ep_summaries[-1]) if ep_summaries else {}
                )
            self.normality_metric.fill(0)
            self.normality_score.fill(0)
            self.t = 0
            self.t_detect.fill(1000)
            self.episode_scores_history = []
            if self.defense_active:
                self.reset_defense_stats()
            return

        self.t += 1
        self._ep_total_steps = self.t

        if isinstance(q_matrix, torch.Tensor):
            q_np = q_matrix.squeeze(0).cpu().numpy()
        else:
            q_np = q_matrix

        if self.current_labels is not None:
            curr_dev = self.current_deviations.squeeze(0).cpu().numpy()
            curr_lab = self.current_labels.squeeze(0).cpu().numpy()
        else:
            return

        singleton_mask = np.zeros(self.n_tracker_agents, dtype=bool)
        for j in range(self.n_tracker_agents):
            if np.sum(avail_actions[j]) <= 1:
                continue
            same_count = 0
            for i in range(self.n_tracker_agents):
                if i != j and np.sum(avail_actions[i]) > 1 and curr_lab[i] == curr_lab[j]:
                    same_count += 1
            if same_count == 0:
                singleton_mask[j] = True

        for i in range(self.n_tracker_agents):
            for j in range(self.n_tracker_agents):
                is_dead = (np.sum(avail_actions[j]) <= 1)

                if is_dead:
                    score = 0.0
                elif self.use_stage1 and self.use_stage2:
                    if curr_dev[j]:
                        score = -100.0
                    elif curr_lab[i] == curr_lab[j]:
                        q_ij = q_np[i, j]
                        m = np.max(q_ij)
                        real_act = all_real_actions[j]
                        score = float(q_ij[real_act] - m)
                    elif singleton_mask[j]:
                        q_ij = q_np[i, j]
                        m = np.max(q_ij)
                        real_act = all_real_actions[j]
                        score = float(q_ij[real_act] - m)
                    else:
                        score = 0.0

                elif self.use_stage1 and not self.use_stage2:
                    if self.current_min_dists is not None:
                        dist_j = self.current_min_dists.squeeze(0).cpu().numpy()[j]
                        score = -float(dist_j)
                    else:
                        score = 0.0

                elif not self.use_stage1 and self.use_stage2:
                    if curr_lab[i] == curr_lab[j]:
                        q_ij = q_np[i, j]
                        m = np.max(q_ij)
                        real_act = all_real_actions[j]
                        score = float(q_ij[real_act] - m)
                    elif singleton_mask[j]:
                        q_ij = q_np[i, j]
                        m = np.max(q_ij)
                        real_act = all_real_actions[j]
                        score = float(q_ij[real_act] - m)
                    else:
                        score = 0.0
                else:
                    score = 0.0

                self.normality_score[i][j] = score

                for w_idx, ws in enumerate(self.windows):
                    curr = self.normality_metric[w_idx, i, j]
                    if ws == -1:
                        metric = (curr * (self.t - 1) + score) / self.t
                    else:
                        alpha = 1.0 / ws; metric = (1 - alpha) * curr + alpha * score
                    self.normality_metric[w_idx, i, j] = metric
                    for th_idx, th_val in enumerate(self.TH):
                        if metric < th_val and self.t_detect[w_idx, th_idx, i, j] > 999:
                            self.t_detect[w_idx, th_idx, i, j] = self.t

        if self.save_detailed_stats:
            self.episode_scores_history.append(self.normality_metric.copy())

        if self.defense_calibrate and self.t >= self.detect_min_steps:
            w_idx = self.defense_window_idx
            for i in range(self.n_tracker_agents):
                for j in range(self.n_tracker_agents):
                    if i == j:
                        continue
                    if np.sum(avail_actions[i]) <= 1 or np.sum(avail_actions[j]) <= 1:
                        continue
                    if curr_lab[i] == curr_lab[j] or singleton_mask[j]:
                        self._calibration_metrics.append(
                            float(self.normality_metric[w_idx, i, j])
                        )

    def save_stats(self, path):
        if self.save_detailed_stats:
            try:
                def convert(o):
                    if isinstance(o, np.int64): return int(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    if isinstance(o, np.float64): return float(o)
                    if isinstance(o, np.float32): return float(o)
                    return o

                with open(path, "w") as f:
                    json.dump(self.out_dict, f, default=convert)
                print(f"[Tracker] Stats saved to {path}")
            except Exception as e:
                print(f"[Tracker] Error saving stats: {e}")

            if self.save_embeddings and self.collected_embeddings and len(self.collected_embeddings) > 0:
                embed_dir = os.path.dirname(path)
                embed_path = os.path.join(embed_dir, "agent_embeddings.npy")
                all_embeds = np.concatenate(self.collected_embeddings, axis=0)
                np.save(embed_path, all_embeds)
                print(f"[Tracker] Saved {all_embeds.shape} embeddings to {embed_path}")
        else:
            print(f"[Tracker] save_detailed_stats=False, skipping JSON/embedding save.")

        if self.defense_calibrate:
            self.print_calibration_stats(path)

    def print_calibration_stats(self, save_path=None):
        """
        Output threshold-calibration suggestions based on the collected normal
        EMA metric data.

        Usage:
            1. Run B0 (no attack): defense_calibrate=True defense_active=False
            2. Inspect the printed percentile statistics
            3. Choose defense_th = a value slightly below the 1st percentile
               -> under normal conditions <1% of observations fall below it -> almost no false positives
               -> under attack the metric drops sharply -> easily exceeds it
        """
        data = np.array(self._calibration_metrics)
        if len(data) == 0:
            print("[Calibration] No calibration data (need to run a B0 no-attack episode)")
            return

        print("\n" + "=" * 70)
        print("  [Calibration] EMA Normality Metric distribution (normal episode)")
        print("=" * 70)
        print(f"  count: {len(data)} (i->j) observations")
        print(f"  window: index={self.defense_window_idx} "
              f"(window={self.windows[self.defense_window_idx]})")
        print(f"  mean: {np.mean(data):.4f}")
        print(f"  std: {np.std(data):.4f}")
        print(f"  min: {np.min(data):.4f}")
        print(f"  max: {np.max(data):.4f}")
        print(f"  ---")
        percentiles = [1, 2, 3, 5, 10, 25, 50]
        for p in percentiles:
            val = np.percentile(data, p)
            print(f"  {p:>3}th percentile: {val:.4f}")
        print(f"  ---")
        print(f"  suggested defense_th:")
        p1 = np.percentile(data, 1)
        p3 = np.percentile(data, 3)
        print(f"    conservative (low false positives): {p1:.2f}  (< 1st percentile)")
        print(f"    balanced:          {p3:.2f}  (< 3rd percentile)")
        print(f"    current setting:      {self.defense_th}")
        print("=" * 70 + "\n")

        if save_path is not None:
            cal_path = save_path.replace(".json", "_calibration.json")
            cal_data = {
                "n_samples": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "percentiles": {str(p): float(np.percentile(data, p))
                                for p in [1, 2, 3, 5, 10, 25, 50, 75, 90, 95, 99]},
                "window_idx": self.defense_window_idx,
                "window_size": self.windows[self.defense_window_idx],
            }
            try:
                with open(cal_path, "w") as f:
                    json.dump(cal_data, f, indent=2)
                print(f"[Calibration] Calibration data saved to {cal_path}")
            except Exception as e:
                print(f"[Calibration] Save failed: {e}")

    def train(self, logger, t_env):
        if self.buffer.num_episodes < self.args.batch_size:
            return

        self.tracker_net.train()
        batch = self.buffer.sample()
        bs = batch.o.shape[0]
        max_t = batch.o.shape[1] - 1

        N = self.n_tracker_agents
        n_pairs = self.n_pairs
        sid = self.single_input_dim
        dev = self.device

        all_pair_obs = batch.o[:, :max_t].reshape(bs, max_t, n_pairs, sid).to(dev)

        obs_dim = self.input_shape
        obs_i_all = all_pair_obs[..., :obs_dim]

        idx = obs_dim
        if self.use_role_input:
            role_dim = self.args.role_embedding_dim
            role_j_all = all_pair_obs[..., idx:idx + role_dim]
            idx += role_dim

        id_i_all = all_pair_obs[..., idx:idx + N]
        idx += N
        id_j_all = all_pair_obs[..., idx:idx + N]

        parts = [obs_i_all]

        if self.use_tf:
            pair_j_dev = self.pair_j_indices.to(dev)
            last_a_n = batch.last_onehot_a_n[:, :max_t].to(dev)
            last_a_j_all = last_a_n[:, :, pair_j_dev, :]
            parts.append(last_a_j_all)

        if self.use_role_input:
            parts.append(role_j_all)

        parts.extend([id_i_all, id_j_all])
        full_input = torch.cat(parts, dim=-1)

        target_actions = batch.a[:, :max_t].long().to(dev)
        mask = batch.m[:, :max_t, 0].float().to(dev)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, n_pairs)

        hidden = self.tracker_net.init_hidden().expand(bs * n_pairs, -1).contiguous()

        total_loss = torch.tensor(0.0, device=dev)
        total_count = torch.tensor(0.0, device=dev)

        for t in range(max_t):
            input_t = full_input[:, t].reshape(bs * n_pairs, -1)
            mask_t = mask_expanded[:, t].reshape(bs * n_pairs)

            if mask_t.sum() > 0:
                pred_q, hidden = self.tracker_net(input_t, hidden)
                hidden = hidden.detach()

                target_t = target_actions[:, t].reshape(bs * n_pairs)
                loss_t = self.loss_fn(pred_q, target_t)
                total_loss += (loss_t * mask_t).sum()
                total_count += mask_t.sum()

        if total_count > 0:
            final_loss = total_loss / total_count
            self.optimiser.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tracker_net.parameters(), self.args.grad_norm_clip)
            self.optimiser.step()
            self.training_steps += 1

            if self.training_steps % 10 == 0:
                logger.log_stat("tracker_loss", final_loss.item(), self.training_steps)

            if self.training_steps % 100 == 0:
                logger.console_logger.info(
                    f"[Tracker] training_step={self.training_steps}, loss={final_loss.item():.4f}"
                )

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.tracker_net.state_dict(), f"{path}/tracker_agent.th")
        torch.save(self.optimiser.state_dict(), f"{path}/tracker_opt.th")
        print(f"[Tracker] Saved model to {path}")
