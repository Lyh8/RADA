import copy
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from sklearn.cluster import MiniBatchKMeans

from harl.algorithms.trackers.tracker_net import TrackerAgentContinuous


ABLATION_FULL = "full"
ABLATION_NO_ROLE = "no_role"
ABLATION_NO_TF = "no_tf"
ABLATION_NO_ROLE_INPUT = "no_role_input"

VALID_ABLATION_VARIANTS = {ABLATION_FULL, ABLATION_NO_ROLE, ABLATION_NO_TF, ABLATION_NO_ROLE_INPUT}


class DecentralizedTracker:

    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        role_embedding_dim,
        hidden_dim=128,
        cluster_num=2,
        device=torch.device("cpu"),
        tracker_lr=5e-4,
        grad_norm_clip=10.0,
        train_epochs=5,
        mini_batch_size=32,
        thresholds=None,
        windows=None,
        role_dev_threshold=15.0,
        ablation_variant="full",
    ):
        assert ablation_variant in VALID_ABLATION_VARIANTS, \
            f"Invalid ablation_variant='{ablation_variant}', must be one of {VALID_ABLATION_VARIANTS}"
        self.ablation_variant = ablation_variant

        self.use_role_input = ablation_variant not in {ABLATION_NO_ROLE, ABLATION_NO_ROLE_INPUT}
        self.use_tf = ablation_variant != ABLATION_NO_TF
        self.use_stage1 = ablation_variant != ABLATION_NO_ROLE

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.role_embedding_dim = role_embedding_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = min(cluster_num, num_agents)
        self.device = device
        self.grad_norm_clip = grad_norm_clip
        self.train_epochs = train_epochs
        self.mini_batch_size = mini_batch_size
        self.role_dev_threshold = role_dev_threshold

        self.thresholds = thresholds if thresholds else [-9.5, -11.0, -12.5, -15.0]
        self.windows = windows if windows else [-1, 5, 10, 20]

        tracker_input_dim = (
            obs_dim +
            action_dim +
            role_embedding_dim +
            num_agents * 2
        )

        self.tracker_net = TrackerAgentContinuous(
            input_dim=tracker_input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(device)

        self.optimizer = Adam(self.tracker_net.parameters(), lr=tracker_lr, eps=1e-5)

        self.obs_mean = None
        self.obs_var = None

        self.cluster_centers = None
        self.cluster_thresholds = None

        self.hidden = None
        self.current_labels = None
        self.current_deviations = None

        self.training_steps = 0
        self.training_history = {
            'nll_loss': [],
            'mean_std': [],
            'grad_norm': [],
        }

        self.reset_detection_stats()

        self.episode_logs = {
            'rewards': [],
            't_detect': [],
            'scores_history': [],
            'episode_lengths': [],
            'attacked_agents': [],
        }

        print(f"[Tracker] Initialized: obs={obs_dim}, act={action_dim}, "
              f"role={role_embedding_dim}, hidden={hidden_dim}, agents={num_agents}")
        print(f"[Tracker] Ablation variant: {ablation_variant} "
              f"(role_input={self.use_role_input}, tf={self.use_tf}, stage1={self.use_stage1})")


    def init_hidden(self, batch_size=1):
        h = self.tracker_net.init_hidden(1)
        self.hidden = h.new_zeros(batch_size, self.num_agents, self.num_agents, self.hidden_dim)
        return self.hidden

    def reset_detection_stats(self):
        N = self.num_agents
        W = len(self.windows)
        T = len(self.thresholds)

        self.raw_scores = np.zeros((N, N))

        self.smoothed_scores = np.zeros((W, N, N))

        self.t_detect = 1e6 * np.ones((W, T, N, N))

        self.t = 0

        self.episode_score_history = []

        self.detected = np.zeros((N, N), dtype=bool)


    def load_normalization(self, obs_mean, obs_var):
        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_var = torch.tensor(obs_var, dtype=torch.float32, device=self.device)
        print(f"[Tracker] Loaded observation normalization stats")

    def load_cluster_info(self, centers, thresholds=None):
        self.cluster_centers = torch.tensor(centers, dtype=torch.float32, device=self.device)
        if thresholds is not None:
            self.cluster_thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
        else:
            self.cluster_thresholds = torch.full(
                (self.cluster_num,), self.role_dev_threshold, device=self.device
            )
        print(f"[Tracker] Loaded {self.cluster_num} cluster centers")

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.tracker_net.state_dict(), os.path.join(path, "tracker_net.pt"))

        torch.save(self.optimizer.state_dict(), os.path.join(path, "tracker_opt.pt"))

        if self.obs_mean is not None:
            torch.save({
                'mean': self.obs_mean,
                'var': self.obs_var
            }, os.path.join(path, "tracker_obs_norm.pt"))

        if self.cluster_centers is not None:
            torch.save({
                'centers': self.cluster_centers,
                'thresholds': self.cluster_thresholds
            }, os.path.join(path, "tracker_clusters.pt"))

        with open(os.path.join(path, "tracker_training_history.json"), 'w') as f:
            json.dump(self.training_history, f, indent=2)

        with open(os.path.join(path, "tracker_ablation_config.json"), 'w') as f:
            json.dump({
                'ablation_variant': self.ablation_variant,
                'use_role_input': self.use_role_input,
                'use_tf': self.use_tf,
                'use_stage1': self.use_stage1,
            }, f, indent=2)

        print(f"[Tracker] Saved to {path}")

    def load(self, path):
        net_path = os.path.join(path, "tracker_net.pt")
        if os.path.exists(net_path):
            self.tracker_net.load_state_dict(
                torch.load(net_path, map_location=self.device)
            )
            print(f"[Tracker] Loaded network from {net_path}")

        opt_path = os.path.join(path, "tracker_opt.pt")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device)
            )

        norm_path = os.path.join(path, "tracker_obs_norm.pt")
        if os.path.exists(norm_path):
            data = torch.load(norm_path, map_location=self.device)
            self.obs_mean = data['mean']
            self.obs_var = data['var']
            print(f"[Tracker] Loaded normalization stats")

        cluster_path = os.path.join(path, "tracker_clusters.pt")
        if os.path.exists(cluster_path):
            data = torch.load(cluster_path, map_location=self.device)
            self.cluster_centers = data['centers']
            self.cluster_thresholds = data['thresholds']
            print(f"[Tracker] Loaded cluster info")


    def _normalize_obs(self, obs):
        if self.obs_mean is not None and self.obs_var is not None:
            return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-6)
        return obs


    def _ablation_mask_inputs(self, role_j, act_j):
        if not self.use_role_input:
            role_j = torch.zeros_like(role_j)
        if not self.use_tf:
            act_j = torch.zeros_like(act_j)
        return role_j, act_j


    def _compute_cluster_assignments(self, role_embeddings):
        if not self.use_stage1:
            self.current_labels = None
            self.current_deviations = None
            return

        if self.cluster_centers is None:
            self.current_labels = None
            self.current_deviations = None
            return

        B, N, D = role_embeddings.shape

        flat_embeds = role_embeddings.reshape(-1, D)

        diffs = flat_embeds.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)

        min_dists, labels = torch.min(dists, dim=-1)

        self.current_labels = labels.reshape(B, N)

        assigned_thresholds = self.cluster_thresholds[labels]
        is_deviated = min_dists > assigned_thresholds
        self.current_deviations = is_deviated.reshape(B, N)


    @torch.no_grad()
    def forward(self, obs, last_action, role_embedding, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.hidden

        B, N = obs.shape[0], obs.shape[1]

        norm_obs = self._normalize_obs(obs)

        self._compute_cluster_assignments(role_embedding)

        obs_i = norm_obs.unsqueeze(2).expand(B, N, N, -1)

        role_j = role_embedding.unsqueeze(1).expand(B, N, N, -1)

        act_j = last_action.unsqueeze(1).expand(B, N, N, -1)

        role_j, act_j = self._ablation_mask_inputs(role_j, act_j)

        eye = torch.eye(N, device=self.device)
        id_i = eye.view(1, N, 1, N).expand(B, -1, N, -1)
        id_j = eye.view(1, 1, N, N).expand(B, N, -1, -1)

        inputs = torch.cat([obs_i, act_j, role_j, id_i, id_j], dim=-1)

        inputs_flat = inputs.reshape(-1, inputs.shape[-1])
        hidden_flat = hidden_states.reshape(-1, self.hidden_dim)

        mu_flat, std_flat, h_flat = self.tracker_net(inputs_flat, hidden_flat)

        mu = mu_flat.reshape(B, N, N, -1)
        std = std_flat.reshape(B, N, N, -1)
        self.hidden = h_flat.reshape(B, N, N, -1)

        return mu, std, self.hidden


    def compute_scores(self, mu, std, real_actions, active_masks=None, use_grouping=True):
        B, N = mu.shape[0], mu.shape[1]

        actions_expanded = real_actions.unsqueeze(1).expand(B, N, N, -1)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions_expanded).sum(dim=-1)
        scores = log_probs.cpu().numpy()

        if use_grouping and self.current_labels is not None:
            labels = self.current_labels.cpu().numpy()
            deviations = self.current_deviations.cpu().numpy() if self.current_deviations is not None else None

            dev_distances = None
            if hasattr(self, 'current_distances') and self.current_distances is not None:
                dev_distances = self.current_distances.cpu().numpy()

            for b in range(B):
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            scores[b, i, j] = np.nan
                            continue

                        same_group = (labels[b, i] == labels[b, j])
                        j_deviated = deviations[b, j] if deviations is not None else False

                        if j_deviated:
                            scores[b, i, j] = -500.0
                        elif same_group:
                            pass
                        else:
                            scores[b, i, j] = np.nan
        else:
            for b in range(B):
                for i in range(N):
                    scores[b, i, i] = 0.0

        return scores


    def update_detection_stats(self, scores):
        N = self.num_agents
        self.raw_scores = scores
        self.t += 1

        for w_idx, window in enumerate(self.windows):
            if window == -1:
                self.smoothed_scores[w_idx] = (
                    self.smoothed_scores[w_idx] * (self.t - 1) + scores
                ) / self.t
            else:
                alpha = 2.0 / (window + 1)
                self.smoothed_scores[w_idx] = (
                    alpha * scores + (1 - alpha) * self.smoothed_scores[w_idx]
                )

            for th_idx, threshold in enumerate(self.thresholds):
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        if (self.t_detect[w_idx, th_idx, i, j] >= 1e5 and
                                self.smoothed_scores[w_idx, i, j] < threshold):
                            self.t_detect[w_idx, th_idx, i, j] = self.t

        self.episode_score_history.append(scores.copy())

    def get_detection_result(self, w_idx, th_idx):
        return {
            't_detect': self.t_detect[w_idx, th_idx].copy(),
            'smoothed_scores': self.smoothed_scores[w_idx].copy(),
        }


    def train_step(self, batch_obs, batch_actions, batch_role_embeddings, batch_masks=None):

        if self.training_steps == 0:
            print(f"[Tracker Train] batch_obs: {batch_obs.shape}")
            print(f"[Tracker Train] batch_actions: {batch_actions.shape}")
            print(f"[Tracker Train] batch_role_embeddings: {batch_role_embeddings.shape}")
            print(f"[Tracker Train] role_embed sample: mean={batch_role_embeddings[0, 0, 0, :5].mean():.4f}")
            print(f"[Tracker Train] cluster_centers loaded: {self.cluster_centers is not None}")
            print(f"[Tracker Train] ablation_variant: {self.ablation_variant}")
        self.tracker_net.train()

        B, T, N = batch_obs.shape[:3]
        num_epochs = self.train_epochs
        mini_batch_size = min(self.mini_batch_size, B)

        obs_t = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(batch_actions, dtype=torch.float32, device=self.device)
        role_t = torch.tensor(batch_role_embeddings, dtype=torch.float32, device=self.device)

        if batch_masks is not None:
            masks_t = torch.tensor(batch_masks, dtype=torch.float32, device=self.device)
        else:
            masks_t = torch.ones(B, T, N, device=self.device)

        obs_t = self._normalize_obs(obs_t)

        eye = torch.eye(N, device=self.device)
        pair_mask_base = 1 - eye

        all_losses = []
        all_stds = []
        all_grads = []

        for epoch in range(num_epochs):
            indices = np.random.permutation(B)

            for start in range(0, B, mini_batch_size):
                end = min(start + mini_batch_size, B)
                mb_idx = indices[start:end]
                mb_B = len(mb_idx)

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_role = role_t[mb_idx]
                mb_mask = masks_t[mb_idx]

                hidden = torch.zeros(mb_B, N, N, self.hidden_dim, device=self.device)

                total_loss = 0.0
                total_count = 0

                for t in range(1, T):
                    obs_curr = mb_obs[:, t]
                    act_prev = mb_act[:, t-1]
                    act_curr = mb_act[:, t]
                    role_curr = mb_role[:, t]
                    mask_curr = mb_mask[:, t]

                    obs_i = obs_curr.unsqueeze(2).expand(mb_B, N, N, -1)
                    role_j = role_curr.unsqueeze(1).expand(mb_B, N, N, -1)
                    act_j_prev = act_prev.unsqueeze(1).expand(mb_B, N, N, -1)
                    id_i = eye.view(1, N, 1, N).expand(mb_B, -1, N, -1)
                    id_j = eye.view(1, 1, N, N).expand(mb_B, N, -1, -1)

                    role_j, act_j_prev = self._ablation_mask_inputs(role_j, act_j_prev)

                    inputs = torch.cat([obs_i, act_j_prev, role_j, id_i, id_j], dim=-1)

                    inputs_flat = inputs.reshape(mb_B * N * N, -1)
                    hidden_flat = hidden.reshape(mb_B * N * N, -1)

                    mu_flat, std_flat, h_flat = self.tracker_net(inputs_flat, hidden_flat)
                    hidden = h_flat.reshape(mb_B, N, N, -1).detach()

                    mu = mu_flat.reshape(mb_B, N, N, -1)
                    std = std_flat.reshape(mb_B, N, N, -1)

                    act_target = act_curr.unsqueeze(1).expand(mb_B, N, N, -1)
                    dist = Normal(mu, std)
                    nll = -dist.log_prob(act_target).sum(dim=-1)

                    pair_mask = pair_mask_base.unsqueeze(0).expand(mb_B, -1, -1)
                    target_mask = mask_curr.unsqueeze(1).expand(mb_B, N, N)
                    final_mask = pair_mask * target_mask

                    if final_mask.sum() > 0:
                        total_loss += (nll * final_mask).sum()
                        total_count += final_mask.sum()

                if total_count > 0:
                    loss = total_loss / total_count

                    self.optimizer.zero_grad()
                    loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.tracker_net.parameters(), self.grad_norm_clip
                    )

                    self.optimizer.step()
                    self.training_steps += 1

                    all_losses.append(loss.item())
                    all_stds.append(std.mean().item())
                    grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    all_grads.append(grad_val)

        if len(all_losses) > 0:
            avg_loss = float(np.mean(all_losses))
            avg_std = float(np.mean(all_stds))
            avg_grad = float(np.mean(all_grads))

            self.training_history['nll_loss'].append(avg_loss)
            self.training_history['mean_std'].append(avg_std)
            self.training_history['grad_norm'].append(avg_grad)

            return {
                'tracker_nll_loss': avg_loss,
                'tracker_mean_std': avg_std,
                'tracker_grad_norm': avg_grad,
                'tracker_training_steps': self.training_steps,
                'tracker_updates_this_call': len(all_losses),
            }

        return {}


    def end_episode(self, reward=None, episode_length=None, attacked_agent=None):
        self.episode_logs['rewards'].append(reward)
        self.episode_logs['t_detect'].append(self.t_detect.copy())
        self.episode_logs['scores_history'].append(self.episode_score_history.copy())
        self.episode_logs['episode_lengths'].append(episode_length if episode_length else self.t)
        self.episode_logs['attacked_agents'].append(attacked_agent)

        self.reset_detection_stats()

    def save_episode_logs(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        logs_serializable = {}
        for key, value in self.episode_logs.items():
            if isinstance(value, list):
                logs_serializable[key] = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        logs_serializable[key].append(item.tolist())
                    elif isinstance(item, dict):
                        logs_serializable[key].append({
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in item.items()
                        })
                    else:
                        logs_serializable[key].append(item)
            else:
                logs_serializable[key] = value

        with open(path, 'w') as f:
            json.dump(logs_serializable, f, indent=2)

        print(f"[Tracker] Saved episode logs to {path}")


    def prep_training(self):
        self.tracker_net.train()

    def prep_rollout(self):
        self.tracker_net.eval()

    def to(self, device):
        self.device = device
        self.tracker_net.to(device)
        if self.obs_mean is not None:
            self.obs_mean = self.obs_mean.to(device)
            self.obs_var = self.obs_var.to(device)
        if self.cluster_centers is not None:
            self.cluster_centers = self.cluster_centers.to(device)
            self.cluster_thresholds = self.cluster_thresholds.to(device)
        return self
