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

        self.use_shared_predictor = getattr(args, 'use_shared_predictor', True)

        self.multi_steps = getattr(args, 'multi_steps', 10)
        self.train_mode = getattr(args, 'tracker_train', False)

        self.use_stage1 = getattr(args, 'use_stage1', True)
        self.use_stage2 = getattr(args, 'use_stage2', True)

        if self.use_stage1 and not self.use_recl:
            self.use_recl = True
        if not self.use_stage1 and not self.use_stage2:
            self.use_stage1 = True
            self.use_stage2 = True

        self.default_threshold = getattr(args, 'role_dev_threshold', 15.0)

        self.centers = None
        self.cluster_thresholds = None
        self.current_labels = None
        self.current_deviations = None
        self.current_min_dists = None
        self.collected_embeddings = []
        self.collected_role_embeddings = []

        if self.use_recl:
            self.recl_net = RECL_NET(args).to(self.device)
            self.recl_net.eval()

        N = n_tracker_agents
        self.n_pairs = N * (N - 1)
        pair_i_list = []
        pair_j_list = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pair_i_list.append(i)
                pair_j_list.append(j)
        self.pair_i_indices = torch.tensor(pair_i_list, dtype=torch.long)
        self.pair_j_indices = torch.tensor(pair_j_list, dtype=torch.long)

        if self.use_shared_predictor:
            predictor_input_shape = input_shape
            if self.use_tf:
                predictor_input_shape += args.n_actions
            if self.use_role_input:
                predictor_input_shape += args.role_embedding_dim
            predictor_input_shape += n_tracker_agents * 2
        else:
            predictor_input_shape = input_shape + args.n_actions + 1

        target_dim = getattr(args, "tracker_hidden_dim", args.hidden_dim)
        if self.use_shared_predictor:
            self.tracker_net = TrackerAgent(
                predictor_input_shape, args, hidden_dim=target_dim
            ).to(self.device)
            self.tracker_net.eval()
        else:
            self.tracker_nets = torch.nn.ModuleList([
                TrackerAgent(predictor_input_shape, args, hidden_dim=target_dim)
                for _ in range(self.n_pairs)
            ]).to(self.device)
            self.tracker_nets.eval()
            self.tracker_net = self.tracker_nets[0]

        self.single_input_dim = input_shape + N * 2
        if self.use_role_input:
            self.single_input_dim += args.role_embedding_dim
        if self.train_mode:
            if self.use_shared_predictor:
                self.tracker_net.train()
                self.optimiser = Adam(self.tracker_net.parameters(), lr=args.tracker_lr)
            else:
                self.tracker_nets.train()
                self.optimiser = Adam(self.tracker_nets.parameters(), lr=args.tracker_lr)

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
        self.TH = args.thresholds if hasattr(args, 'thresholds') else [-3.0]

        self.normality_score = np.zeros((n_tracker_agents, n_tracker_agents))
        self.normality_metric = np.zeros((len(self.windows), self.n_tracker_agents, self.n_tracker_agents))
        self.t = 0
        self.t_detect = 1000 * np.ones((len(self.windows), len(self.TH), self.n_tracker_agents, self.n_tracker_agents))

        self.out_dict = {
            "attacked": [], "t_start": [], "window_size": self.windows,
            "threshold": self.TH, "t_detect": [], "battle_won": [],
            "ep_length": [], "actual_victim": [], "scores": []
        }
        self.episode_scores_history = []

    def init_hidden(self):
        h = self.tracker_net.init_hidden()
        return h.unsqueeze(1).unsqueeze(1).expand(1, self.n_tracker_agents, self.n_tracker_agents, -1).contiguous()

    def load_recl_weights(self, path):
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
        if self.use_shared_predictor:
            p = os.path.join(path, "tracker_agent.th")
            if os.path.exists(p):
                self.tracker_net.load_state_dict(torch.load(p, map_location=self.device))
                print(f"[Tracker] Loaded Tracker weights from {p}.")
            else:
                print(f"[Tracker] Warning: Tracker weights not found in {path}")
        else:
            p = os.path.join(path, "tracker_agents_v4.th")
            if os.path.exists(p):
                self.tracker_nets.load_state_dict(torch.load(p, map_location=self.device))
                print(f"[Tracker] Loaded V4 Tracker weights from {p}.")
            else:
                print(f"[Tracker] Warning: V4 Tracker weights not found in {path}")

        if self.train_mode:
            opt_p = os.path.join(path, "tracker_opt.th")
            if os.path.exists(opt_p):
                self.optimiser.load_state_dict(torch.load(opt_p, map_location=self.device))
                print(f"[Tracker] Loaded optimizer state from {opt_p}.")

    def load_clustering_info(self, path):
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

            self.collected_embeddings.append(agent_embed.detach().cpu().numpy())
            self.collected_role_embeddings.append(role_embed.detach().cpu().numpy())
            self.update_clustering_gpu(agent_embed, t_ep)

            if self.current_labels is None:
                return torch.zeros(B, N, N, self.args.n_actions).to(self.device), hidden_states
        else:
            self.current_labels = torch.zeros(B, N, dtype=torch.long, device=self.device)
            self.current_deviations = torch.zeros(B, N, dtype=torch.bool, device=self.device)

        if not self.use_stage2:
            q_out = torch.zeros(B, N, N, self.args.n_actions).to(self.device)
            return q_out, hidden_states

        if hidden_states.shape[0] != B:
            hidden_states = hidden_states.expand(B, -1, -1, -1)

        if self.use_shared_predictor:
            obs_i = obs_tensor.unsqueeze(2).expand(B, N, N, -1)
            parts = [obs_i]

            if self.use_tf:
                act_j = last_a_tensor.unsqueeze(1).expand(B, N, N, -1)
                parts.append(act_j)

            if self.use_role_input:
                role_j = role_embed.unsqueeze(1).expand(B, N, N, -1)
                parts.append(role_j)

            eye = torch.eye(N, device=self.device)
            id_i = eye.unsqueeze(0).unsqueeze(2).expand(B, N, N, -1)
            id_j = eye.unsqueeze(0).unsqueeze(1).expand(B, N, N, -1)
            parts.extend([id_i, id_j])
            inputs = torch.cat(parts, dim=-1)

            inputs_flat = inputs.reshape(-1, inputs.shape[-1])
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

            with torch.no_grad():
                q_flat, h_flat = self.tracker_net(inputs_flat, hidden_flat)

            q_out = q_flat.reshape(B, N, N, -1)
            h_out = h_flat.reshape(B, N, N, -1)

        else:
            reward_val = self.last_reward if hasattr(self, 'last_reward') else 0
            reward_tensor = torch.tensor([[reward_val]], dtype=torch.float32, device=self.device)
            reward_tensor = reward_tensor.expand(B, 1)

            q_out = torch.zeros(B, N, N, self.args.n_actions, device=self.device)
            h_out = hidden_states.clone()

            pair_idx = 0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    inp_ij = torch.cat([
                        obs_tensor[:, i, :],
                        last_a_tensor[:, i, :],
                        reward_tensor
                    ], dim=-1)

                    h_ij = hidden_states[:, i, j, :]

                    with torch.no_grad():
                        q_ij, h_ij_new = self.tracker_nets[pair_idx](inp_ij, h_ij)

                    q_out[:, i, j, :] = q_ij
                    h_out[:, i, j, :] = h_ij_new
                    pair_idx += 1

        return q_out, h_out

    def output_statistics(self, q_matrix, all_real_actions, avail_actions, reset=False):
        if reset:
            self.out_dict["t_detect"].append(self.t_detect.copy())
            self.out_dict["scores"].append(copy.deepcopy(self.episode_scores_history))
            self.normality_metric.fill(0)
            self.normality_score.fill(0)
            self.t = 0
            self.t_detect.fill(1000)
            self.episode_scores_history = []
            return

        self.t += 1
        if isinstance(q_matrix, torch.Tensor):
            q_np = q_matrix.squeeze(0).cpu().numpy()
        else:
            q_np = q_matrix

        if self.current_labels is not None:
            curr_dev = self.current_deviations.squeeze(0).cpu().numpy()
            curr_lab = self.current_labels.squeeze(0).cpu().numpy()
        else:
            return

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

        self.episode_scores_history.append(self.normality_metric.copy())

    def save_stats(self, path):
        try:
            def convert(o):
                if isinstance(o, np.int64): return int(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return o
            with open(path, "w") as f:
                json.dump(self.out_dict, f, default=convert)
            print(f"[Tracker] Stats saved to {path}")
        except Exception as e:
            print(f"[Tracker] Error saving stats: {e}")

        if len(self.collected_embeddings) > 0:
            embed_dir = os.path.dirname(path)
            embed_path = os.path.join(embed_dir, "agent_embeddings.npy")
            all_embeds = np.concatenate(self.collected_embeddings, axis=0)
            np.save(embed_path, all_embeds)
            print(f"[Tracker] Saved {all_embeds.shape} embeddings to {embed_path}")

        if len(self.collected_role_embeddings) > 0:
            role_embed_dir = os.path.dirname(path)
            role_embed_path = os.path.join(role_embed_dir, "role_embeddings.npy")
            all_role_embeds = np.concatenate(self.collected_role_embeddings, axis=0)
            np.save(role_embed_path, all_role_embeds)
            print(f"[Tracker] Saved {all_role_embeds.shape} role embeddings to {role_embed_path}")

    def train(self, logger, t_env):
        if self.buffer.num_episodes < self.args.batch_size:
            return

        if self.use_shared_predictor:
            self.tracker_net.train()
        else:
            self.tracker_nets.train()

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

        target_actions = batch.a[:, :max_t].long().to(dev)
        mask = batch.m[:, :max_t, 0].float().to(dev)

        total_loss = torch.tensor(0.0, device=dev)
        total_count = torch.tensor(0.0, device=dev)

        if self.use_shared_predictor:
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

            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, n_pairs)
            hidden = self.tracker_net.init_hidden().expand(bs * n_pairs, -1).contiguous()

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
        else:
            pair_i_dev = self.pair_i_indices.to(dev)
            last_a_n = batch.last_onehot_a_n[:, :max_t].to(dev)
            last_a_i_all = last_a_n[:, :, pair_i_dev, :]

            reward_all = batch.r[:, :max_t].to(dev)
            reward_expanded = reward_all.unsqueeze(2).expand(-1, -1, n_pairs, -1)

            full_input = torch.cat([obs_i_all, last_a_i_all, reward_expanded], dim=-1)

            hiddens = [
                net.init_hidden().expand(bs, -1).contiguous()
                for net in self.tracker_nets
            ]

            for t in range(max_t):
                mask_t = mask[:, t]
                if mask_t.sum() == 0:
                    continue

                for p in range(n_pairs):
                    input_p = full_input[:, t, p, :]
                    pred_q, hiddens[p] = self.tracker_nets[p](input_p, hiddens[p])
                    hiddens[p] = hiddens[p].detach()

                    target_p = target_actions[:, t, p]
                    loss_p = self.loss_fn(pred_q, target_p)
                    total_loss += (loss_p * mask_t).sum()
                    total_count += mask_t.sum()

        if total_count > 0:
            final_loss = total_loss / total_count
            self.optimiser.zero_grad()
            final_loss.backward()

            if self.use_shared_predictor:
                torch.nn.utils.clip_grad_norm_(self.tracker_net.parameters(), self.args.grad_norm_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.tracker_nets.parameters(), self.args.grad_norm_clip)

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
        if self.use_shared_predictor:
            torch.save(self.tracker_net.state_dict(), f"{path}/tracker_agent.th")
        else:
            torch.save(self.tracker_nets.state_dict(), f"{path}/tracker_agents_v4.th")
        torch.save(self.optimiser.state_dict(), f"{path}/tracker_opt.th")
        print(f"[Tracker] Saved model to {path}")
