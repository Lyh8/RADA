import copy
import os
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.roles.role_nets import RECL_NET
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from components.standarize_stream import RunningMeanStd
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.mixer = QMixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mac = copy.deepcopy(mac)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.optimiser = Adam(params=self.params, lr=args.lr)

        if getattr(self.args, "use_recl", False):
            self.recl_net = RECL_NET(args)
            self.recl_params = list(self.recl_net.parameters())
            self.recl_optimiser = Adam(params=self.recl_params, lr=args.recl_lr)
            self.last_cluster_centers = None

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        device = "cuda" if args.use_cuda else "cpu"
        if getattr(self.args, "standardise_returns", False): self.ret_ms = RunningMeanStd(shape=(1,), device=device)
        if getattr(self.args, "standardise_rewards", False): self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length - 1

        all_agent_embeds, all_role_embeds, all_role_embeds_target, cluster_masks = None, None, None, None

        if getattr(self.args, "use_recl", False):
            all_agent_embeds, all_role_embeds, all_role_embeds_target, cluster_masks = self._compute_embeddings_and_clusters(
                batch, bs, max_t)

        td_loss = self._calculate_qmix_loss(batch)

        recl_loss = th.tensor(0.0, device=self.args.device)
        if getattr(self.args, "use_recl", False):
            recl_loss = self._calculate_recl_loss(all_role_embeds, all_role_embeds_target, cluster_masks,
                                                  batch["filled"])

        recl_weight = getattr(self.args, "recl_loss_weight", 0.1)


        total_loss = td_loss + recl_weight * recl_loss

        self.optimiser.zero_grad()
        if getattr(self.args, "use_recl", False): self.recl_optimiser.zero_grad()

        total_loss.backward()

        grad_norm_qmix = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if getattr(self.args, "use_recl", False):
            th.nn.utils.clip_grad_norm_(self.recl_params, self.args.grad_norm_clip)
            self.recl_optimiser.step()

        self.training_steps += 1
        if (self.training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            if getattr(self.args, "use_recl", False): self.logger.log_stat("loss_recl", recl_loss.item(), t_env)
            self.logger.log_stat("grad_norm_qmix", grad_norm_qmix.item(), t_env)
            self.log_stats_t = t_env

    def _compute_embeddings_and_clusters(self, batch, bs, max_t):
        obs = batch["obs"][:, :-1]
        actions_onehot = batch["actions_onehot"]
        last_actions = th.cat([th.zeros_like(actions_onehot[:, 0:1]), actions_onehot[:, :-1]], dim=1)

        agent_embeds_t, role_embeds_t, role_embeds_target_t = [], [], []
        agent_embed_hidden = self.recl_net.agent_embedding_net.fc1.weight.new(bs * self.args.n_agents,
                                                                              self.args.agent_embedding_dim).zero_()

        for t in range(max_t):
            obs_t = obs[:, t].reshape(-1, self.args.obs_shape)
            last_a_t = last_actions[:, t].reshape(-1, self.args.n_actions)
            agent_embed, agent_embed_hidden = self.recl_net.agent_embedding_net(obs_t, last_a_t, agent_embed_hidden)
            role_embed = self.recl_net.role_embedding_net(agent_embed)
            with th.no_grad():
                role_embed_target = self.recl_net.role_embedding_target_net(agent_embed.detach())
            agent_embeds_t.append(agent_embed.reshape(bs, self.args.n_agents, -1))
            role_embeds_t.append(role_embed.reshape(bs, self.args.n_agents, -1))
            role_embeds_target_t.append(role_embed_target.reshape(bs, self.args.n_agents, -1))

        all_agent_embeds = th.stack(agent_embeds_t, dim=1)
        all_role_embeds = th.stack(role_embeds_t, dim=1)
        all_role_embeds_target = th.stack(role_embeds_target_t, dim=1)

        cluster_masks = th.zeros(bs, max_t, self.args.n_agents, self.args.n_agents, device=self.args.device)

        for t in range(max_t):
            if t % self.args.multi_steps == 0:
                for b in range(bs):
                    if batch["filled"][b, t, 0] == 1:
                        embed_np = all_agent_embeds[b, t].detach().cpu().numpy()
                        if np.isfinite(embed_np).all():
                            k = min(self.args.cluster_num, self.args.n_agents)
                            try:
                                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=self.args.seed).fit(embed_np)
                                labels = kmeans.labels_
                                self.last_cluster_centers = kmeans.cluster_centers_
                                mask = (labels[:, None] == labels[None, :]).astype(np.float32)
                                cluster_masks[b, t] = th.from_numpy(mask).to(self.args.device)
                            except Exception:
                                cluster_masks[b, t] = th.ones(self.args.n_agents, self.args.n_agents,
                                                              device=self.args.device)
                        else:
                            cluster_masks[b, t] = th.ones(self.args.n_agents, self.args.n_agents,
                                                          device=self.args.device)
            elif t > 0:
                cluster_masks[:, t] = cluster_masks[:, t - 1]
        return all_agent_embeds, all_role_embeds, all_role_embeds_target, cluster_masks

    def _calculate_qmix_loss(self, batch):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if getattr(self.args, "standardise_returns", False):
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if getattr(self.args, "standardise_returns", False):
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = (chosen_action_qvals - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        return loss

    def _calculate_recl_loss(self, role_embeds_query, role_embeds_key, cluster_masks, filled):
        bs, max_t, n_agents, _ = role_embeds_query.shape
        logits1 = th.matmul(role_embeds_query, self.recl_net.W)
        logits = th.matmul(logits1, role_embeds_key.permute(0, 1, 3, 2))
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        numerator = (exp_logits * cluster_masks).sum(dim=-1)
        denominator = exp_logits.sum(dim=-1)
        loss = -th.log(numerator / (denominator + 1e-8) + 1e-8)
        mask = filled[:, :-1].expand_as(loss)
        masked_loss = (loss * mask).sum() / mask.sum()
        return masked_loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        if getattr(self.args, "use_recl", False):
            for target, source in zip(self.recl_net.role_embedding_target_net.parameters(),
                                      self.recl_net.role_embedding_net.parameters()):
                target.data.copy_(target.data * (1.0 - self.args.role_tau) + source.data * self.args.role_tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        if getattr(self.args, "use_recl", False): self.recl_net.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")
        if getattr(self.args, "use_recl", False):
            th.save(self.recl_net.state_dict(), f"{path}/recl_net.th")
            th.save(self.recl_optimiser.state_dict(), f"{path}/recl_opt.th")

            if self.last_cluster_centers is not None:
                np.save(f"{path}/cluster_centers.npy", self.last_cluster_centers)

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
        if getattr(self.args, "use_recl", False):
            self.recl_net.load_state_dict(th.load(f"{path}/recl_net.th", map_location=lambda storage, loc: storage))
            self.recl_optimiser.load_state_dict(
                th.load(f"{path}/recl_opt.th", map_location=lambda storage, loc: storage))

            if os.path.exists(f"{path}/cluster_centers.npy"):
                try:
                    self.last_cluster_centers = np.load(f"{path}/cluster_centers.npy")
                except Exception:
                    pass

        self._update_targets()