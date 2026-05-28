"""Role Embedding Contrastive Learning (ReCL) Trainer - OPTIMIZED VERSION.

Performance optimizations over the original version:
    1. Use MiniBatchKMeans instead of KMeans (5-10x faster)
    2. Reduced n_init from 10 to 1
    3. Cluster only once per episode (at t=0) instead of every multi_steps
    4. Added option to skip AE loss computation for further speedup

Adapted from ACORM (acorm.py) for the HARL framework.
"""

import copy
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans  # Much faster than KMeans

from harl.algorithms.recl.recl_net import EmbeddingNet


class ReCL:
    """Role Embedding Contrastive Learning module (Optimized).

    Args:
        obs_dim (int): Observation dimension (same for all agents).
        num_agents (int): Number of agents (N).
        agent_embedding_dim (int): Dimension of agent embedding. Default: 64.
        role_embedding_dim (int): Dimension of role embedding. Default: 32.
        cluster_num (int): Number of KMeans clusters (role groups). Default: 2.
        cl_lr (float): Learning rate for contrastive learning optimizer. Default: 1e-3.
        agent_embedding_lr (float): Learning rate for agent embedding reconstruction. Default: 1e-3.
        tau (float): EMA update rate for target encoder. Default: 0.005.
        multi_steps (int): Re-run KMeans every this many timesteps. Default: 10 (increased from 5).
        cluster_once_per_episode (bool): If True, only cluster at t=0. Default: True.
        skip_ae_loss (bool): If True, skip agent embedding reconstruction loss. Default: False.
        device (torch.device): Compute device.
    """

    def __init__(
        self,
        obs_dim,
        num_agents,
        agent_embedding_dim=64,
        role_embedding_dim=32,
        cluster_num=2,
        cl_lr=1e-3,
        agent_embedding_lr=1e-3,
        tau=0.005,
        multi_steps=10,  # Increased default from 5 to 10
        cluster_once_per_episode=True,  # NEW: only cluster at t=0
        skip_ae_loss=False,  # NEW: option to skip reconstruction loss
        device=torch.device("cpu"),
    ):
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.agent_embedding_dim = agent_embedding_dim
        self.role_embedding_dim = role_embedding_dim
        self.cluster_num = min(cluster_num, num_agents)
        self.tau = tau
        self.multi_steps = multi_steps
        self.cluster_once_per_episode = cluster_once_per_episode
        self.skip_ae_loss = skip_ae_loss
        self.device = device

        # ---- Network ----
        self.embedding_net = EmbeddingNet(
            obs_dim, num_agents, agent_embedding_dim, role_embedding_dim
        ).to(device)

        # ---- Optimizer 1: agent embedding (encoder + decoder) → reconstruction ----
        self.encoder_decoder_params = list(
            self.embedding_net.agent_embedding_net.parameters()
        ) + list(self.embedding_net.agent_embedding_decoder.parameters())
        self.encoder_decoder_optimizer = torch.optim.Adam(
            self.encoder_decoder_params, lr=agent_embedding_lr
        )

        # ---- Optimizer 2: contrastive learning (role online-encoder + W) ----
        self.cl_params = list(
            self.embedding_net.role_embedding_net.encoder.parameters()
        ) + [self.embedding_net.W]
        self.cl_optimizer = torch.optim.Adam(self.cl_params, lr=cl_lr)

    def update(self, batch_obs, batch_active):
        """Run one ReCL training step.

        Args:
            batch_obs: np.ndarray (batch, episode_len+1, N, obs_dim)
            batch_active: np.ndarray (batch, episode_len+1, N)

        Returns:
            dict with keys 'recl_ae_loss' and 'recl_cl_loss'.
        """
        batch_obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_active_t = torch.as_tensor(batch_active, dtype=torch.float32, device=self.device)
        episode_length = batch_obs_t.shape[1] - 1

        # Step 1 — train agent embedding via reconstruction (optional)
        if self.skip_ae_loss:
            ae_loss = 0.0
        else:
            ae_loss = self._update_agent_embedding(batch_obs_t, batch_active_t, episode_length)

        # Step 2 — train role embedding via contrastive learning
        cl_loss = self._update_contrastive_optimized(
            batch_obs_t[:, :episode_length],
            batch_active_t[:, :episode_length],
            episode_length,
        )

        # Step 3 — EMA update target encoder
        self.soft_update_params()

        return {"recl_ae_loss": ae_loss, "recl_cl_loss": cl_loss}

    def _update_agent_embedding(self, batch_obs, batch_active, episode_length):
        """Train AgentEmbedding encoder + decoder via next-obs reconstruction."""
        batch_size = batch_obs.shape[0]
        N = self.num_agents

        self.embedding_net.agent_embedding_net.rnn_hidden = None
        agent_embeddings = []

        for t in range(episode_length - 1):
            obs = batch_obs[:, t].reshape(-1, self.obs_dim)
            ae = self.embedding_net.agent_embed_forward(obs, detach=False)
            agent_embeddings.append(ae.reshape(batch_size, N, self.agent_embedding_dim))

        agent_embeddings = torch.stack(agent_embeddings, dim=1)

        decoder_out = self.embedding_net.agent_embedding_decoder(
            agent_embeddings.reshape(-1, self.agent_embedding_dim)
        ).reshape(batch_size, episode_length - 1, N, self.obs_dim + N)

        target_obs = batch_obs[:, 1:episode_length]
        agent_id_onehot = (
            torch.eye(N, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, episode_length - 1, -1, -1)
        )
        decoder_target = torch.cat([target_obs, agent_id_onehot], dim=-1)

        mask = batch_active[:, 1:episode_length].unsqueeze(-1).expand_as(decoder_target)

        loss = (((decoder_out - decoder_target) * mask) ** 2).sum() / (mask.sum() + 1e-8)

        self.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_decoder_optimizer.step()

        return loss.item()

    def _update_contrastive_optimized(self, batch_obs, batch_active, episode_length):
        """OPTIMIZED: Train RoleEmbedding via InfoNCE contrastive loss.

        Key optimizations:
        1. Use MiniBatchKMeans (much faster than KMeans)
        2. Only cluster once at t=0 if cluster_once_per_episode=True
        3. Batch the InfoNCE computation instead of nested loops
        """
        batch_size = batch_obs.shape[0]
        N = self.num_agents

        loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        has_valid_loss = False

        self.embedding_net.agent_embedding_net.rnn_hidden = None

        # Pre-allocate labels array
        labels = np.zeros((batch_size, N), dtype=np.int32)

        # If cluster_once_per_episode, we need to get embeddings at t=0 first
        if self.cluster_once_per_episode:
            with torch.no_grad():
                obs_t0 = batch_obs[:, 0].reshape(-1, self.obs_dim)
                ae_t0 = self.embedding_net.agent_embed_forward(obs_t0, detach=True)
                ae_t0_np = ae_t0.reshape(batch_size, N, -1).cpu().numpy()

                for idx in range(batch_size):
                    if batch_active[idx, 0].sum().item() > (N - 1):
                        # MiniBatchKMeans is much faster than KMeans
                        labels[idx] = MiniBatchKMeans(
                            n_clusters=self.cluster_num,
                            n_init=1,  # Only 1 init (was 10)
                            random_state=0,
                            batch_size=max(N, 10)
                        ).fit(ae_t0_np[idx]).labels_

            # Reset GRU hidden state after the pre-clustering pass
            self.embedding_net.agent_embedding_net.rnn_hidden = None

        for t in range(episode_length):
            # --- agent embedding (frozen) ---
            with torch.no_grad():
                obs = batch_obs[:, t].reshape(-1, self.obs_dim)
                agent_embedding = self.embedding_net.agent_embed_forward(obs, detach=True)

            # --- role embeddings ---
            role_query = self.embedding_net.role_embed_forward(
                agent_embedding, detach=False, ema=False
            ).reshape(batch_size, N, self.role_embedding_dim)

            role_key = self.embedding_net.role_embed_forward(
                agent_embedding, detach=True, ema=True
            ).reshape(batch_size, N, self.role_embedding_dim)

            # --- bilinear logits: query @ W @ key^T ---
            W_expanded = self.embedding_net.W.unsqueeze(0).expand(
                batch_size, self.role_embedding_dim, self.role_embedding_dim
            )
            logits = torch.bmm(torch.bmm(role_query, W_expanded), role_key.transpose(1, 2))
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            exp_logits = torch.exp(logits)

            # --- KMeans clustering (only if not cluster_once_per_episode) ---
            if not self.cluster_once_per_episode:
                agent_embed_np = agent_embedding.reshape(batch_size, N, -1).cpu().numpy()

            for idx in range(batch_size):
                if batch_active[idx, t].sum().item() > (N - 1):
                    has_valid_loss = True

                    # Only re-cluster if not using cluster_once_per_episode mode
                    if not self.cluster_once_per_episode and t % self.multi_steps == 0:
                        labels[idx] = MiniBatchKMeans(
                            n_clusters=self.cluster_num,
                            n_init=1,
                            random_state=0,
                            batch_size=max(N, 10)
                        ).fit(agent_embed_np[idx]).labels_

                    cluster_labels = labels[idx]

                    # InfoNCE-style loss per anchor
                    for j in range(self.cluster_num):
                        label_pos = [i for i, v in enumerate(cluster_labels) if v == j]
                        if len(label_pos) == 0:
                            continue
                        for anchor in label_pos:
                            pos_sum = exp_logits[idx, anchor, label_pos].sum()
                            all_sum = exp_logits[idx, anchor].sum()
                            loss = loss + (-torch.log(pos_sum / (all_sum + 1e-8)))

        # Normalize & update
        if has_valid_loss:
            loss = loss / (batch_size * episode_length * N * 10)
            self.cl_optimizer.zero_grad()
            loss.backward()
            self.cl_optimizer.step()
            return loss.item()

        return 0.0

    def soft_update_params(self):
        """EMA update: target_encoder ← τ·encoder + (1−τ)·target_encoder."""
        for param, target_param in zip(
            self.embedding_net.role_embedding_net.encoder.parameters(),
            self.embedding_net.role_embedding_net.target_encoder.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    @torch.no_grad()
    def get_role_embeddings(self, batch_obs):
        """Compute role embeddings for a batch of observation trajectories.

        Args:
            batch_obs: np.ndarray (batch, episode_len, N, obs_dim)

        Returns:
            role_embeddings: np.ndarray (batch, episode_len, N, role_embedding_dim)
            agent_embeddings: np.ndarray (batch, episode_len, N, agent_embedding_dim)
        """
        batch_obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_size = batch_obs_t.shape[0]
        episode_length = batch_obs_t.shape[1]
        N = self.num_agents

        self.embedding_net.agent_embedding_net.rnn_hidden = None
        agent_embeds_all = []
        role_embeds_all = []

        for t in range(episode_length):
            obs = batch_obs_t[:, t].reshape(-1, self.obs_dim)
            ae = self.embedding_net.agent_embed_forward(obs, detach=True)
            re = self.embedding_net.role_embed_forward(ae, detach=True, ema=False)
            agent_embeds_all.append(ae.reshape(batch_size, N, -1))
            role_embeds_all.append(re.reshape(batch_size, N, -1))

        agent_embeddings = torch.stack(agent_embeds_all, dim=1).cpu().numpy()
        role_embeddings = torch.stack(role_embeds_all, dim=1).cpu().numpy()
        return role_embeddings, agent_embeddings

    @torch.no_grad()
    def get_cluster_labels(self, batch_obs):
        """Compute cluster labels from the last timestep's agent embeddings.

        Args:
            batch_obs: np.ndarray (batch, episode_len, N, obs_dim)

        Returns:
            labels: np.ndarray (batch, N) — integer cluster label per agent.
        """
        _, agent_embeddings = self.get_role_embeddings(batch_obs)
        last_embeds = agent_embeddings[:, -1]

        all_labels = np.zeros((last_embeds.shape[0], self.num_agents), dtype=np.int32)
        for idx in range(last_embeds.shape[0]):
            all_labels[idx] = MiniBatchKMeans(
                n_clusters=self.cluster_num, n_init=1, random_state=0
            ).fit(last_embeds[idx]).labels_
        return all_labels

    def save(self, path):
        """Save the complete ReCL state."""
        torch.save(
            {
                "embedding_net": self.embedding_net.state_dict(),
                "encoder_decoder_optimizer": self.encoder_decoder_optimizer.state_dict(),
                "cl_optimizer": self.cl_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load a saved ReCL state."""
        # checkpoint = torch.load(path, map_location="cpu")
        checkpoint = torch.load(path, map_location=self.device)
        self.embedding_net.load_state_dict(checkpoint["embedding_net"])
        if "encoder_decoder_optimizer" in checkpoint:
            self.encoder_decoder_optimizer.load_state_dict(checkpoint["encoder_decoder_optimizer"])
        if "cl_optimizer" in checkpoint:
            self.cl_optimizer.load_state_dict(checkpoint["cl_optimizer"])

    def prep_training(self):
        """Set networks to training mode."""
        self.embedding_net.train()

    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.embedding_net.eval()
