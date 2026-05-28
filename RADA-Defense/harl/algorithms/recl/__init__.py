"""Role Embedding Contrastive Learning (ReCL) Trainer.

Standalone module that trains alongside MAPPO to learn agent role groupings
via contrastive learning. Does NOT modify MAPPO's actor or critic — it is
a completely independent network that uses the same observations.

Training pipeline (called each episode after MAPPO update):
    1. Update AgentEmbedding encoder via reconstruction loss (predict next obs + agent id).
    2. Update RoleEmbedding online encoder via InfoNCE-style contrastive loss
       with KMeans pseudo-labels.
    3. EMA-update the target encoder from the online encoder.

After training, the learned role embeddings / cluster labels can be extracted
for downstream use via get_role_embeddings() and get_cluster_labels().

Adapted from ACORM (acorm.py) for the HARL framework.
"""

import copy
import numpy as np
import torch
from sklearn.cluster import KMeans

# Use relative or absolute import depending on your project structure.
# If placed at harl/algorithms/recl/recl.py, use:
from harl.algorithms.recl.recl_net import EmbeddingNet


class ReCL:
    """Role Embedding Contrastive Learning module.

    Args:
        obs_dim (int): Observation dimension (same for all agents).
        num_agents (int): Number of agents (N).
        agent_embedding_dim (int): Dimension of agent embedding. Default: 64.
        role_embedding_dim (int): Dimension of role embedding. Default: 32.
        cluster_num (int): Number of KMeans clusters (role groups). Default: 2.
        cl_lr (float): Learning rate for contrastive learning optimizer. Default: 1e-3.
        agent_embedding_lr (float): Learning rate for agent embedding reconstruction. Default: 1e-3.
        tau (float): EMA update rate for target encoder. Default: 0.005.
        multi_steps (int): Re-run KMeans every this many timesteps. Default: 5.
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
        multi_steps=5,
        device=torch.device("cpu"),
    ):
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.agent_embedding_dim = agent_embedding_dim
        self.role_embedding_dim = role_embedding_dim
        self.cluster_num = min(cluster_num, num_agents)  # can't have more clusters than agents
        self.tau = tau
        self.multi_steps = multi_steps
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
        #      Note: target_encoder is NOT in this optimizer; it is EMA-updated.
        self.cl_params = list(
            self.embedding_net.role_embedding_net.encoder.parameters()
        ) + [self.embedding_net.W]
        self.cl_optimizer = torch.optim.Adam(self.cl_params, lr=cl_lr)

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def update(self, batch_obs, batch_active):
        """Run one ReCL training step.

        Should be called once per training episode after MAPPO's own update.

        Args:
            batch_obs: np.ndarray (batch, episode_len+1, N, obs_dim)
                       All observations including the terminal observation.
            batch_active: np.ndarray (batch, episode_len+1, N)
                          Active masks (1 = alive, 0 = dead/padded).

        Returns:
            dict with keys 'recl_ae_loss' and 'recl_cl_loss'.
        """
        batch_obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_active_t = torch.as_tensor(batch_active, dtype=torch.float32, device=self.device)
        episode_length = batch_obs_t.shape[1] - 1  # actual number of steps

        # Step 1 — train agent embedding via reconstruction
        ae_loss = self._update_agent_embedding(batch_obs_t, batch_active_t, episode_length)

        # Step 2 — train role embedding via contrastive learning
        cl_loss = self._update_contrastive(
            batch_obs_t[:, :episode_length],
            batch_active_t[:, :episode_length],
            episode_length,
        )

        # Step 3 — EMA update target encoder
        self.soft_update_params()

        return {"recl_ae_loss": ae_loss, "recl_cl_loss": cl_loss}

    # ------------------------------------------------------------------
    #  Agent Embedding Reconstruction
    # ------------------------------------------------------------------
    def _update_agent_embedding(self, batch_obs, batch_active, episode_length):
        """Train AgentEmbedding encoder + decoder via next-obs reconstruction.

        Forward the GRU at t = 0 … episode_length-2, decode each embedding,
        and compare against (obs_{t+1}, agent_id_one_hot).
        """
        batch_size = batch_obs.shape[0]
        N = self.num_agents

        self.embedding_net.agent_embedding_net.rnn_hidden = None
        agent_embeddings = []

        for t in range(episode_length - 1):
            obs = batch_obs[:, t].reshape(-1, self.obs_dim)  # (batch*N, obs_dim)
            ae = self.embedding_net.agent_embed_forward(obs, detach=False)
            agent_embeddings.append(ae.reshape(batch_size, N, self.agent_embedding_dim))

        # (batch, episode_length-1, N, agent_embedding_dim)
        agent_embeddings = torch.stack(agent_embeddings, dim=1)

        # Decode → (batch, episode_length-1, N, obs_dim+N)
        decoder_out = self.embedding_net.agent_embedding_decoder(
            agent_embeddings.reshape(-1, self.agent_embedding_dim)
        ).reshape(batch_size, episode_length - 1, N, self.obs_dim + N)

        # Target: next obs + one-hot agent id
        target_obs = batch_obs[:, 1:episode_length]  # (batch, ep_len-1, N, obs_dim)
        agent_id_onehot = (
            torch.eye(N, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, episode_length - 1, -1, -1)
        )
        decoder_target = torch.cat([target_obs, agent_id_onehot], dim=-1)

        # Mask: use active-mask at the *target* timesteps
        mask = batch_active[:, 1:episode_length].unsqueeze(-1).expand_as(decoder_target)

        loss = (((decoder_out - decoder_target) * mask) ** 2).sum() / (mask.sum() + 1e-8)

        self.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_decoder_optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    #  Contrastive Role Learning
    # ------------------------------------------------------------------
    def _update_contrastive(self, batch_obs, batch_active, episode_length):
        """Train RoleEmbedding via InfoNCE contrastive loss.

        For each timestep:
          1. Compute agent_embedding (no grad — GRU is frozen here).
          2. Get role_embedding_query from online encoder (with grad).
          3. Get role_embedding_key from target encoder (detached).
          4. KMeans on agent_embedding → pseudo cluster labels.
          5. Accumulate contrastive loss: pull same-cluster, push different-cluster.
        """
        batch_size = batch_obs.shape[0]
        N = self.num_agents

        loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        has_valid_loss = False

        self.embedding_net.agent_embedding_net.rnn_hidden = None
        labels = np.zeros((batch_size, N))  # cached KMeans labels

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
            # numerical stability
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            exp_logits = torch.exp(logits)  # (batch, N, N)

            # --- KMeans clustering on agent embeddings ---
            agent_embed_np = agent_embedding.reshape(batch_size, N, -1).cpu().numpy()

            for idx in range(batch_size):
                if batch_active[idx, t].sum().item() > (N - 1):
                    has_valid_loss = True

                    if t % self.multi_steps == 0:
                        cluster_labels = (
                            KMeans(n_clusters=self.cluster_num, n_init=10, random_state=0)
                            .fit(agent_embed_np[idx])
                            .labels_
                        )
                        labels[idx] = cluster_labels.copy()
                    else:
                        cluster_labels = labels[idx].copy()

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

    # ------------------------------------------------------------------
    #  EMA update
    # ------------------------------------------------------------------
    def soft_update_params(self):
        """Exponential moving average update: target_encoder ← τ·encoder + (1−τ)·target_encoder."""
        for param, target_param in zip(
            self.embedding_net.role_embedding_net.encoder.parameters(),
            self.embedding_net.role_embedding_net.target_encoder.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    #  Inference / Downstream Use
    # ------------------------------------------------------------------
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
        last_embeds = agent_embeddings[:, -1]  # (batch, N, agent_embedding_dim)

        all_labels = np.zeros((last_embeds.shape[0], self.num_agents), dtype=np.int32)
        for idx in range(last_embeds.shape[0]):
            all_labels[idx] = (
                KMeans(n_clusters=self.cluster_num, n_init=10, random_state=0)
                .fit(last_embeds[idx])
                .labels_
            )
        return all_labels

    # ------------------------------------------------------------------
    #  Save / Load / Mode switching
    # ------------------------------------------------------------------
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
