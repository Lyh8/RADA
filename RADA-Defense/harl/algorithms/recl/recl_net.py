"""ReCL Network Definitions.

Standalone embedding networks for Role Embedding Contrastive Learning (ReCL).
Extracted and adapted from ACORM's acorm_net.py for use with the HARL framework.

Components:
    - AgentEmbedding: GRU-based encoder that maps observation sequences to agent embeddings.
    - AgentEmbeddingDecoder: Predicts next observation + agent identity (for reconstruction training).
    - RoleEmbedding: Online encoder + EMA target encoder for contrastive role learning.
    - EmbeddingNet: Wraps all above modules plus a learnable bilinear matrix W.
"""

import torch
import torch.nn as nn


class AgentEmbedding(nn.Module):
    """GRU-based agent embedding encoder.

    Processes observation sequences to produce agent embeddings that capture
    each agent's temporal behavior pattern. Must be called sequentially
    across timesteps to maintain GRU hidden state.

    Args:
        obs_dim (int): Dimension of the observation vector.
        agent_embedding_dim (int): Dimension of the output agent embedding.
    """

    def __init__(self, obs_dim, agent_embedding_dim):
        super(AgentEmbedding, self).__init__()
        self.obs_dim = obs_dim
        self.agent_embedding_dim = agent_embedding_dim

        self.fc1 = nn.Linear(obs_dim, obs_dim)
        self.rnn_hidden = None
        self.gru_cell = nn.GRUCell(obs_dim, agent_embedding_dim)

    def forward(self, obs, detach=False):
        """Forward pass.

        Args:
            obs: (batch*N, obs_dim) observation input.
            detach: If True, detach the output from computation graph.

        Returns:
            agent_embedding: (batch*N, agent_embedding_dim)
        """
        x = torch.relu(self.fc1(obs))
        self.rnn_hidden = self.gru_cell(x, self.rnn_hidden)
        output = self.rnn_hidden
        if detach:
            output = output.detach()  # Fixed: original ACORM had a no-op detach bug
        return output


class AgentEmbeddingDecoder(nn.Module):
    """Decoder for agent embedding reconstruction training.

    Predicts next-step observation concatenated with a one-hot agent identity,
    used as a self-supervised objective to train the AgentEmbedding encoder.

    Args:
        agent_embedding_dim (int): Input dimension (agent embedding size).
        obs_dim (int): Observation dimension.
        num_agents (int): Number of agents (for one-hot identity).
    """

    def __init__(self, agent_embedding_dim, obs_dim, num_agents):
        super(AgentEmbeddingDecoder, self).__init__()
        self.decoder_out_dim = obs_dim + num_agents

        self.fc1 = nn.Linear(agent_embedding_dim, agent_embedding_dim)
        self.fc2 = nn.Linear(agent_embedding_dim, self.decoder_out_dim)

    def forward(self, agent_embedding):
        """Forward pass.

        Args:
            agent_embedding: (..., agent_embedding_dim)

        Returns:
            decoder_output: (..., obs_dim + num_agents)
        """
        x = torch.relu(self.fc1(agent_embedding))
        return self.fc2(x)


class RoleEmbedding(nn.Module):
    """Role embedding network with online encoder and EMA target encoder.

    The online encoder is updated via gradient descent (contrastive loss).
    The target encoder is updated via exponential moving average (EMA) of
    the online encoder, following the BYOL / MoCo paradigm.

    Args:
        agent_embedding_dim (int): Input dimension.
        role_embedding_dim (int): Output role embedding dimension.
    """

    def __init__(self, agent_embedding_dim, role_embedding_dim):
        super(RoleEmbedding, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(agent_embedding_dim, agent_embedding_dim),
            nn.Linear(agent_embedding_dim, role_embedding_dim),
        ])
        self.target_encoder = nn.ModuleList([
            nn.Linear(agent_embedding_dim, agent_embedding_dim),
            nn.Linear(agent_embedding_dim, role_embedding_dim),
        ])
        # Initialize target encoder with same weights as online encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def forward(self, agent_embedding, detach=False, ema=False):
        """Forward pass.

        Args:
            agent_embedding: (..., agent_embedding_dim)
            detach: If True, detach output from computation graph.
            ema: If True, use target encoder; if False, use online encoder.

        Returns:
            role_embedding: (..., role_embedding_dim)
        """
        if ema:
            output = torch.relu(self.target_encoder[0](agent_embedding))
            output = self.target_encoder[1](output)
        else:
            output = torch.relu(self.encoder[0](agent_embedding))
            output = self.encoder[1](output)

        if detach:
            output = output.detach()  # Fixed: original ACORM had a no-op detach bug
        return output


class EmbeddingNet(nn.Module):
    """Complete embedding network for ReCL.

    Wraps AgentEmbedding, AgentEmbeddingDecoder, RoleEmbedding, and
    a learnable bilinear matrix W used in contrastive loss computation.

    Args:
        obs_dim (int): Observation dimension.
        num_agents (int): Number of agents.
        agent_embedding_dim (int): Agent embedding dimension.
        role_embedding_dim (int): Role embedding dimension.
    """

    def __init__(self, obs_dim, num_agents, agent_embedding_dim, role_embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.agent_embedding_net = AgentEmbedding(obs_dim, agent_embedding_dim)
        self.agent_embedding_decoder = AgentEmbeddingDecoder(
            agent_embedding_dim, obs_dim, num_agents
        )
        self.role_embedding_net = RoleEmbedding(agent_embedding_dim, role_embedding_dim)
        self.W = nn.Parameter(torch.rand(role_embedding_dim, role_embedding_dim))

    def agent_embed_forward(self, obs, detach=False):
        """Compute agent embedding from observations.

        Args:
            obs: (batch*N, obs_dim)
            detach: Whether to detach output.

        Returns:
            agent_embedding: (batch*N, agent_embedding_dim)
        """
        return self.agent_embedding_net(obs, detach)

    def role_embed_forward(self, agent_embedding, detach=False, ema=False):
        """Compute role embedding from agent embedding.

        Args:
            agent_embedding: (batch*N, agent_embedding_dim)
            detach: Whether to detach output.
            ema: Whether to use target encoder.

        Returns:
            role_embedding: (batch*N, role_embedding_dim)
        """
        return self.role_embedding_net(agent_embedding, detach, ema)
