import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackerAgentContinuous(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        action_dim,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super(TrackerAgentContinuous, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.orthogonal_(self.fc_mean.weight, gain=0.1)
        nn.init.constant_(self.fc_mean.bias, 0.10)

        nn.init.normal_(self.fc_log_std.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_log_std.bias, -0.71)

    def init_hidden(self, batch_size=1):
        return self.fc1.weight.new_zeros(batch_size, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))

        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h_out = self.gru(x, h_in)

        mean = self.fc_mean(h_out)

        log_std = self.fc_log_std(h_out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        return mean, std, h_out


class TrackerNetworkEnsemble(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        role_embedding_dim,
        num_agents,
        hidden_dim=128,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super(TrackerNetworkEnsemble, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.role_embedding_dim = role_embedding_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        input_dim = obs_dim + action_dim + role_embedding_dim + 2 * num_agents

        self.tracker = TrackerAgentContinuous(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def init_hidden(self, batch_size=1):
        N = self.num_agents
        h = self.tracker.init_hidden(1)
        return h.new_zeros(batch_size, N, N, self.hidden_dim)

    def forward(self, obs, last_action, role_embedding, hidden_states):
        B, N = obs.shape[0], obs.shape[1]
        device = obs.device

        eye = torch.eye(N, device=device)
        obs_i = obs.unsqueeze(2).expand(B, N, N, -1)
        role_j = role_embedding.unsqueeze(1).expand(B, N, N, -1)
        act_j = last_action.unsqueeze(1).expand(B, N, N, -1)
        id_i = eye.view(1, N, 1, N).expand(B, -1, N, -1)
        id_j = eye.view(1, 1, N, N).expand(B, N, -1, -1)

        inputs = torch.cat([obs_i, act_j, role_j, id_i, id_j], dim=-1)
        inputs_flat = inputs.reshape(B * N * N, -1)
        hidden_flat = hidden_states.reshape(B * N * N, -1)

        mu_flat, std_flat, h_flat = self.tracker(inputs_flat, hidden_flat)

        mu = mu_flat.reshape(B, N, N, -1)
        std = std_flat.reshape(B, N, N, -1)
        new_hidden = h_flat.reshape(B, N, N, -1)

        return mu, std, new_hidden
