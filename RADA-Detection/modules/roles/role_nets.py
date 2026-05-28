import torch
import torch.nn as nn


class Agent_Embedding(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding, self).__init__()
        self.args = args
        self.input_dim = args.obs_shape + args.n_actions
        self.agent_embedding_dim = args.agent_embedding_dim
        self.fc1 = nn.Linear(self.input_dim, self.agent_embedding_dim)
        self.agent_embedding_rnn = nn.GRUCell(self.agent_embedding_dim, self.agent_embedding_dim)
        self.fc2 = nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim)

    def forward(self, obs, last_a, hidden_state):
        inputs = torch.cat([obs, last_a], dim=-1)
        fc1_out = torch.relu(self.fc1(inputs))
        orig_shape = fc1_out.shape
        if len(orig_shape) == 3:
            b, n, d = orig_shape
            fc1_out_flat = fc1_out.reshape(b * n, d)
            h_in_flat = hidden_state.reshape(b * n, self.agent_embedding_dim)
        else:
            fc1_out_flat = fc1_out
            h_in_flat = hidden_state.reshape(-1, self.agent_embedding_dim)

        h = self.agent_embedding_rnn(fc1_out_flat, h_in_flat)
        fc2_out = self.fc2(h)
        if len(orig_shape) == 3:
            h = h.reshape(b, n, -1)
            fc2_out = fc2_out.reshape(b, n, -1)

        return fc2_out, h

class Role_Embedding(nn.Module):
    def __init__(self, args):
        super(Role_Embedding, self).__init__()
        self.args = args
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        use_ln = getattr(args, "use_ln", False)
        if use_ln:
            self.role_embedding_layer = nn.Sequential(nn.Linear(self.agent_embedding_dim, self.role_embedding_dim), nn.LayerNorm(self.role_embedding_dim))
        else:
            self.role_embedding_layer = nn.Linear(self.agent_embedding_dim, self.role_embedding_dim)
    def forward(self, agent_embedding):
        output = self.role_embedding_layer(agent_embedding)
        return torch.sigmoid(output)

class RECL_NET(nn.Module):
    def __init__(self, args):
        super(RECL_NET, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.role_embedding_dim = args.role_embedding_dim
        self.agent_embedding_net = Agent_Embedding(args)
        self.role_embedding_net = Role_Embedding(args)
        self.role_embedding_target_net = Role_Embedding(args)
        self.role_embedding_target_net.load_state_dict(self.role_embedding_net.state_dict())
        self.W = nn.Parameter(torch.rand(self.role_embedding_dim, self.role_embedding_dim))