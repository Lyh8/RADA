import torch.nn as nn
import torch.nn.functional as F


class TrackerAgent(nn.Module):
    def __init__(self, input_shape, args, hidden_dim=None):
        super(TrackerAgent, self).__init__()
        self.args = args
        self.dim = hidden_dim if hidden_dim is not None else args.hidden_dim

        self.fc1 = nn.Linear(input_shape, self.dim)
        self.rnn = nn.GRUCell(self.dim, self.dim)
        self.fc2 = nn.Linear(self.dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h