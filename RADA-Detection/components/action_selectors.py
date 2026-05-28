import torch as th
from torch.distributions import Categorical

class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args
        self.epsilon = args.epsilon_start
        self.epsilon_decay = (args.epsilon_start - args.epsilon_finish) / args.epsilon_anneal_time
        self.epsilon_finish = args.epsilon_finish

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        self.epsilon = max(self.epsilon_finish, self.epsilon - self.epsilon_decay) if not test_mode else self.args.evaluation_epsilon
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

REGISTRY = {}
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector