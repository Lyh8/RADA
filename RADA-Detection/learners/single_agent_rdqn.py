from modules.agents.rnn_agent import RNNAgent
from utils.buffer_recurrent import RecurrentReplayBuffer
import torch
import os
import numpy as np
import math


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RDQNAgent:
    def __init__(self, obs_size, action_size, args, lambda_init=None):
        if lambda_init is None:
            lambda_init = [1 / 16] * (args.n_agents - 1)

        self.obs_size = obs_size
        self.n_actions = action_size
        self.args = args
        self.device = get_device()
        self.gamma = getattr(args, 'gamma', 0.99)
        self.lr = getattr(args, 'adv_lr', 1e-4)
        self.exploration_proba = 1.0
        self.exploration_proba_decay = getattr(args, 'adv_exploration_proba_decay', 0.005)
        self.batch_size = getattr(args, 'adv_batch_size', 32)
        self.buffer_size = getattr(args, 'adv_buffer_size', 5000)
        self.input_shape = self.obs_size
        self.buffer = RecurrentReplayBuffer(self.obs_size, 1, self.args.env_args.get("time_limit", 150),
                                            self.buffer_size, self.batch_size)
        self.args.use_rnn = True
        self.args.hidden_dim = 64

        self.model = RNNAgent(self.input_shape, self.args).to(self.device)
        self.hidden = self.model.init_hidden()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.test_mode = getattr(self.args, "adv_test_mode", False)

        self.lambda_coef = np.array(lambda_init)
        self.lambda_lr = 0.02
        self.constraint_value = []
        self.training_steps = 0

    def load_model(self, save_dir):
        if os.path.isfile(save_dir):
            self.model.load_state_dict(torch.load(save_dir, map_location=self.device))
            print(f"Loaded Adversary model from {save_dir} (legacy single-file format)")
            return

        model_path = os.path.join(save_dir, "adv_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded Adversary model from {model_path}")
        else:
            print(f"Adversary model not found at {model_path}, starting from scratch.")
            return

        opt_path = os.path.join(save_dir, "adv_optimizer.pth")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
            print(f"Loaded Adversary optimizer from {opt_path}")
        else:
            print(f"[Info] Adversary optimizer not found at {opt_path}, using fresh optimizer.")

        state_path = os.path.join(save_dir, "adv_train_state.pth")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.exploration_proba = state.get("exploration_proba", 1.0)
            self.lambda_coef = np.array(state.get("lambda_coef", self.lambda_coef.tolist()))
            self.constraint_value = state.get("constraint_value", [])
            self.training_steps = state.get("training_steps", 0)
            print(f"Loaded Adversary train state: "
                  f"epsilon={self.exploration_proba:.6f}, "
                  f"training_steps={self.training_steps}, "
                  f"lambda_mean={np.mean(self.lambda_coef):.6f}")
        else:
            print(f"[Info] Adversary train state not found at {state_path}, "
                  f"using defaults (epsilon=1.0, training_steps=0)")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "adv_model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "adv_optimizer.pth"))
        train_state = {
            "exploration_proba": self.exploration_proba,
            "lambda_coef": self.lambda_coef.tolist(),
            "constraint_value": self.constraint_value,
            "training_steps": self.training_steps,
        }
        torch.save(train_state, os.path.join(save_dir, "adv_train_state.pth"))

        print(f"Saved Adversary model + optimizer + state to {save_dir} "
              f"(epsilon={self.exploration_proba:.6f}, steps={self.training_steps})")

    def compute_action(self, obs, avail_actions):
        avail_actions_ind = np.nonzero(avail_actions)[0]
        if len(avail_actions_ind) == 0:
            return 0

        if not self.test_mode:
            if np.random.uniform(0, 1) < self.exploration_proba:
                return np.random.choice(avail_actions_ind)

        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            inputs = []
            inputs.append(obs)
            inp = torch.cat([x.reshape(1, -1) for x in inputs], dim=1)

            q_values, self.hidden = self.model(inp, self.hidden)

            for ind in range(self.n_actions):
                if avail_actions[ind] == 0:
                    q_values[0][ind] = -math.inf

            m, i = torch.max(q_values, dim=1)
            return torch.squeeze(i).tolist()

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def train(self):
        if self.buffer.num_episodes < self.batch_size:
            return

        batch = self.buffer.sample()
        bs, num_bptt = batch.r.shape[0], batch.r.shape[1]
        obs = torch.FloatTensor(batch.o).to(self.device)

        hidden = self.model.init_hidden().unsqueeze(0).expand(bs, 1, -1)
        predictions = []

        for t in range(num_bptt):
            inp = []
            inp.append(obs[:, t, :])
            inp = torch.cat([x.reshape(bs, -1) for x in inp], dim=1)
            agent_outs, hidden = self.model(inp, hidden)
            predictions.append(agent_outs)

        predictions = torch.stack(predictions, dim=1)

        actions = torch.tensor(batch.a, dtype=torch.int64).to(self.device)
        q_values = torch.gather(predictions, dim=2, index=actions)

        target_hidden = self.model.init_hidden().unsqueeze(0).expand(bs, 1, -1)
        targets = []
        with torch.no_grad():
            for t in range(num_bptt + 1):
                next_inp = []
                next_inp.append(obs[:, t, :])
                next_inp = torch.cat([x.reshape(bs, -1) for x in next_inp], dim=1)
                target_outs, target_hidden = self.model(next_inp, target_hidden)
                targets.append(target_outs)

            targets = torch.stack(targets[1:], dim=1)
            m, i = torch.max(targets, dim=2, keepdim=True)
            rewards = torch.FloatTensor(batch.r).to(self.device)
            dones = torch.FloatTensor(batch.d).to(self.device)
            q_target = rewards + self.gamma * (1 - dones) * m
        mask = torch.FloatTensor(batch.m).to(self.device)
        Q_loss_elementwise = (q_values - q_target) ** 2
        Q_loss = torch.mean(Q_loss_elementwise * mask) / mask.sum() * np.prod(mask.shape)
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        return Q_loss.item()

    def store_episode(self, current_obs, action, reward, next_state, done, victim_action):
        self.buffer.push(current_obs, action, reward, next_state, done, False)

    def constraint_reward(self, z):
        return np.sum(np.array(z) * self.lambda_coef)

    def lambda_update(self, V, th):
        self.lambda_coef = self.lambda_coef - self.lambda_lr * (V - th / (1 - self.gamma))

    def reset(self):
        self.buffer.reset()
        self.hidden = self.model.init_hidden()
        self.exploration_proba = 1.0
        self.constraint_value = []
        self.training_steps = 0
