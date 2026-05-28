import numpy as np
from collections import namedtuple

RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')


class RecurrentReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_episode_len, capacity, batch_size):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_len = max_episode_len
        self.capacity = capacity
        self.batch_size = batch_size
        self.ptr = 0
        self.num_episodes = 0

        self.obs_buf = np.zeros((capacity, max_episode_len + 1, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)
        self.mask_buf = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)

        self.curr_obs = []
        self.curr_act = []
        self.curr_rew = []
        self.curr_done = []

    # Added *args to accept the extra 'False' argument from original code's store_episode
    def push(self, obs, action, reward, next_obs, done, *args):
        self.curr_obs.append(obs)
        self.curr_act.append(action)
        self.curr_rew.append(reward)
        self.curr_done.append(done)

        if done:
            T = len(self.curr_obs)
            if T > self.max_episode_len: T = self.max_episode_len

            self.curr_obs.append(next_obs)

            idx = self.ptr
            self.obs_buf[idx] = 0
            self.act_buf[idx] = 0
            self.rew_buf[idx] = 0
            self.done_buf[idx] = 0
            self.mask_buf[idx] = 0

            length = min(T, self.max_episode_len)

            self.obs_buf[idx, :length + 1] = np.array(self.curr_obs[:length + 1])
            self.act_buf[idx, :length] = np.array(self.curr_act[:length]).reshape(-1, 1)
            self.rew_buf[idx, :length] = np.array(self.curr_rew[:length]).reshape(-1, 1)
            self.done_buf[idx, :length] = np.array(self.curr_done[:length]).reshape(-1, 1)
            self.mask_buf[idx, :length] = 1.0

            self.ptr = (self.ptr + 1) % self.capacity
            self.num_episodes = min(self.num_episodes + 1, self.capacity)

            self.curr_obs = []
            self.curr_act = []
            self.curr_rew = []
            self.curr_done = []

    def sample(self):
        if self.num_episodes < self.batch_size:
            return None

        indices = np.random.choice(self.num_episodes, self.batch_size, replace=False)

        return RecurrentBatch(
            self.obs_buf[indices],
            self.act_buf[indices],
            self.rew_buf[indices],
            self.done_buf[indices],
            self.mask_buf[indices]
        )

    def reset(self):
        self.curr_obs = []
        self.curr_act = []
        self.curr_rew = []
        self.curr_done = []