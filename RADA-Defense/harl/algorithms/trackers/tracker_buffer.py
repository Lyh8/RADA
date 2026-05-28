import numpy as np
from collections import namedtuple


TrackerBatch = namedtuple('TrackerBatch', [
    'obs',
    'actions',
    'role_embeddings',
    'masks',
    'episode_lengths',
])


class TrackerReplayBuffer:

    def __init__(
        self,
        obs_dim,
        action_dim,
        role_embedding_dim,
        num_agents,
        max_episode_len,
        capacity=1000,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.role_embedding_dim = role_embedding_dim
        self.num_agents = num_agents
        self.max_episode_len = max_episode_len
        self.capacity = capacity

        self.obs = np.zeros((capacity, max_episode_len, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, max_episode_len, num_agents, action_dim), dtype=np.float32)
        self.role_embeddings = np.zeros((capacity, max_episode_len, num_agents, role_embedding_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, max_episode_len, num_agents), dtype=np.float32)
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)

        self.episode_ptr = 0
        self.num_episodes = 0

        self._current_episode = {
            'obs': [],
            'actions': [],
            'role_embeddings': [],
            'masks': [],
        }

    def start_episode(self):
        self._current_episode = {
            'obs': [],
            'actions': [],
            'role_embeddings': [],
            'masks': [],
        }

    def add_step(self, obs, action, role_embedding, mask):
        self._current_episode['obs'].append(obs)
        self._current_episode['actions'].append(action)
        self._current_episode['role_embeddings'].append(role_embedding)

        if mask.ndim == 2:
            mask = mask.squeeze(-1)
        self._current_episode['masks'].append(mask)

    def end_episode(self):
        if len(self._current_episode['obs']) == 0:
            return

        ep_len = len(self._current_episode['obs'])

        actual_len = min(ep_len, self.max_episode_len)

        idx = self.episode_ptr

        self.obs[idx, :actual_len] = np.array(self._current_episode['obs'][:actual_len])
        self.actions[idx, :actual_len] = np.array(self._current_episode['actions'][:actual_len])
        self.role_embeddings[idx, :actual_len] = np.array(self._current_episode['role_embeddings'][:actual_len])
        self.masks[idx, :actual_len] = np.array(self._current_episode['masks'][:actual_len])

        if actual_len < self.max_episode_len:
            self.obs[idx, actual_len:] = 0
            self.actions[idx, actual_len:] = 0
            self.role_embeddings[idx, actual_len:] = 0
            self.masks[idx, actual_len:] = 0

        self.episode_lengths[idx] = actual_len

        self.episode_ptr = (self.episode_ptr + 1) % self.capacity
        self.num_episodes = min(self.num_episodes + 1, self.capacity)

        self._current_episode = {
            'obs': [],
            'actions': [],
            'role_embeddings': [],
            'masks': [],
        }

    def push_episode(self, obs, actions, role_embeddings, masks):
        ep_len = obs.shape[0]
        actual_len = min(ep_len, self.max_episode_len)

        idx = self.episode_ptr

        self.obs[idx, :actual_len] = obs[:actual_len]
        self.actions[idx, :actual_len] = actions[:actual_len]
        self.role_embeddings[idx, :actual_len] = role_embeddings[:actual_len]
        self.masks[idx, :actual_len] = masks[:actual_len]

        if actual_len < self.max_episode_len:
            self.obs[idx, actual_len:] = 0
            self.actions[idx, actual_len:] = 0
            self.role_embeddings[idx, actual_len:] = 0
            self.masks[idx, actual_len:] = 0

        self.episode_lengths[idx] = actual_len

        self.episode_ptr = (self.episode_ptr + 1) % self.capacity
        self.num_episodes = min(self.num_episodes + 1, self.capacity)

    def sample(self, batch_size):
        if self.num_episodes < batch_size:
            indices = np.random.choice(self.num_episodes, batch_size, replace=True)
        else:
            indices = np.random.choice(self.num_episodes, batch_size, replace=False)

        return TrackerBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            role_embeddings=self.role_embeddings[indices].copy(),
            masks=self.masks[indices].copy(),
            episode_lengths=self.episode_lengths[indices].copy(),
        )

    def get_all(self):
        indices = np.arange(self.num_episodes)

        return TrackerBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            role_embeddings=self.role_embeddings[indices].copy(),
            masks=self.masks[indices].copy(),
            episode_lengths=self.episode_lengths[indices].copy(),
        )

    def clear(self):
        self.obs.fill(0)
        self.actions.fill(0)
        self.role_embeddings.fill(0)
        self.masks.fill(0)
        self.episode_lengths.fill(0)

        self.episode_ptr = 0
        self.num_episodes = 0

        self._current_episode = {
            'obs': [],
            'actions': [],
            'role_embeddings': [],
            'masks': [],
        }

    def __len__(self):
        return self.num_episodes
