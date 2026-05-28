"""
Simple Tag Environment Wrapper for HARL
========================================

Wraps PettingZoo simple_tag as a cooperative MARL env for the adversary team only.
- Only adversary agents (predators) are exposed to HARL
- Prey (good agent) uses a heuristic escape policy automatically
- Adversary team reward is shared (already cooperative in original env)
- Compatible with HARL's PettingZooMPEEnv interface

Usage in HARL config:
    env_name: "pettingzoo_mpe"
    scenario: "simple_tag_v2"
    num_good: 1
    num_adversaries: 6
    num_obstacles: 2
    max_cycles: 50
    continuous_actions: true
"""

import copy
import importlib
import logging
import numpy as np
import supersuit as ss
from gymnasium import spaces

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class SimpleTagAdversaryEnv:
    """Simple Tag env that only exposes adversary agents to HARL.

    Prey uses a heuristic policy (flee from nearest predator).
    Adversaries are trained cooperatively with shared reward.

    Interface is identical to PettingZooMPEEnv so it's a drop-in replacement.
    """

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]

        # Parse config
        self.num_adversaries = self.args.get("num_adversaries", 6)
        self.num_good = self.args.get("num_good", 1)
        self.num_obstacles = self.args.get("num_obstacles", 2)

        # Continuous actions
        self.discrete = True
        if self.args.get("continuous_actions", False):
            self.discrete = False

        # Max cycles
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 50
            self.args["max_cycles"] = 51

        self.cur_step = 0

        # Create PettingZoo parallel env
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)

        # Build env args (only pass what PettingZoo expects)
        pz_args = {
            "num_good": self.num_good,
            "num_adversaries": self.num_adversaries,
            "num_obstacles": self.num_obstacles,
            "max_cycles": self.args["max_cycles"],
            "continuous_actions": not self.discrete,
        }
        if "render_mode" in self.args:
            pz_args["render_mode"] = self.args["render_mode"]

        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**pz_args))
        )
        self.env.reset()

        # Identify adversary and prey agents
        self.all_agents = self.env.agents
        self.adversary_agents = [a for a in self.all_agents if "adversary" in a]
        self.prey_agents = [a for a in self.all_agents if "agent" in a]

        assert len(self.adversary_agents) == self.num_adversaries, \
            f"Expected {self.num_adversaries} adversaries, got {len(self.adversary_agents)}"
        assert len(self.prey_agents) == self.num_good, \
            f"Expected {self.num_good} prey, got {len(self.prey_agents)}"

        print(f"[SimpleTagAdversaryEnv] Adversaries: {self.adversary_agents}")
        print(f"[SimpleTagAdversaryEnv] Prey (heuristic): {self.prey_agents}")
        print(f"[SimpleTagAdversaryEnv] Obstacles: {self.num_obstacles}")

        # HARL interface: only expose adversary agents
        self.n_agents = self.num_adversaries
        self.agents = self.adversary_agents

        # Observation and action spaces (only for adversaries)
        self.observation_space = [self.env.observation_spaces[a] for a in self.adversary_agents]
        self.action_space = [self.env.action_spaces[a] for a in self.adversary_agents]

        # Shared observation (global state)
        self.share_observation_space = self._build_share_obs_space()

        self._seed = 0

        # Cache for prey policy
        self._last_obs = None

    def _build_share_obs_space(self):
        """Build shared observation space for centralized critic."""
        # Use full state from env
        state_space = self.env.state_space
        return [state_space for _ in range(self.n_agents)]

    def _prey_heuristic_action(self, obs_dict):
        """Heuristic prey policy: flee from nearest adversary.

        Prey observation: [self_vel(2), self_pos(2), landmark_rel_pos(num_obs*2),
                           other_agent_rel_pos(num_adv*2), ...]

        Strategy: Move away from the nearest adversary.
        """
        actions = {}
        for prey_name in self.prey_agents:
            obs = obs_dict.get(prey_name, None)
            if obs is None:
                # Prey might be done
                if self.discrete:
                    actions[prey_name] = 0  # no action
                else:
                    actions[prey_name] = np.zeros(self.env.action_spaces[prey_name].shape[0])
                continue

            # Parse observation
            # obs = [self_vel(2), self_pos(2), landmark_rel_pos(num_obs*2),
            #        other_agent_rel_pos((num_adv + num_good - 1)*2), ...]
            offset = 4 + self.num_obstacles * 2  # skip vel, pos, landmarks

            # Find nearest adversary from relative positions
            min_dist = float('inf')
            escape_dir = np.zeros(2)

            for i in range(self.num_adversaries):
                idx = offset + i * 2
                if idx + 1 < len(obs):
                    rel_pos = obs[idx:idx + 2]  # relative position of adversary
                    dist = np.linalg.norm(rel_pos)
                    if dist < min_dist:
                        min_dist = dist
                        # Escape direction: opposite of relative position
                        if dist > 1e-6:
                            escape_dir = -rel_pos / dist
                        else:
                            escape_dir = np.random.randn(2)
                            escape_dir /= np.linalg.norm(escape_dir) + 1e-6

            if self.discrete:
                # Convert escape direction to discrete action
                # 0: no_action, 1: left, 2: right, 3: down, 4: up
                if abs(escape_dir[0]) > abs(escape_dir[1]):
                    actions[prey_name] = 1 if escape_dir[0] < 0 else 2  # left/right
                else:
                    actions[prey_name] = 3 if escape_dir[1] < 0 else 4  # down/up
            else:
                # Continuous: [no_action, left, right, down, up]
                action = np.zeros(self.env.action_spaces[prey_name].shape[0])
                # Add some noise for unpredictability
                noise = np.random.randn(2) * 0.1
                escape = escape_dir + noise

                action[1] = max(0, -escape[0])  # left
                action[2] = max(0, escape[0])  # right
                action[3] = max(0, -escape[1])  # down
                action[4] = max(0, escape[1])  # up
                actions[prey_name] = np.clip(action, 0, 1)

        return actions

    def step(self, actions):
        """Step with adversary actions + heuristic prey actions.

        Args:
            actions: adversary actions only, shape depends on discrete/continuous

        Returns:
            local_obs, global_state, rewards, dones, infos, available_actions
            (same interface as PettingZooMPEEnv)
        """
        # Build full action dict
        full_actions = {}

        # Adversary actions from HARL
        for i, agent_name in enumerate(self.adversary_agents):
            if self.discrete:
                full_actions[agent_name] = actions.flatten()[i]
            else:
                full_actions[agent_name] = actions[i]

        # Prey actions from heuristic
        prey_actions = self._prey_heuristic_action(self._last_obs or {})
        full_actions.update(prey_actions)

        # Step environment
        obs, rew, term, trunc, info = self.env.step(full_actions)
        self._last_obs = obs
        self.cur_step += 1

        # Handle max cycles
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.all_agents}
            for agent in self.all_agents:
                if agent in info:
                    info[agent]["bad_transition"] = True
                else:
                    info[agent] = {"bad_transition": True}

        # Compute dones (adversary only)
        dones = []
        for agent in self.adversary_agents:
            d = term.get(agent, False) or trunc.get(agent, False)
            dones.append(d)

        # Compute shared reward for adversary team
        # In simple_tag, all adversaries get the same reward when any one catches prey
        adv_rewards = [rew.get(agent, 0.0) for agent in self.adversary_agents]
        total_reward = sum(adv_rewards)
        rewards = [[total_reward]] * self.n_agents

        # Extract adversary observations
        adv_obs = [obs.get(agent, np.zeros(self.observation_space[i].shape))
                   for i, agent in enumerate(self.adversary_agents)]

        # Global state
        s_obs = [self.env.state() for _ in range(self.n_agents)]

        # Info
        adv_info = [info.get(agent, {}) for agent in self.adversary_agents]

        return (
            adv_obs,
            s_obs,
            rewards,
            dones,
            adv_info,
            self.get_avail_actions(),
        )

    def reset(self):
        """Reset environment."""
        self._seed += 1
        self.cur_step = 0
        obs = self.env.reset(seed=self._seed)

        # Handle both dict and tuple returns
        if isinstance(obs, tuple):
            obs = obs[0]  # (obs, info) format

        self._last_obs = obs

        # Extract adversary observations
        adv_obs = [obs[agent] for agent in self.adversary_agents]

        # Global state
        s_obs = [self.env.state() for _ in range(self.n_agents)]

        return adv_obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            return [[1] * self.action_space[i].n for i in range(self.n_agents)]
        else:
            return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
