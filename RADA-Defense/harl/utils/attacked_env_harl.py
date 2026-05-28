"""
Attacked Environment for HARL-based MAPPO Systems.
====================================================

将 PettingZoo 多智能体环境包装为 Gymnasium 单智能体环境，
用于训练对抗攻击策略（ACT / DYN）。

设计思路：
    - victim agent 由外部 RL 算法控制（攻击者）
    - 其余 agent 使用冻结的 MAPPO 共享策略
    - 攻击者的奖励 = -团队奖励 + λ × 隐蔽性奖励

支持模式：
    - ACT (attack_lambda=None): 纯动作替换，不考虑 Tracker
    - DYN (attack_lambda>0):  带隐蔽性约束，Tracker 感知的攻击

支持环境：
    - simple_spread_v2/v3 (N=6): obs_dim=36, action_dim=5
    - simple_tag_v2/v3 (6 adversaries + 1 prey): obs_dim=22, action_dim=5

优化 (v2):
    - 批量推理：将所有正常 agent 的观测拼成一个 batch 一次性推理
    - 减少 CPU↔GPU 数据搬运次数（从每步 5 次降为每步 1 次）
    - 预分配 rnn_states/masks 张量，避免重复创建
"""

import gymnasium as gym
import numpy as np
import torch
import os
import json
import copy


# ============================================================================
# PettingZoo 环境创建
# ============================================================================

def create_pettingzoo_env(env_args=None, N=6, max_cycles=25):
    """创建 PettingZoo 环境。支持 simple_spread 和 simple_tag。

    Args:
        env_args: 环境配置字典，包含 scenario 等参数
        N: 智能体数量（simple_spread 用）
        max_cycles: 每个 episode 的最大步数

    Returns:
        PettingZoo parallel env 或兼容的 wrapper
    """
    scenario = env_args.get("scenario", "simple_spread_v2") if env_args else "simple_spread_v2"

    if "simple_tag" in scenario:
        return SimpleTagParallelWrapper(env_args, max_cycles=max_cycles)
    else:
        try:
            from pettingzoo.mpe import simple_spread_v3 as simple_spread
        except ImportError:
            from pettingzoo.mpe import simple_spread_v2 as simple_spread
        env = simple_spread.parallel_env(
            N=N, max_cycles=max_cycles, continuous_actions=True,
        )
        return env


class SimpleTagParallelWrapper:
    """让 SimpleTagAdversaryEnv 兼容 AttackedEnvHARL 期望的 PettingZoo parallel_env 接口。

    AttackedEnvHARL 期望：
      - env.possible_agents: agent 名称列表
      - env.observation_space(agent_name): 观测空间
      - env.action_space(agent_name): 动作空间
      - env.reset() → (obs_dict, info_dict)
      - env.step(action_dict) → (obs_dict, rew_dict, term_dict, trunc_dict, info_dict)
    """

    def __init__(self, env_args, max_cycles=50):
        from harl.envs.pettingzoo_mpe.simple_tag_env import SimpleTagAdversaryEnv
        tag_args = {
            "scenario": env_args.get("scenario", "simple_tag_v2"),
            "num_good": env_args.get("num_good", 1),
            "num_adversaries": env_args.get("num_adversaries", 6),
            "num_obstacles": env_args.get("num_obstacles", 2),
            "max_cycles": max_cycles,
            "continuous_actions": True,
        }
        self._inner = SimpleTagAdversaryEnv(tag_args)

        self.possible_agents = self._inner.adversary_agents
        self.agents = self._inner.adversary_agents
        self.num_agents = self._inner.n_agents

        self._obs_spaces = {name: self._inner.observation_space[i]
                            for i, name in enumerate(self.possible_agents)}
        self._act_spaces = {name: self._inner.action_space[i]
                            for i, name in enumerate(self.possible_agents)}

    def observation_space(self, agent_name):
        return self._obs_spaces[agent_name]

    def action_space(self, agent_name):
        return self._act_spaces[agent_name]

    def reset(self, seed=None):
        obs_list, _, _ = self._inner.reset()
        obs_dict = {name: obs_list[i] for i, name in enumerate(self.possible_agents)}
        info_dict = {name: {} for name in self.possible_agents}
        return obs_dict, info_dict

    def step(self, action_dict):
        actions = [action_dict[name] for name in self.possible_agents]
        obs_list, _, rewards, dones, infos, _ = self._inner.step(actions)

        obs_dict = {name: obs_list[i] for i, name in enumerate(self.possible_agents)}
        rew_dict = {name: rewards[i][0] for i, name in enumerate(self.possible_agents)}
        term_dict = {name: dones[i] for i, name in enumerate(self.possible_agents)}
        trunc_dict = {name: False for name in self.possible_agents}
        info_dict = {name: infos[i] if isinstance(infos[i], dict) else {}
                     for i, name in enumerate(self.possible_agents)}

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def render(self):
        self._inner.render()

    def close(self):
        self._inner.close()


# ============================================================================
# HARL Actor 加载工具
# ============================================================================

def load_harl_actor(model_dir, config_path=None, obs_dim=36, action_dim=5, device="cpu"):
    """加载 HARL 训练好的 MAPPO Actor。

    Args:
        model_dir: 模型保存目录，包含 actor_agent0.pt 等文件
        config_path: config.json 的路径（如果有的话，可以自动读取参数）
        obs_dim: 观测维度
        action_dim: 动作维度
        device: 计算设备

    Returns:
        actor: HARL MAPPO actor 对象（已加载权重，eval 模式）
    """
    # 构建 args 字典
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        algo_args = config["algo_args"]
        args = {**algo_args["model"], **algo_args["algo"]}
    else:
        # 使用默认参数（与 mappo.yaml 一致）
        args = {
            # model 参数
            "hidden_sizes": [128, 128],
            "activation_func": "relu",
            "use_feature_normalization": True,
            "initialization_method": "orthogonal_",
            "gain": 0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy": False,
            "recurrent_n": 1,
            "data_chunk_length": 10,
            "lr": 0.0005,
            "critic_lr": 0.0005,
            "opti_eps": 1e-5,
            "weight_decay": 0,
            "std_x_coef": 1,
            "std_y_coef": 0.5,
            # algo 参数
            "ppo_epoch": 5,
            "clip_param": 0.2,
            "actor_num_mini_batch": 1,
            "critic_epoch": 5,
            "critic_num_mini_batch": 1,
            "entropy_coef": 0.01,
            "value_loss_coef": 1,
            "use_max_grad_norm": True,
            "max_grad_norm": 10.0,
            "use_gae": True,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "use_huber_loss": True,
            "use_clipped_value_loss": True,
            "use_policy_active_masks": True,
            "huber_delta": 10.0,
            "action_aggregation": "prod",
            "share_param": True,
            "fixed_order": True,
        }

    # 创建观测/动作空间
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    act_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
    )

    # 通过 HARL 注册表创建 Actor
    from harl.algorithms.actors import ALGO_REGISTRY
    actor = ALGO_REGISTRY["mappo"](
        args, obs_space, act_space, device=torch.device(device)
    )

    # 加载权重（share_param=True 时所有 agent 共享 actor_agent0.pt）
    weight_path = os.path.join(model_dir, "actor_agent0.pt")
    state_dict = torch.load(weight_path, map_location=device)
    # state_dict = torch.load(weight_path, map_location="cpu")
    actor.actor.load_state_dict(state_dict)
    actor.actor.eval()

    # 冻结所有参数
    for param in actor.actor.parameters():
        param.requires_grad = False

    print(f"[AttackedEnv] 已加载 MAPPO Actor: {weight_path} (device={device})")
    return actor


def load_recl(model_dir, obs_dim=36, num_agents=6, device="cpu",
              agent_embedding_dim=64, role_embedding_dim=32, cluster_num=2):
    """加载 Phase 1 训练好的 ReCL 模块。"""
    from harl.algorithms.recl.recl import ReCL

    recl = ReCL(
        obs_dim=obs_dim,
        num_agents=num_agents,
        agent_embedding_dim=agent_embedding_dim,
        role_embedding_dim=role_embedding_dim,
        cluster_num=cluster_num,
        device=torch.device(device),
    )

    recl_path = os.path.join(model_dir, "recl.pt")
    if os.path.exists(recl_path):
        recl.load(recl_path)
        print(f"[AttackedEnv] 已加载 ReCL: {recl_path}")
    else:
        raise FileNotFoundError(f"ReCL 模型不存在: {recl_path}")

    recl.prep_rollout()
    return recl


def _detect_ablation_variant(tracker_dir):
    """从 tracker 目录中自动检测消融变体。"""
    config_path = os.path.join(tracker_dir, "tracker_ablation_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variant = config.get("ablation_variant", "full")
        print(f"[AttackedEnv] 检测到消融变体: {variant}")
        return variant
    print(f"[AttackedEnv] 未找到 ablation config，默认 full")
    return "full"


def load_tracker(tracker_dir, num_agents=6, obs_dim=36, action_dim=5,
                 role_embedding_dim=32, hidden_dim=128, cluster_num=2,
                 device="cpu"):
    """加载 Phase 2 训练好的 Tracker。"""
    from harl.algorithms.trackers.tracker import DecentralizedTracker

    # ---- 自动检测消融变体 ----
    ablation_variant = _detect_ablation_variant(tracker_dir)

    tracker = DecentralizedTracker(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        role_embedding_dim=role_embedding_dim,
        hidden_dim=hidden_dim,
        cluster_num=cluster_num,
        device=torch.device(device),
        ablation_variant=ablation_variant,  # ← 关键修复
    )

    tracker.load(tracker_dir)

    # 也加载 cluster 信息（可能在 tracker_dir 的上级目录）
    parent_dir = os.path.dirname(tracker_dir)
    cluster_path = os.path.join(parent_dir, "cluster_centers.npy")
    if os.path.exists(cluster_path):
        centers = np.load(cluster_path)
        threshold_path = os.path.join(parent_dir, "cluster_thresholds.npy")
        thresholds = np.load(threshold_path) if os.path.exists(threshold_path) else None
        tracker.load_cluster_info(centers, thresholds)

    tracker.prep_rollout()
    print(f"[AttackedEnv] 已加载 Tracker: {tracker_dir} (ablation={ablation_variant})")
    return tracker



# ============================================================================
# 核心：被攻击环境
# ============================================================================

class AttackedEnvHARL(gym.Env):
    """将多智能体环境包装为单智能体 Gym 环境，用于训练攻击策略。

    N-1 个 agent 使用冻结的 MAPPO 策略自动控制，
    1 个 agent（victim）由外部 RL 算法（攻击者）控制。

    Args:
        harl_actor: 已加载的 HARL MAPPO actor（共享参数）
        victim_idx: 被攻击的 agent 索引
        num_agents: 智能体总数
        max_cycles: 每个 episode 的最大步数
        attack_lambda: 隐蔽性权重。None=ACT，>0=DYN
        tracker: Phase 2 训练好的 DecentralizedTracker（DYN 模式需要）
        recl: Phase 1 训练好的 ReCL 模块（DYN 模式需要）
        device: 计算设备
        env_args: 环境配置字典（包含 scenario 等）
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            harl_actor,
            victim_idx=0,
            num_agents=6,
            max_cycles=25,
            attack_lambda=None,
            tracker=None,
            recl=None,
            device="cpu",
            env_args=None,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.victim_idx = victim_idx
        self.max_cycles = max_cycles
        self.attack_lambda = attack_lambda
        self.device = device

        # ---- 创建底层 PettingZoo 环境 ----
        self.marl_env = create_pettingzoo_env(
            env_args=env_args, N=num_agents, max_cycles=max_cycles
        )
        self.agent_names = self.marl_env.possible_agents
        self.victim_name = self.agent_names[victim_idx]

        # ---- Gym 空间定义 ----
        self.observation_space = self.marl_env.observation_space(self.victim_name)
        self.action_space = self.marl_env.action_space(self.victim_name)
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # ---- HARL Actor（冻结的共享策略）----
        self.harl_actor = harl_actor

        # MAPPO 使用前馈网络（非 RNN），但 HARL 接口仍需要 rnn_states
        self.recurrent_n = 1
        self.rnn_hidden_size = 128  # hidden_sizes[-1]

        # ---- DYN 攻击相关 ----
        self.use_dyn = (attack_lambda is not None and attack_lambda > 0)
        if self.use_dyn:
            assert tracker is not None, "DYN 模式需要提供 Tracker"
            assert recl is not None, "DYN 模式需要提供 ReCL"
            self.tracker = tracker
            self.recl = recl
        else:
            self.tracker = None
            self.recl = None

        # ---- 运行时状态 ----
        self.last_obs = None
        self.last_actions = None  # for Tracker

        # ---- [优化] 预分配正常 agent 索引列表 ----
        self.normal_agent_indices = [i for i in range(num_agents) if i != victim_idx]
        self.num_normal = len(self.normal_agent_indices)

        # ---- [优化] 预分配批量推理用的 rnn_states 和 masks ----
        # 形状: (num_normal, recurrent_n, rnn_hidden_size)
        self.batch_rnn_states = np.zeros(
            (self.num_normal, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        self.batch_masks = np.ones((self.num_normal, 1), dtype=np.float32)

        # 验证维度
        print(f"[AttackedEnv] obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
              f"victim={self.victim_name}(idx={victim_idx}), "
              f"mode={'DYN(λ=' + str(attack_lambda) + ')' if self.use_dyn else 'ACT'}, "
              f"device={device}, batch_normal={self.num_normal}")

    def reset(self, *, seed=None, options=None):
        """重置环境。"""
        result = self.marl_env.reset(seed=seed)
        if isinstance(result, tuple):
            obs_dict, info_dict = result
        else:
            obs_dict = result
            info_dict = {agent: {} for agent in self.agent_names}

        self.last_obs = obs_dict

        # [优化] rnn_states 和 masks 重置为零/一（就地操作，不重新分配）
        self.batch_rnn_states[:] = 0.0
        self.batch_masks[:] = 1.0

        # DYN: 重置 Tracker 和 ReCL 状态
        if self.use_dyn:
            self.tracker.init_hidden(1)
            self.tracker.reset_detection_stats()
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None
            self.last_actions = np.zeros((1, self.num_agents, self.action_dim))

        victim_obs = obs_dict[self.victim_name]
        return victim_obs.astype(np.float32), {}

    def step(self, attack_action):
        """执行一步。

        [优化] 将所有正常 agent 的观测拼成 batch，一次性推理，
        减少 CPU↔GPU 数据搬运从 5 次/step 降为 1 次/step。

        Args:
            attack_action: 攻击者为 victim 选择的动作, shape (action_dim,)

        Returns:
            obs, reward, terminated, truncated, info
        """
        team_action = {}
        all_actions_list = [None] * self.num_agents  # 预分配，按索引填充

        # ---- 受害者使用攻击动作 ----
        victim_action = np.array(attack_action, dtype=np.float32).flatten()
        victim_action = np.clip(victim_action, self.action_space.low, self.action_space.high)
        team_action[self.victim_name] = victim_action
        all_actions_list[self.victim_idx] = victim_action.copy()

        # ---- [优化] 批量收集正常 agent 观测 ----
        batch_obs = np.stack([
            self.last_obs[self.agent_names[i]].astype(np.float32)
            for i in self.normal_agent_indices
        ])  # shape: (num_normal, obs_dim)

        # ---- [优化] 一次性批量推理 ----
        with torch.no_grad():
            batch_action_tensor, batch_rnn_out = self.harl_actor.act(
                batch_obs,                # (num_normal, obs_dim)
                self.batch_rnn_states,    # (num_normal, recurrent_n, hidden)
                self.batch_masks,         # (num_normal, 1)
                None,                     # available_actions
                deterministic=False,
            )

        # 一次性搬回 CPU
        batch_actions = batch_action_tensor.cpu().numpy()  # (num_normal, action_dim)

        # 分发到各 agent
        for local_idx, global_idx in enumerate(self.normal_agent_indices):
            action = np.clip(batch_actions[local_idx], self.action_space.low, self.action_space.high)
            team_action[self.agent_names[global_idx]] = action
            all_actions_list[global_idx] = action

        # ---- 环境步进 ----
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.marl_env.step(team_action)
        self.last_obs = obs_dict

        # ---- 计算团队奖励 ----
        team_reward = sum(reward_dict.values())

        # ---- 计算隐蔽性奖励（DYN 模式）----
        stealth_reward = 0.0
        if self.use_dyn:
            stealth_reward = self._compute_stealth_reward(
                obs_dict, all_actions_list, victim_action
            )

        # ---- 攻击者总奖励 ----
        # -team_reward: 攻击者希望最小化团队表现
        # +stealth_reward: 攻击者希望最大化隐蔽性
        total_reward = -team_reward + stealth_reward

        # ---- 终止判断 ----
        terminated = any(term_dict.values())
        truncated = any(trunc_dict.values())

        victim_obs = obs_dict[self.victim_name].astype(np.float32)
        return victim_obs, float(total_reward), terminated, truncated, {}

    def _compute_stealth_reward(self, obs_dict, all_actions_list, victim_action):
        """计算 DYN 攻击的隐蔽性奖励。

        通过 ReCL 获取角色嵌入，通过 Tracker 预测动作分布，
        然后计算受害者实际动作与预测分布的匹配度。
        匹配度越高（log_prob 越大）= 越隐蔽 = 奖励越高。

        注意：训练时不使用分组过滤（use_grouping=False），
        让攻击者学习欺骗所有 observer，这是更难的任务。

        使用公式（与原版 attacked_env.py 对齐）：
            z_i = log P(a_victim | μ_{i→victim}, σ_{i→victim})
            stealth = -Σ|z_i + action_dim/2|
            reward = λ × stealth
        """
        # 构建观测张量 (1, N, obs_dim)
        obs_array = np.stack([
            obs_dict[agent] for agent in self.agent_names
        ]).reshape(1, self.num_agents, -1).astype(np.float32)

        # 构建当前动作张量 (1, N, action_dim)
        curr_actions = np.stack(all_actions_list).reshape(
            1, self.num_agents, -1
        ).astype(np.float32)

        obs_tensor = torch.tensor(obs_array, device=self.device)
        last_act_tensor = torch.tensor(self.last_actions, dtype=torch.float32, device=self.device)

        # 通过 ReCL 获取角色嵌入
        with torch.no_grad():
            ae = self.recl.embedding_net.agent_embed_forward(
                obs_tensor.reshape(-1, self.obs_dim), detach=True
            )
            role_embed = self.recl.embedding_net.role_embed_forward(
                ae, detach=True, ema=False
            ).reshape(1, self.num_agents, -1)

        # Tracker 前向推理
        mu, std, _ = self.tracker.forward(obs_tensor, last_act_tensor, role_embed)

        # 计算异常分数（log probability）
        # ★ 训练时不使用分组过滤，让攻击者学习欺骗所有 observer
        action_tensor = torch.tensor(curr_actions, dtype=torch.float32, device=self.device)
        scores = self.tracker.compute_scores(mu, std, action_tensor, use_grouping=False)
        # scores shape: (1, N, N), scores[0, i, j] = observer i 对 target j 的评分

        # 提取其他 agent 对 victim 的评分
        victim_scores = scores[0, :, self.victim_idx]  # shape: (N,)
        victim_scores = np.delete(victim_scores, self.victim_idx)  # 去掉自我预测

        # 与原版对齐: z = -|score + action_dim/2|
        # 当 score ≈ -action_dim/2 时为最优（隐蔽性最好），stealth=0
        z = -np.abs(victim_scores + self.action_dim / 2.0)
        stealth_reward = float(np.sum(z) * self.attack_lambda)

        # 更新 last_actions
        self.last_actions = curr_actions

        return stealth_reward

    def render(self):
        return self.marl_env.render()

    def close(self):
        self.marl_env.close()


# ============================================================================
# 辅助：创建环境的工厂函数
# ============================================================================

def make_attacked_env(
        phase1_model_dir,
        victim_idx=0,
        num_agents=6,
        max_cycles=25,
        attack_lambda=None,
        phase2_model_dir=None,
        config_path=None,
        device="cpu",
        env_args=None,
):
    """一键创建 AttackedEnv 的工厂函数。

    Args:
        phase1_model_dir: Phase 1 模型目录（包含 actor_agent*.pt, recl.pt）
        victim_idx: 被攻击的 agent 索引
        num_agents: 智能体数量
        max_cycles: episode 最大步数
        attack_lambda: None=ACT, >0=DYN
        phase2_model_dir: Phase 2 模型目录（DYN 模式需要，包含 tracker/ 子目录）
        config_path: config.json 路径
        device: 计算设备
        env_args: 环境配置字典（包含 scenario 等）

    Returns:
        AttackedEnvHARL instance
    """
    # 自动检测环境类型和维度
    if env_args is None:
        env_args = {"scenario": "simple_spread_v2"}

    scenario = env_args.get("scenario", "simple_spread_v2")

    if "simple_tag" in scenario:
        obs_dim = 22
        action_dim = 5
    else:
        obs_dim = 36
        action_dim = 5

    role_embedding_dim = 32

    # 加载 MAPPO Actor
    actor = load_harl_actor(
        phase1_model_dir, config_path=config_path,
        obs_dim=obs_dim, action_dim=action_dim, device=device
    )

    # DYN 模式：加载 Tracker + ReCL
    tracker = None
    recl = None
    if attack_lambda is not None and attack_lambda > 0:
        if phase2_model_dir is None:
            phase2_model_dir = phase1_model_dir

        recl = load_recl(
            phase1_model_dir, obs_dim=obs_dim, num_agents=num_agents, device=device,
            role_embedding_dim=role_embedding_dim,
        )

        tracker_dir = os.path.join(phase2_model_dir, "tracker")
        if not os.path.exists(tracker_dir):
            raise FileNotFoundError(
                f"Tracker 目录不存在: {tracker_dir}\n"
                f"DYN 攻击需要 Phase 2 训练好的 Tracker。"
            )
        tracker = load_tracker(
            tracker_dir, num_agents=num_agents, obs_dim=obs_dim,
            action_dim=action_dim, role_embedding_dim=role_embedding_dim,
            device=device,
        )

    return AttackedEnvHARL(
        harl_actor=actor,
        victim_idx=victim_idx,
        num_agents=num_agents,
        max_cycles=max_cycles,
        attack_lambda=attack_lambda,
        tracker=tracker,
        recl=recl,
        device=device,
        env_args=env_args,
    )
