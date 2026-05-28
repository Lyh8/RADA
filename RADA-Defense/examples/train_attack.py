import argparse
import os
import json
import numpy as np
from datetime import datetime

from harl.utils.attacked_env_harl import make_attacked_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train attack policy against HARL MAPPO")

    parser.add_argument("--phase1_model_dir", type=str, required=True,
                        help="Phase 1 model directory (contains actor_agent*.pt, recl.pt)")
    parser.add_argument("--phase2_model_dir", type=str, default=None,
                        help="Phase 2 model directory (contains tracker/ subdir, required for DYN mode)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="config.json path (optional, for auto-reading training params)")

    parser.add_argument("--attack_type", type=str, default="ACT",
                        choices=["ACT", "DYN"],
                        help="Attack type: ACT (pure action replacement) or DYN (stealthy attack)")
    parser.add_argument("--attack_lambda", type=float, default=10.0,
                        help="Stealthiness weight λ for DYN attack (only effective in DYN mode)")
    parser.add_argument("--victim_idx", type=int, default=3,
                        help="Index of the attacked agent")

    parser.add_argument("--scenario", type=str, default="simple_spread_v2",
                        help="Environment scenario: simple_spread_v2 or simple_tag_v2")
    parser.add_argument("--num_agents", type=int, default=6,
                        help="Number of agents")
    parser.add_argument("--max_cycles", type=int, default=25,
                        help="Max steps per episode (simple_spread=25, simple_tag=50)")

    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=128,
                        help="Steps collected per rollout (per env)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--use_lstm", action="store_true",
                        help="Use LSTM policy (requires sb3-contrib)")

    parser.add_argument("--output_dir", type=str, default="trained_attacks",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Logging verbosity")

    return parser.parse_args()


def build_env_args(args):
    scenario = args.scenario

    if "simple_tag" in scenario:
        env_args = {
            "scenario": scenario,
            "num_good": 1,
            "num_adversaries": args.num_agents,
            "num_obstacles": 2,
        }
    else:
        env_args = {
            "scenario": scenario,
        }

    return env_args


def make_env_fn(args, env_args, rank=0):

    def _init():
        lambda_val = args.attack_lambda if args.attack_type == "DYN" else None

        env = make_attacked_env(
            phase1_model_dir=args.phase1_model_dir,
            victim_idx=args.victim_idx,
            num_agents=args.num_agents,
            max_cycles=args.max_cycles,
            attack_lambda=lambda_val,
            phase2_model_dir=args.phase2_model_dir,
            config_path=args.config_path,
            device=args.device,
            env_args=env_args,
        )
        return env

    return _init


def train_with_sb3(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    env_args = build_env_args(args)

    print(f"\n{'=' * 60}")
    print(f"[Train] Creating {args.n_envs} parallel environments...")
    print(f"[Train] Scenario: {args.scenario}")
    print(f"[Train] Attack type: {args.attack_type}, victim_idx: {args.victim_idx}")
    if args.attack_type == "DYN":
        print(f"[Train] λ = {args.attack_lambda}")
    print(f"{'=' * 60}\n")

    if args.n_envs == 1:
        env = DummyVecEnv([make_env_fn(args, env_args, 0)])
    else:
        try:
            env = SubprocVecEnv([make_env_fn(args, env_args, i) for i in range(args.n_envs)])
        except Exception as e:
            print(f"[Warn] SubprocVecEnv failed ({e}), using DummyVecEnv")
            env = DummyVecEnv([make_env_fn(args, env_args, i) for i in range(args.n_envs)])

    eval_env = DummyVecEnv([make_env_fn(args, env_args, 0)])

    scenario_short = args.scenario.replace("simple_", "").replace("_v2", "").replace("_v3", "")
    exp_name = f"{scenario_short}_{args.attack_type}_victim{args.victim_idx}"
    if args.attack_type == "DYN":
        exp_name += f"_lambda{args.attack_lambda}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    config = vars(args)
    config["timestamp"] = timestamp
    config["env_args"] = env_args
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    if args.use_lstm:
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO(
                "MlpLstmPolicy", env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                verbose=args.verbose,
                tensorboard_log=os.path.join(save_dir, "tb_logs"),
                seed=args.seed,
                device=args.device,
                policy_kwargs=dict(
                    net_arch=dict(pi=[128], vf=[128]),
                    lstm_hidden_size=128,
                ),
            )
            print("[Train] Using RecurrentPPO (LSTM policy)")
        except ImportError:
            print("[Warn] sb3-contrib not installed, falling back to MLP policy")
            print("[Warn] Install with: pip install sb3-contrib")
            args.use_lstm = False

    if not args.use_lstm:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            verbose=args.verbose,
            tensorboard_log=os.path.join(save_dir, "tb_logs"),
            seed=args.seed,
            device=args.device,
            policy_kwargs=policy_kwargs,
        )
        print("[Train] Using PPO (MLP policy)")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.total_timesteps // 20, 1000),
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="attack",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=max(args.total_timesteps // 50, 500),
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"\n[Train] Starting training! Total timesteps: {args.total_timesteps}")
    print(f"[Train] Output directory: {save_dir}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n[Train] Training complete! Model saved to: {final_path}")

    env.close()
    eval_env.close()
    return final_path


def train_with_rllib(args):
    import ray
    from ray import air, tune
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig

    lambda_val = args.attack_lambda if args.attack_type == "DYN" else None
    env_args = build_env_args(args)

    def env_creator(config):
        return make_attacked_env(
            phase1_model_dir=args.phase1_model_dir,
            victim_idx=args.victim_idx,
            num_agents=args.num_agents,
            max_cycles=args.max_cycles,
            attack_lambda=lambda_val,
            phase2_model_dir=args.phase2_model_dir,
            config_path=args.config_path,
            device=args.device,
            env_args=env_args,
        )

    env_id = f"attacked_{args.scenario}_{args.attack_type}"
    register_env(env_id, env_creator)

    ray.init(ignore_reinit_error=True, num_cpus=8)

    config = (
        PPOConfig()
        .environment(env_id, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=4, num_envs_per_worker=2)
        .training(
            model={
                "use_lstm": args.use_lstm,
                "lstm_cell_size": 128,
                "max_seq_len": args.max_cycles,
            },
            lr=args.learning_rate,
            gamma=args.gamma,
            train_batch_size=args.batch_size * 4,
        )
    )

    exp_name = f"attack_{args.scenario}_{args.attack_type}_victim{args.victim_idx}"
    if args.attack_type == "DYN":
        exp_name += f"_lambda{args.attack_lambda}"

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": args.total_timesteps},
            storage_path=os.path.join(args.output_dir, exp_name),
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=20),
        ),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()


def main():
    args = parse_args()

    if not os.path.exists(args.phase1_model_dir):
        raise FileNotFoundError(f"Phase 1 model directory does not exist: {args.phase1_model_dir}")

    if args.attack_type == "DYN" and args.phase2_model_dir is None:
        print("[Warn] DYN mode did not specify phase2_model_dir, defaulting to phase1_model_dir")
        args.phase2_model_dir = args.phase1_model_dir

    if args.config_path is None:
        parent_dir = os.path.dirname(args.phase1_model_dir)
        candidate = os.path.join(parent_dir, "config.json")
        if os.path.exists(candidate):
            args.config_path = candidate
            print(f"[Info] Auto-discovered config: {candidate}")

    print(f"\n[Info] Scenario: {args.scenario}")
    if "simple_tag" in args.scenario:
        print(f"[Info] Environment: Simple Tag ({args.num_agents} adversaries + 1 prey)")
        print(f"[Info] obs_dim=22, action_dim=5, max_cycles={args.max_cycles}")
    else:
        print(f"[Info] Environment: Simple Spread ({args.num_agents} agents)")
        print(f"[Info] obs_dim=36, action_dim=5, max_cycles={args.max_cycles}")

    try:
        train_with_sb3(args)
    except ImportError:
        print("\n[Error] stable-baselines3 not installed.")
        print("[Info] Install with: pip install stable-baselines3")
        print("[Info] Trying Ray RLlib as a fallback...")
        try:
            train_with_rllib(args)
        except ImportError:
            print("\n[Error] Ray RLlib is also not installed.")
            print("Please install at least one training framework:")
            print("  pip install stable-baselines3")
            print("  or pip install 'ray[rllib]'")
            raise


if __name__ == "__main__":
    main()
