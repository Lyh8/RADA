#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
import yaml
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Part 2 Defense Evaluation Experiment")

    parser.add_argument("--load_config", type=str, required=True,
                        help="Path to the config.json from Phase 2 training")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Model directory from Phase 2 training")

    parser.add_argument("--attack_type", type=str, default="ACT",
                        choices=["ACT", "DYN", "random", "none"])
    parser.add_argument("--attack_model", type=str, default=None)

    parser.add_argument("--act_model", type=str, default=None)
    parser.add_argument("--dyn001_model", type=str, default=None)
    parser.add_argument("--dyn002_model", type=str, default=None)

    parser.add_argument("--victim_id", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--detection_threshold", type=float, default=-11.0)
    parser.add_argument("--methods", type=str, nargs="+", default=None)

    parser.add_argument("--experiment", type=str, default="main",
                        choices=["calibrate", "main", "delay", "all", "trajectory"])

    parser.add_argument("--w_same", type=float, default=3.0,
                        help="Same-role weight for C4 (default: 3.0)")
    parser.add_argument("--w_diff", type=float, default=0.0,
                        help="Different-role weight for C4 (default: 0.0)")

    parser.add_argument("--calibrate_episodes", type=int, default=50)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument("--ema_window", type=int, default=5)

    parser.add_argument("--env", type=str, default="pettingzoo_mpe")
    parser.add_argument("--scenario", type=str, default="simple_tag_v2-continuous")
    parser.add_argument("--n_threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--pos_indices", type=int, nargs=2, default=[2, 3],
                        help="Obs indices for (x,y) position (default: 2 3)")
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Episodes per method to pick median from")
    parser.add_argument("--traj_methods", type=str, nargs="+",
                        default=["B0", "B1", "C4"],
                        help="Methods to compare in trajectory plot")
    parser.add_argument("--prey_ids", type=int, nargs="+", default=None,
                        help="Agent indices that are prey (for coloring)")

    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()



def _create_runner(args):
    from harl.runners.on_policy_tracker_runner import OnPolicyMARunnerWithTracker
    from harl.algorithms.trackers.defense_eval_runner import patch_runner_with_defense

    config_path = args.load_config
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        if config_path.endswith(".json"):
            saved_config = json.load(f)
        else:
            saved_config = yaml.safe_load(f)
    print(f"[Config] Loaded from {config_path}")

    algo_args = _build_algo_args(args, saved_config)
    env_args = _build_env_args(args, saved_config)
    main_args = _build_main_args(args, saved_config)

    print(f"\n[Init] Creating runner...")
    print(f"  model_dir = {args.model_dir}")
    runner = OnPolicyMARunnerWithTracker(main_args, algo_args, env_args)
    patch_runner_with_defense(runner)

    return runner


def _create_attacker(attack_type, model_path, runner):
    from harl.algorithms.attackers.attacker import HARLAttacker

    act_space = runner.eval_envs.action_space[0]
    action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 5
    action_low = float(getattr(act_space, 'low', np.zeros(1))[0])
    action_high = float(getattr(act_space, 'high', np.ones(1))[0])

    return HARLAttacker(
        attack_type=attack_type.lower(),
        model_path=model_path,
        action_space_low=action_low,
        action_space_high=action_high,
        action_dim=action_dim,
    )



def _run_calibrate(runner, args, output_dir):
    runner.log_dir = output_dir
    threshold = runner.calibrate_threshold(
        n_episodes=args.calibrate_episodes,
        ema_window=args.ema_window,
        target_fpr=args.target_fpr,
    )
    print(f"\n[Calibrate] Calibration complete: η* = {threshold:.4f}")
    print(f"[Calibrate] Suggested for subsequent experiments: --detection_threshold {threshold:.4f}")
    return threshold


def _run_main(runner, args, output_dir):
    if args.attack_model is None and args.attack_type not in ("none", "random"):
        print("[Error] --attack_model is required for main experiment")
        sys.exit(1)

    attacker = _create_attacker(args.attack_type, args.attack_model, runner)
    runner.log_dir = output_dir

    methods = args.methods
    if methods is None:
        from harl.algorithms.trackers.defense import ALL_METHODS
        methods = ALL_METHODS

    results = runner.run_main_comparison(
        attacker=attacker,
        attack_name=args.attack_type,
        methods=methods,
        n_episodes=args.n_episodes,
        victim_id=args.victim_id,
        detection_threshold=args.detection_threshold,
        w_same=args.w_same,
        w_diff=args.w_diff,
    )

    return results


def _run_delay(runner, args, output_dir):
    if args.attack_model is None and args.attack_type not in ("none", "random"):
        print("[Error] --attack_model is required for delay experiment")
        sys.exit(1)

    attacker = _create_attacker(args.attack_type, args.attack_model, runner)
    runner.log_dir = output_dir

    results = runner.run_delay_experiment(
        attacker=attacker,
        attack_name=args.attack_type,
        delays=[0, 2, 5, 10, 15],
        n_episodes=args.n_episodes,
        victim_id=args.victim_id,
        detection_threshold=args.detection_threshold,
    )
    return results


def _run_trajectory(runner, args, output_dir):
    if args.attack_model is None and args.attack_type not in ("none", "random"):
        print("[Error] --attack_model is required for trajectory experiment")
        sys.exit(1)

    attacker = _create_attacker(args.attack_type, args.attack_model, runner)
    runner.log_dir = output_dir

    trajectories = runner.run_trajectory_experiment(
        attacker=attacker,
        attack_name=args.attack_type,
        victim_id=args.victim_id,
        detection_threshold=args.detection_threshold,
        methods=args.traj_methods,
        pos_indices=tuple(args.pos_indices),
        n_candidates=args.n_candidates,
        output_dir=output_dir,
        w_same=args.w_same,
        w_diff=args.w_diff,
        prey_agent_ids=args.prey_ids,
    )
    return trajectories


def _run_all(runner, args, output_dir):
    from harl.algorithms.trackers.defense import ALL_METHODS

    attacks = []
    if args.act_model:
        attacks.append(("ACT", "ACT", args.act_model))
    if args.dyn001_model:
        attacks.append(("DYN", "DYN_001", args.dyn001_model))
    if args.dyn002_model:
        attacks.append(("DYN", "DYN_002", args.dyn002_model))

    if not attacks:
        print("[Error] --experiment all requires at least one attack model path")
        return {}

    runner.log_dir = output_dir

    all_results = {}
    for atk_type, atk_name, atk_path in attacks:
        print(f"\n\n{'#' * 70}")
        print(f"# Attack: {atk_name}")
        print(f"{'#' * 70}")

        attacker = _create_attacker(atk_type, atk_path, runner)

        results = runner.run_main_comparison(
            attacker=attacker,
            attack_name=atk_name,
            methods=ALL_METHODS,
            n_episodes=args.n_episodes,
            victim_id=args.victim_id,
            detection_threshold=args.detection_threshold,
            w_same=args.w_same,
            w_diff=args.w_diff,
        )
        all_results[atk_name] = results

    path = os.path.join(output_dir, "defense_all_attacks.json")
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Result] Combined results → {path}")

    return all_results



def _build_algo_args(args, saved_config):
    algo_args = {}

    if "algo_args" in saved_config:
        algo_args = dict(saved_config["algo_args"])
    elif "algo" in saved_config:
        algo_args = dict(saved_config)
    else:
        algo_args = dict(saved_config)

    algo = algo_args.setdefault("algo", {})
    model = algo_args.setdefault("model", {})
    train = algo_args.setdefault("train", {})
    eval_cfg = algo_args.setdefault("eval", {})
    render = algo_args.setdefault("render", {})

    algo["use_tracker"] = True
    algo["tracker_train"] = False
    algo["tracker_model_dir"] = args.model_dir
    algo.setdefault("use_recl", True)

    eval_cfg["n_eval_rollout_threads"] = args.n_threads
    eval_cfg["eval_episodes"] = args.n_episodes
    eval_cfg["use_eval"] = True

    train["model_dir"] = args.model_dir

    render["use_render"] = False

    return algo_args


def _build_env_args(args, saved_config):
    if "env_args" in saved_config:
        env_args = dict(saved_config["env_args"])
    elif "algo_args" in saved_config and "env_args" in saved_config.get("algo_args", {}):
        env_args = dict(saved_config["algo_args"]["env_args"])
    else:
        env_args = {}

    env_args.setdefault("scenario", args.scenario)
    env_args.setdefault("continuous_actions", True)

    return env_args


def _build_main_args(args, saved_config):
    if "main_args" in saved_config:
        main = dict(saved_config["main_args"])
        main["algo"] = main.get("algo", "mappo")
        main["env"] = main.get("env", args.env)
        return main

    main = {
        "algo": "mappo",
        "env": args.env,
        "exp_name": "defense_eval",
        "seed_specify": True,
        "seed": args.seed,
        "cuda": True,
        "cuda_deterministic": True,
        "torch_threads": 4,
    }

    if "seed" in saved_config:
        seed_cfg = saved_config["seed"]
        if isinstance(seed_cfg, dict):
            main.update(seed_cfg)
    if "device" in saved_config:
        dev_cfg = saved_config["device"]
        if isinstance(dev_cfg, dict):
            main.update(dev_cfg)

    return main



def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Part 2 Defense Experiment")
    print("=" * 70)
    print(f"  Config:       {args.load_config}")
    print(f"  Model dir:    {args.model_dir}")
    print(f"  Experiment:   {args.experiment}")
    print(f"  Attack:       {args.attack_type}")
    print(f"  Victim:       {args.victim_id}")
    print(f"  Episodes:     {args.n_episodes}")
    print(f"  Threshold:    {args.detection_threshold}")
    print("=" * 70 + "\n")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.model_dir), "defense_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Output] Results will be saved to: {output_dir}\n")

    runner = _create_runner(args)

    if args.experiment == "calibrate":
        _run_calibrate(runner, args, output_dir)
    elif args.experiment == "main":
        _run_main(runner, args, output_dir)
    elif args.experiment == "delay":
        _run_delay(runner, args, output_dir)
    elif args.experiment == "all":
        _run_all(runner, args, output_dir)
    elif args.experiment == "trajectory":
        _run_trajectory(runner, args, output_dir)

    print("\n[Done] All experiments completed.")


if __name__ == "__main__":
    main()
