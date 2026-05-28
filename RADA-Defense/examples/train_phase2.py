import argparse
import json
import os
import sys
import torch
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


VALID_ABLATION_VARIANTS = {"full", "no_role", "no_tf", "no_role_input"}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--env", type=str, default="pettingzoo_mpe")
    parser.add_argument("--exp_name", type=str, default="phase2_tracker")
    parser.add_argument("--load_config", type=str, default="")

    parser.add_argument(
        "--phase1_model_dir",
        type=str,
        required=True,
        help="Path to Phase 1 model directory (contains actor_agent*.pt, critic.pt, etc.)"
    )

    parser.add_argument(
        "--ablation_variant",
        type=str,
        default="full",
        choices=sorted(VALID_ABLATION_VARIANTS),
        help="Ablation variant: full(V0), no_role(V1), no_tf(V2), no_role_input(V3)"
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)

    if args["load_config"] != "":
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    update_args(unparsed_dict, algo_args, env_args)

    phase1_dir = args["phase1_model_dir"]
    if not os.path.exists(phase1_dir):
        print(f"ERROR: Phase 1 model dir does not exist: {phase1_dir}")
        sys.exit(1)

    has_actor = any("actor" in f for f in os.listdir(phase1_dir))
    if not has_actor:
        print(f"ERROR: No actor files found in {phase1_dir}")
        print(f"  Contents: {os.listdir(phase1_dir)}")
        sys.exit(1)

    algo_args["train"]["model_dir"] = phase1_dir
    print(f"[Phase 2] model_dir set to: {phase1_dir}")

    algo_args["algo"]["use_tracker"] = True
    algo_args["algo"]["tracker_train"] = True

    ablation_variant = args["ablation_variant"]
    algo_args["algo"]["ablation_variant"] = ablation_variant

    if args["exp_name"] == "phase2_tracker":
        args["exp_name"] = f"phase2_{ablation_variant}"

    if "tracker_model_dir" not in algo_args["algo"] or algo_args["algo"]["tracker_model_dir"] is None:
        algo_args["algo"]["tracker_model_dir"] = phase1_dir

    from harl.runners.on_policy_tracker_runner import OnPolicyMARunnerWithTracker

    print("=" * 60)
    print("Phase 2: Tracker Training")
    print("=" * 60)
    print(f"  algo:              {args['algo']}")
    print(f"  env:               {args['env']}")
    print(f"  exp_name:          {args['exp_name']}")
    print(f"  phase1_model_dir:  {phase1_dir}")
    print(f"  model_dir (train): {algo_args['train']['model_dir']}")
    print(f"  use_tracker:       {algo_args['algo'].get('use_tracker')}")
    print(f"  tracker_train:     {algo_args['algo'].get('tracker_train')}")
    print(f"  ablation_variant:  {ablation_variant}")
    print(f"    → use_role_input:  {ablation_variant not in {'no_role', 'no_role_input'}}")
    print(f"    → use_tf:          {ablation_variant != 'no_tf'}")
    print(f"    → use_stage1:      {ablation_variant != 'no_role'}")
    print("=" * 60)

    runner = OnPolicyMARunnerWithTracker(args, algo_args, env_args)

    if hasattr(runner, 'tracker'):
        runner.tracker.tracker_net._init_weights()
        from torch.optim import Adam
        runner.tracker.optimizer = Adam(
            runner.tracker.tracker_net.parameters(),
            lr=algo_args["algo"].get("tracker_lr", 5e-4),
            eps=1e-5,
        )
        print(f"[Phase 2] Tracker network force re-initialized (discarding any old weights)")

        net = runner.tracker.tracker_net
        print(f"  fc_mean.bias:    {net.fc_mean.bias.data.tolist()}")
        print(f"  fc_log_std.bias: {net.fc_log_std.bias.data.tolist()}")
        print(f"  fc_log_std.weight std: {net.fc_log_std.weight.data.std():.4f} (should ≈ 0.01)")

    frozen_count = 0

    for agent_id in range(runner.num_agents):
        for param in runner.actor[agent_id].actor.parameters():
            param.requires_grad = False
            frozen_count += 1

    for param in runner.critic.critic.parameters():
        param.requires_grad = False
        frozen_count += 1

    if hasattr(runner, 'recl') and runner.recl is not None:
        if hasattr(runner.recl, 'embedding_net'):
            for param in runner.recl.embedding_net.parameters():
                param.requires_grad = False
                frozen_count += 1
        if hasattr(runner.recl, 'cl_net'):
            for param in runner.recl.cl_net.parameters():
                param.requires_grad = False
                frozen_count += 1

    tracker_trainable = sum(
        p.numel() for p in runner.tracker.tracker_net.parameters() if p.requires_grad
    )
    print(f"[Phase 2] Froze {frozen_count} MAPPO/ReCL parameter groups")
    print(f"[Phase 2] Tracker trainable parameters: {tracker_trainable}")

    runner.run()
    runner.close()

    print("\n" + "=" * 60)
    print(f"Phase 2 Training Complete! (ablation={ablation_variant})")
    print("=" * 60)


if __name__ == "__main__":
    main()
