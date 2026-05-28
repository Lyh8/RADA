import argparse
import json
import os
import sys
import glob
import numpy as np
import torch
from pathlib import Path


def find_model_dir(base_dir):
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir

    seed_dirs = glob.glob(os.path.join(base_dir, "seed-*"))
    if seed_dirs:
        seed_dirs.sort()
        candidate = seed_dirs[-1]
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
            return full

    return base_dir


def find_models_subdir(model_dir):
    models_path = os.path.join(model_dir, "models")
    if os.path.isdir(models_path):
        pt_files = glob.glob(os.path.join(models_path, "*.pt"))
        if pt_files:
            return models_path

    pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
    if pt_files:
        return model_dir

    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f == "actor_agent0.pt":
                return root

    return os.path.join(model_dir, "models")


def inject_model_dir_into_config(algo_args, models_path):
    possible_sections = ["render", "train", "eval", "algo"]
    for section in possible_sections:
        if section in algo_args and isinstance(algo_args[section], dict):
            algo_args[section]["model_dir"] = models_path

    algo_args["model_dir"] = models_path
    return algo_args


def force_set_runner_model_dir(runner, models_path):
    runner.model_dir = models_path

    if hasattr(runner, 'save_dir'):
        pass

    print(f"[Phase 1.5] Force-set runner.model_dir = {models_path}")


def manual_restore(runner, models_path):
    device = runner.device
    num_agents = runner.num_agents

    print(f"[Phase 1.5] Manually loading model weights from: {models_path}")

    for agent_id in range(num_agents):
        actor_path = os.path.join(models_path, f"actor_agent{agent_id}.pt")
        if os.path.exists(actor_path):
            state_dict = torch.load(actor_path, map_location=device)
            runner.actor[agent_id].actor.load_state_dict(state_dict)
            print(f"  ✓ Loaded actor_agent{agent_id}.pt")
        else:
            print(f"  ✗ Not found {actor_path}")

    critic_path = os.path.join(models_path, "critic.pt")
    if os.path.exists(critic_path):
        state_dict = torch.load(critic_path, map_location=device)
        runner.critic.critic.load_state_dict(state_dict)
        print(f"  ✓ Loaded critic.pt")

    recl_candidates = [
        "recl_net.pt", "recl.pt", "recl_embedding_net.pt",
        "embedding_net.pt", "role_net.pt"
    ]
    recl_loaded = False
    for fname in recl_candidates:
        recl_path = os.path.join(models_path, fname)
        if os.path.exists(recl_path):
            state_dict = torch.load(recl_path, map_location=device)
            recl = getattr(runner, 'recl', None)
            if recl is not None:
                try:
                    recl.embedding_net.load_state_dict(state_dict)
                    print(f"  ✓ Loaded {fname} → recl.embedding_net")
                    recl_loaded = True
                    break
                except RuntimeError:
                    try:
                        recl.load_state_dict(state_dict)
                        print(f"  ✓ Loaded {fname} → recl")
                        recl_loaded = True
                        break
                    except RuntimeError as e:
                        print(f"  ⚠ {fname} failed to load: {e}")

    if not recl_loaded:
        all_pt = glob.glob(os.path.join(models_path, "*.pt"))
        print(f"  ⚠ Failed to load ReCL weights. Available .pt files:")
        for f in all_pt:
            print(f"    - {os.path.basename(f)}")
        print(f"  Please confirm the ReCL model filename and manually modify this script.")


def collect_embeddings(runner, num_episodes=100):
    from harl.utils.trans_tools import _t2n

    recl = getattr(runner, 'recl', None)
    if recl is None:
        raise ValueError(
            "Runner has no ReCL module!\n"
            "Please confirm Phase 1 training used --use_recl True\n"
            "and that the HARL version supports ReCL."
        )

    envs = getattr(runner, 'eval_envs', None) or runner.envs
    actors = runner.actor
    num_agents = runner.num_agents
    device = runner.device

    obs_space = envs.observation_space[0]
    obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n

    recurrent_n = getattr(runner, 'recurrent_n', 1)
    rnn_hidden_size = getattr(runner, 'rnn_hidden_size', 64)

    if hasattr(envs, 'num_envs'):
        n_threads = envs.num_envs
    elif hasattr(runner, 'algo_args'):
        n_threads = runner.algo_args.get("eval", {}).get(
            "n_eval_rollout_threads",
            runner.algo_args.get("train", {}).get("n_rollout_threads", 1)
        )
    else:
        n_threads = 1

    all_embeddings = []
    episode_count = 0

    print(f"\n[Phase 1.5] Starting to collect role embeddings...")
    print(f"  Env threads: {n_threads}")
    print(f"  Number of agents: {num_agents}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Target episodes: {num_episodes}")

    while episode_count < num_episodes:
        obs, share_obs, available_actions = envs.reset()

        rnn_states = np.zeros(
            (n_threads, num_agents, recurrent_n, rnn_hidden_size),
            dtype=np.float32
        )
        masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

        if hasattr(recl, 'embedding_net') and hasattr(recl.embedding_net, 'agent_embedding_net'):
            if hasattr(recl.embedding_net.agent_embedding_net, 'rnn_hidden'):
                recl.embedding_net.agent_embedding_net.rnn_hidden = None

        done_flags = np.zeros(n_threads, dtype=bool)

        while not done_flags.all():
            actions_list = []
            for agent_id in range(num_agents):
                obs_input = obs[:, agent_id]
                rnn_input = rnn_states[:, agent_id]
                mask_input = masks[:, agent_id]
                avail = (
                    available_actions[:, agent_id]
                    if available_actions is not None and available_actions[0] is not None
                    else None
                )

                action, rnn_state = actors[agent_id].act(
                    obs_input, rnn_input, mask_input, avail, deterministic=True
                )
                rnn_states[:, agent_id] = _t2n(rnn_state)
                actions_list.append(_t2n(action))

            actions = np.array(actions_list).transpose(1, 0, 2)

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                flat_obs = obs_tensor.reshape(-1, obs_dim)
                agent_embed = recl.embedding_net.agent_embed_forward(flat_obs, detach=True)
                role_embed = recl.embedding_net.role_embed_forward(agent_embed, detach=True, ema=False)
                role_embed = role_embed.reshape(n_threads, num_agents, -1)

            for t in range(n_threads):
                if not done_flags[t]:
                    all_embeddings.append(role_embed[t].cpu().numpy())

            obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)

            done_env = np.all(dones, axis=1)
            rnn_states[done_env] = 0.0
            masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
            masks[done_env] = 0.0

            new_done = done_env & ~done_flags
            episode_count += new_done.sum()
            done_flags = done_flags | done_env

            if episode_count >= num_episodes:
                break

        print(f"  Progress: {min(episode_count, num_episodes)}/{num_episodes} episodes, "
              f"{len(all_embeddings)} timestep samples")

    embeddings = np.array(all_embeddings)
    print(f"[Phase 1.5] Collection complete! Embedding shape: {embeddings.shape}")
    return embeddings


def generate_clusters(embeddings, num_clusters, threshold_multiplier=1.2):
    from sklearn.cluster import KMeans

    flat = embeddings.reshape(-1, embeddings.shape[-1])

    print(f"\n[Phase 1.5] Starting KMeans clustering...")
    print(f"  Number of samples: {flat.shape[0]}")
    print(f"  Embedding dim: {flat.shape[1]}")
    print(f"  Number of clusters: {num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat)
    centers = kmeans.cluster_centers_

    thresholds = np.zeros(num_clusters)
    cluster_info = {}

    for k in range(num_clusters):
        mask = labels == k
        cluster_points = flat[mask]
        if len(cluster_points) > 0:
            dists = np.linalg.norm(cluster_points - centers[k], axis=1)
            thresholds[k] = dists.max() * threshold_multiplier
            cluster_info[f"cluster_{k}"] = {
                "size": int(mask.sum()),
                "mean_dist": float(dists.mean()),
                "max_dist": float(dists.max()),
                "threshold": float(thresholds[k]),
            }
            print(f"  Cluster {k}: size={mask.sum()}, "
                  f"mean_dist={dists.mean():.4f}, "
                  f"max_dist={dists.max():.4f}, "
                  f"threshold={thresholds[k]:.4f}")

    return centers, thresholds, cluster_info


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.5: Collect role embeddings and generate cluster centers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Phase 1 trained model directory (the directory containing config.json)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="How many episodes of embeddings to collect",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=2,
        help="Number of KMeans clusters",
    )
    parser.add_argument(
        "--threshold_multiplier", type=float, default=1.2,
        help="Deviation threshold = max_dist × multiplier",
    )
    args = parser.parse_args()

    model_dir = find_model_dir(args.model_dir)
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"[ERROR] config.json not found: {config_path}")
        print(f"[Hint] Please confirm the path. The HARL directory structure is usually:")
        print(f"  results/<env>/<scenario>/<algo>/<exp_name>/seed-xxxxx/config.json")
        sys.exit(1)

    models_path = find_models_subdir(model_dir)
    if not os.path.exists(os.path.join(models_path, "actor_agent0.pt")):
        print(f"[ERROR] Model weight file not found: {models_path}/actor_agent0.pt")
        print(f"[Hint] File list under model_dir:")
        for item in sorted(os.listdir(model_dir)):
            full = os.path.join(model_dir, item)
            if os.path.isdir(full):
                sub_items = os.listdir(full)
                print(f"  {item}/ ({len(sub_items)} files)")
                for si in sorted(sub_items)[:10]:
                    print(f"    {si}")
            else:
                print(f"  {item}")
        sys.exit(1)

    print("=" * 60)
    print("Phase 1.5: Collect role embeddings + generate cluster centers")
    print("=" * 60)
    print(f"Model dir:   {model_dir}")
    print(f"Weights dir: {models_path}")
    print(f"Episodes:   {args.num_episodes}")
    print(f"Clusters:   {args.num_clusters}")
    print("=" * 60)

    print(f"\nModel file list:")
    for f in sorted(os.listdir(models_path)):
        fpath = os.path.join(models_path, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size:.1f} KB)")

    with open(config_path, encoding="utf-8") as f:
        all_config = json.load(f)

    main_args = all_config["main_args"]
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]

    print(f"\nConfig info:")
    print(f"  Algorithm: {main_args['algo']}")
    print(f"  Environment: {main_args['env']}")
    print(f"  algo_args subkeys: {list(algo_args.keys())}")

    use_recl = False
    for section_name, section in algo_args.items():
        if isinstance(section, dict) and section.get("use_recl"):
            use_recl = True
            print(f"  use_recl: True (in algo_args['{section_name}'])")
            break
    if not use_recl:
        if algo_args.get("use_recl"):
            use_recl = True

    if not use_recl:
        print("[WARNING] use_recl=True not found in config, but still trying to continue...")
        print("[Hint] If the Runner has no ReCL module, the script will error when collecting embeddings.")

    inject_model_dir_into_config(algo_args, models_path)
    print(f"\nInjected model_dir={models_path} into algo_args")

    print(f"\nCreating Runner...")
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)

    print(f"\nLoading model weights...")

    try:
        force_set_runner_model_dir(runner, models_path)
        runner.restore()
        print("[Phase 1.5] ✓ Successfully loaded model via runner.restore()")
    except Exception as e:
        print(f"[Phase 1.5] ⚠ runner.restore() failed: {e}")
        print("[Phase 1.5] Trying to manually load model weights...")

        try:
            manual_restore(runner, models_path)
            print("[Phase 1.5] ✓ Manual model loading complete")
        except Exception as e2:
            print(f"[Phase 1.5] ✗ Manual loading also failed: {e2}")
            print("\nPlease send me the following info for further debugging:")
            print(f"1. runner attributes: {[a for a in dir(runner) if not a.startswith('_')]}")
            if hasattr(runner, 'actor') and len(runner.actor) > 0:
                print(f"2. actor[0] attributes: {[a for a in dir(runner.actor[0]) if not a.startswith('_')]}")
            if hasattr(runner, 'recl') and runner.recl is not None:
                print(f"3. recl attributes: {[a for a in dir(runner.recl) if not a.startswith('_')]}")
            runner.close()
            sys.exit(1)

    embeddings = collect_embeddings(runner, num_episodes=args.num_episodes)

    emb_path = os.path.join(model_dir, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"[Phase 1.5] Saved embeddings to: {emb_path}")

    centers, thresholds, cluster_info = generate_clusters(
        embeddings,
        args.num_clusters,
        args.threshold_multiplier
    )

    centers_path = os.path.join(model_dir, "cluster_centers.npy")
    thresholds_path = os.path.join(model_dir, "cluster_thresholds.npy")
    info_path = os.path.join(model_dir, "cluster_info.json")

    np.save(centers_path, centers)
    np.save(thresholds_path, thresholds)

    cluster_info["meta"] = {
        "num_episodes": args.num_episodes,
        "num_clusters": args.num_clusters,
        "threshold_multiplier": args.threshold_multiplier,
        "embeddings_shape": list(embeddings.shape),
        "model_dir": model_dir,
    }
    with open(info_path, "w") as f:
        json.dump(cluster_info, f, indent=2)

    print(f"\n[Phase 1.5] Saved cluster centers to: {centers_path}")
    print(f"[Phase 1.5] Saved deviation thresholds to: {thresholds_path}")
    print(f"[Phase 1.5] Saved cluster info to: {info_path}")

    runner.close()

    print("\n" + "=" * 60)
    print("Phase 1.5 complete!")
    print("=" * 60)
    print(f"  embeddings.npy:        {embeddings.shape}")
    print(f"  cluster_centers.npy:   {centers.shape}")
    print(f"  cluster_thresholds:    {thresholds}")
    print(f"\nNext step: Run Phase 2 to train Tracker")
    print(f"  python train.py --algo mappo --env pettingzoo_mpe --exp_name phase2_tracker \\")
    print(f"      --use_recl True --use_tracker True --tracker_train True \\")
    print(f"      --tracker_model_dir \"{model_dir}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
