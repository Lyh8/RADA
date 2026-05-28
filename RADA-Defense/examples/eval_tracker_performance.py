#!/usr/bin/env python3
import numpy as np
import torch
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("[Error] matplotlib not found. Install: pip install matplotlib")
    sys.exit(1)

try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    print("[Error] scikit-learn not found. Install: pip install scikit-learn")
    sys.exit(1)



def _get_env_dims(scenario):
    if "simple_tag" in scenario:
        return 22, 5
    else:
        return 36, 5


def _build_env_args(scenario, num_agents):
    if "simple_tag" in scenario:
        return {
            "scenario": scenario,
            "num_good": 1,
            "num_adversaries": num_agents,
            "num_obstacles": 2,
        }
    else:
        return {
            "scenario": scenario,
        }


def calibrate_tracker_clusters(
    tracker, actor, recl,
    env_args, num_agents, obs_dim, max_cycles,
    num_calib_episodes=20, num_clusters=2,
    threshold_multiplier=1.2, device="cpu",
):
    from sklearn.cluster import KMeans
    from harl.utils.attacked_env_harl import create_pettingzoo_env

    print(f"  [calibrate] Collecting embeddings from {num_calib_episodes} episodes...")

    env = create_pettingzoo_env(env_args=env_args, N=num_agents, max_cycles=max_cycles)
    agent_names = env.possible_agents

    rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)

    all_embeddings = []

    for ep in range(num_calib_episodes):
        obs_dict = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        recl.embedding_net.agent_embedding_net.rnn_hidden = None

        for t in range(max_cycles):
            obs_array = np.stack(
                [obs_dict[a] for a in agent_names]
            ).reshape(1, num_agents, -1).astype(np.float32)
            obs_t = torch.tensor(obs_array, device=device)

            with torch.no_grad():
                ae = recl.embedding_net.agent_embed_forward(
                    obs_t.reshape(-1, obs_dim), detach=True)
                re = recl.embedding_net.role_embed_forward(
                    ae, detach=True, ema=False)
                re = re.reshape(1, num_agents, -1)

            all_embeddings.append(re[0].cpu().numpy())

            actions = {}
            for an in agent_names:
                obs = obs_dict[an].reshape(1, -1).astype(np.float32)
                with torch.no_grad():
                    act, _ = actor.act(obs, rnn_states, masks, None, deterministic=True)
                actions[an] = np.clip(act.cpu().numpy().flatten(), 0, 1)

            step_result = env.step(actions)
            obs_dict = step_result[0]

            if len(step_result) == 5:
                terms, truncs = step_result[2], step_result[3]
            else:
                terms = step_result[2]
                truncs = {a: False for a in terms}
            if any(terms.values()) or any(truncs.values()):
                break

    env.close()

    embeddings = np.array(all_embeddings)
    flat = embeddings.reshape(-1, embeddings.shape[-1])

    print(f"  [calibrate] KMeans: {flat.shape[0]} samples, dim={flat.shape[1]}, k={num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat)
    centers = kmeans.cluster_centers_

    thresholds = np.zeros(num_clusters)
    for k in range(num_clusters):
        mask = labels == k
        dists = np.linalg.norm(flat[mask] - centers[k], axis=1)
        thresholds[k] = dists.max() * threshold_multiplier
        print(f"  [calibrate] Cluster {k}: n={mask.sum()}, "
              f"max_dist={dists.max():.2f}, threshold={thresholds[k]:.2f}")

    tracker.load_cluster_info(centers, thresholds)
    print(f"  [calibrate] Tracker cluster updated!")

    return centers, thresholds


def collect_episode_scores(
    phase1_model_dir,
    phase2_model_dir,
    attack_type,
    attack_model_path=None,
    victim_idx=0,
    num_agents=6,
    num_episodes=50,
    max_cycles=25,
    budget=0.35,
    config_path=None,
    device="cpu",
    scenario="simple_spread_v2",
    auto_calibrate=True,
    num_calib_episodes=20,
    _cached_calibration=None,
):
    from harl.utils.attacked_env_harl import (
        create_pettingzoo_env, load_harl_actor, load_recl, load_tracker
    )
    from harl.algorithms.attackers import HARLAttacker

    obs_dim, action_dim = _get_env_dims(scenario)
    role_dim = 32
    env_args = _build_env_args(scenario, num_agents)

    print(f"  [collect] scenario={scenario}, obs_dim={obs_dim}, action_dim={action_dim}")

    actor = load_harl_actor(
        phase1_model_dir, config_path=config_path,
        obs_dim=obs_dim, action_dim=action_dim, device=device)

    recl = load_recl(
        phase1_model_dir, obs_dim=obs_dim, num_agents=num_agents,
        device=device, role_embedding_dim=role_dim)

    tracker_dir = os.path.join(phase2_model_dir, "tracker") if phase2_model_dir else None
    if tracker_dir and os.path.exists(tracker_dir):
        tracker = load_tracker(
            tracker_dir, num_agents=num_agents, obs_dim=obs_dim,
            action_dim=action_dim, role_embedding_dim=role_dim, device=device)
    else:
        raise ValueError(f"Tracker not found at: {tracker_dir}")
    if auto_calibrate and _cached_calibration is not None:
        tracker.load_cluster_info(_cached_calibration[0], _cached_calibration[1])
        print(f"  [collect] Reusing existing cluster calibration")
    elif auto_calibrate:
        calibrate_tracker_clusters(
            tracker, actor, recl,
            env_args=env_args, num_agents=num_agents,
            obs_dim=obs_dim, max_cycles=max_cycles,
            num_calib_episodes=num_calib_episodes,
            device=device,
        )
    if attack_type not in ["none", "random"]:
        attacker = HARLAttacker(
            attack_type=attack_type,
            model_path=attack_model_path,
            budget=budget, action_dim=action_dim)
    elif attack_type == "random":
        attacker = HARLAttacker(
            attack_type="random", model_path=None,
            budget=budget, action_dim=action_dim)
    else:
        attacker = None

    env = create_pettingzoo_env(env_args=env_args, N=num_agents, max_cycles=max_cycles)
    agent_names = env.possible_agents

    rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)

    all_scores = []
    all_rewards = []
    all_lengths = []

    for ep in range(num_episodes):
        reset_result = env.reset()
        obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        tracker.init_hidden(1)
        tracker.reset_detection_stats()
        recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((1, num_agents, action_dim))
        ep_scores = []
        ep_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            all_actions = []
            team_action = {}

            for agent_num, agent_name in enumerate(agent_names):
                agent_obs = obs_dict[agent_name].reshape(1, -1).astype(np.float32)
                with torch.no_grad():
                    action_t, _ = actor.act(
                        agent_obs, rnn_states, masks, None, deterministic=True)
                normal_action = action_t.cpu().numpy().flatten()

                if agent_num == victim_idx and attacker is not None:
                    agent_action = attacker.get_attack_action(
                        agent_obs.flatten(), normal_action, deterministic=True)
                else:
                    agent_action = normal_action

                agent_action = np.clip(agent_action, 0.0, 1.0)
                team_action[agent_name] = agent_action
                all_actions.append(agent_action)

            obs_array = np.stack(
                [obs_dict[a] for a in agent_names]
            ).reshape(1, num_agents, -1).astype(np.float32)
            act_array = np.stack(all_actions).reshape(1, num_agents, -1)

            obs_t = torch.tensor(obs_array, device=device)
            last_act_t = torch.tensor(last_actions, dtype=torch.float32, device=device)

            with torch.no_grad():
                ae = recl.embedding_net.agent_embed_forward(
                    obs_t.reshape(-1, obs_dim), detach=True)
                re = recl.embedding_net.role_embed_forward(
                    ae, detach=True, ema=False)
                re = re.reshape(1, num_agents, -1)

            mu, std, _ = tracker.forward(obs_t, last_act_t, re)
            act_t = torch.tensor(act_array, dtype=torch.float32, device=device)
            scores = tracker.compute_scores(mu, std, act_t, use_grouping=True)

            ep_scores.append(scores[0].copy())
            last_actions = act_array

            step_result = env.step(team_action)
            if len(step_result) == 5:
                obs_dict, rewards, terms, truncs, infos = step_result
            else:
                obs_dict, rewards, dones, infos = step_result
                terms = dones
                truncs = {a: False for a in dones}

            for a in agent_names:
                ep_reward += rewards[a]

            terminated = any(terms.values())
            truncated = any(truncs.values())

        all_scores.append(np.stack(ep_scores))
        all_rewards.append(ep_reward)
        all_lengths.append(len(ep_scores))

        if (ep + 1) % 10 == 0:
            print(f"  [{attack_type:>6}] Episode {ep+1:>3}/{num_episodes}: "
                  f"reward={ep_reward:.1f}")

    env.close()

    print(f"  [{attack_type:>6}] Done. Avg reward={np.mean(all_rewards):.2f}")

    return {
        'scores': all_scores,
        'rewards': all_rewards,
        'lengths': all_lengths,
        'attack_type': attack_type,
        'victim_idx': victim_idx,
        '_calibration': (tracker.cluster_centers, tracker.cluster_thresholds),
    }



def smooth_scores_ema(raw_scores, window):
    T = raw_scores.shape[0]
    smoothed = np.zeros_like(raw_scores)

    for t in range(T):
        if window == -1:
            alpha = 1.0 / (t + 1)
        else:
            alpha = min(1.0, 1.0 / window)

        if t == 0:
            smoothed[t] = raw_scores[t]
        else:
            smoothed[t] = (1 - alpha) * smoothed[t - 1] + alpha * raw_scores[t]

    return smoothed


def get_episode_level_scores(episode_raw_scores, victim_idx, window, method="final"):
    smoothed = smooth_scores_ema(episode_raw_scores, window)
    N = smoothed.shape[1]

    victim_list = []
    normal_list = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            if method == "final":
                score = smoothed[-1, i, j]
            elif method == "mean":
                score = np.nanmean(smoothed[:, i, j])
            elif method == "min":
                score = np.nanmin(smoothed[:, i, j])
            else:
                score = smoothed[-1, i, j]

            if np.isnan(score):
                continue

            if j == victim_idx:
                victim_list.append(score)
            else:
                normal_list.append(score)

    return victim_list, normal_list



def compute_roc_curve(attack_data, normal_data, victim_idx, window, method="final"):
    positive_scores = []
    for ep_scores in attack_data['scores']:
        vs, _ = get_episode_level_scores(ep_scores, victim_idx, window, method)
        positive_scores.extend(vs)

    negative_scores = []
    for ep_scores in normal_data['scores']:
        vs, ns = get_episode_level_scores(ep_scores, victim_idx, window, method)
        negative_scores.extend(vs)
        negative_scores.extend(ns)

    positive_scores = [s for s in positive_scores if not np.isnan(s)]
    negative_scores = [s for s in negative_scores if not np.isnan(s)]

    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
    all_scores = np.array(positive_scores + negative_scores)

    fpr, tpr, thresholds = roc_curve(labels, -all_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, thresholds, auc_score


def compute_step_level_roc(attack_data, normal_data, victim_idx, window):
    positive_scores = []
    negative_scores = []
    N = attack_data['scores'][0].shape[1]

    for ep_scores in attack_data['scores']:
        smoothed = smooth_scores_ema(ep_scores, window)
        for t in range(smoothed.shape[0]):
            for i in range(N):
                if i == victim_idx:
                    continue
                score = smoothed[t, i, victim_idx]
                if not np.isnan(score):
                    positive_scores.append(score)

    for ep_scores in normal_data['scores']:
        smoothed = smooth_scores_ema(ep_scores, window)
        for t in range(smoothed.shape[0]):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    score = smoothed[t, i, j]
                    if not np.isnan(score):
                        negative_scores.append(score)

    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
    all_scores = np.array(positive_scores + negative_scores)

    fpr, tpr, thresholds = roc_curve(labels, -all_scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, thresholds, auc_score


def compute_detection_time_vs_fpr(attack_data, normal_data, victim_idx,
                                   window, thresholds):
    N = attack_data['scores'][0].shape[1]
    n_th = len(thresholds)

    fpr_arr = np.zeros(n_th)
    avg_time_arr = np.full(n_th, np.inf)
    detect_rate_arr = np.zeros(n_th)

    for th_idx, threshold in enumerate(thresholds):
        total_pairs = 0
        false_pos = 0
        for ep_scores in normal_data['scores']:
            smoothed = smooth_scores_ema(ep_scores, window)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(np.isnan(smoothed[:, i, j])):
                        continue
                    total_pairs += 1
                    if np.nanmin(smoothed[:, i, j]) < threshold:
                        false_pos += 1
        fpr_arr[th_idx] = false_pos / max(total_pairs, 1)

        detect_times = []
        total_attack = 0
        detected = 0
        for ep_scores in attack_data['scores']:
            smoothed = smooth_scores_ema(ep_scores, window)
            for i in range(N):
                if i == victim_idx:
                    continue
                if np.all(np.isnan(smoothed[:, i, victim_idx])):
                    continue
                total_attack += 1
                valid_mask = ~np.isnan(smoothed[:, i, victim_idx])
                below_mask = smoothed[:, i, victim_idx] < threshold
                combined = valid_mask & below_mask
                below = np.where(combined)[0]
                if len(below) > 0:
                    detect_times.append(below[0])
                    detected += 1

        if detect_times:
            avg_time_arr[th_idx] = np.mean(detect_times)
        detect_rate_arr[th_idx] = detected / max(total_attack, 1)

    return fpr_arr, avg_time_arr, detect_rate_arr



def setup_plot_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


COLORS = {
    'ACT': '#e74c3c',
    'DYN': '#3498db',
    'DYN_0.01': '#2ecc71',
    'DYN_0.02': '#27ae60',
    'DYN_0.05': '#27ae60',
    'DYN_0.1': '#3498db',
    'DYN_0.15': '#2980b9',
    'DYN_0.5': '#8e44ad',
    'DYN_1.0': '#9b59b6',
    'random': '#95a5a6',
    'grad': '#f39c12',
}

def get_color(attack_type):
    if attack_type in COLORS:
        return COLORS[attack_type]
    for key in COLORS:
        if key in attack_type:
            return COLORS[key]
    return '#34495e'


def plot_roc_curves(roc_results, output_path, title_prefix=""):
    setup_plot_style()

    windows = sorted(set(w for ar in roc_results.values() for w in ar.keys()))
    n_windows = len(windows)

    fig, axes = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5.5))
    if n_windows == 1:
        axes = [axes]

    for w_idx, window in enumerate(windows):
        ax = axes[w_idx]

        for attack_type, attack_results in roc_results.items():
            if window not in attack_results:
                continue
            fpr, tpr, auc_val = attack_results[window]
            color = get_color(attack_type)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{attack_type} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        w_label = "Cumulative Avg" if window == -1 else f"EMA w={window}"
        ax.set_title(f'{title_prefix}ROC Curve ({w_label})')
        ax.legend(loc='lower right')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] ROC curves → {output_path}")


def plot_step_level_roc(step_roc_results, output_path):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(7, 6))

    for attack_type, (fpr, tpr, auc_val) in step_roc_results.items():
        color = get_color(attack_type)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{attack_type} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Step-Level ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] Step-level ROC → {output_path}")


def plot_detection_time_vs_fpr(dt_results, output_path):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    for idx, (attack_type, (fprs, times, rates)) in enumerate(dt_results.items()):
        color = get_color(attack_type)
        marker = markers[idx % len(markers)]

        valid = [i for i, t in enumerate(times) if t < float('inf')]
        if not valid:
            continue

        fprs_v = fprs[valid]
        times_v = times[valid]
        rates_v = rates[valid]

        sort_idx = np.argsort(fprs_v)
        fprs_v = fprs_v[sort_idx]
        times_v = times_v[sort_idx]
        rates_v = rates_v[sort_idx]

        step = max(1, len(fprs_v) // 15)

        ax1.plot(fprs_v, times_v, color=color, linewidth=2, label=attack_type)
        ax1.scatter(fprs_v[::step], times_v[::step], color=color,
                    marker=marker, s=30, zorder=5)

        ax2.plot(fprs_v, rates_v, color=color, linewidth=2, label=attack_type)
        ax2.scatter(fprs_v[::step], rates_v[::step], color=color,
                    marker=marker, s=30, zorder=5)

    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('Average Detection Time (steps)')
    ax1.set_title('Detection Time vs FPR (lower is better)')
    ax1.legend()
    ax1.set_xlim(left=-0.02)

    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('Detection Rate (TPR)')
    ax2.set_title('Detection Rate vs FPR (higher is better)')
    ax2.legend()
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] Detection time → {output_path}")


def plot_score_trajectories(attack_data, normal_data, victim_idx, window,
                             output_path, num_examples=3):
    setup_plot_style()

    N = attack_data['scores'][0].shape[1]
    n_ex = min(num_examples, len(attack_data['scores']), len(normal_data['scores']))

    fig, axes = plt.subplots(2, n_ex, figsize=(5 * n_ex, 8), squeeze=False)

    for ex in range(n_ex):
        smoothed = smooth_scores_ema(attack_data['scores'][ex], window)
        ax = axes[0, ex]

        for i in range(N):
            if i == victim_idx:
                continue
            vals = smoothed[:, i, victim_idx]
            if np.all(np.isnan(vals)):
                continue
            ax.plot(vals, color='red', alpha=0.6, linewidth=1.5,
                    label='Observer→Victim' if (ex == 0 and i == (0 if victim_idx != 0 else 1)) else None)

        plotted_normal = False
        for i in range(N):
            for j in range(N):
                if i == j or j == victim_idx or i == victim_idx:
                    continue
                vals = smoothed[:, i, j]
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(vals, color='blue', alpha=0.3, linewidth=0.8,
                        label='Observer→Normal' if (ex == 0 and not plotted_normal) else None)
                plotted_normal = True
                break
            if plotted_normal:
                break

        ax.set_title(f'Attack Ep {ex + 1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Smoothed Score')
        if ex == 0:
            ax.legend(fontsize=8, loc='lower left')

        smoothed_n = smooth_scores_ema(normal_data['scores'][ex], window)
        ax2 = axes[1, ex]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                vals = smoothed_n[:, i, j]
                if np.all(np.isnan(vals)):
                    continue
                color = 'green' if j == victim_idx else 'blue'
                alpha = 0.6 if j == victim_idx else 0.25
                ax2.plot(vals, color=color, alpha=alpha, linewidth=0.8)

        ax2.set_title(f'Normal Ep {ex + 1}')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Smoothed Score')

    w_str = "cumulative" if window == -1 else f"w={window}"
    fig.suptitle(f'Score Trajectories ({w_str})', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] Trajectories → {output_path}")


def plot_score_distributions(attack_data, normal_data, victim_idx, window,
                              output_path, method="final"):
    setup_plot_style()

    pos_scores = []
    neg_scores = []

    for ep_scores in attack_data['scores']:
        vs, _ = get_episode_level_scores(ep_scores, victim_idx, window, method)
        pos_scores.extend(vs)

    for ep_scores in normal_data['scores']:
        vs, ns = get_episode_level_scores(ep_scores, victim_idx, window, method)
        neg_scores.extend(vs)
        neg_scores.extend(ns)

    pos_scores = [s for s in pos_scores if not np.isnan(s)]
    neg_scores = [s for s in neg_scores if not np.isnan(s)]

    if not pos_scores or not neg_scores:
        print(f"[Warn] Empty scores for distribution plot, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(min(pos_scores), min(neg_scores)),
        max(max(pos_scores), max(neg_scores)),
        50)

    ax.hist(neg_scores, bins=bins, alpha=0.6, color='blue',
            label=f'Normal (n={len(neg_scores)})', density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color='red',
            label=f'Attacked (n={len(pos_scores)})', density=True)

    ax.set_xlabel('Anomaly Score (log probability)')
    ax.set_ylabel('Density')
    at = attack_data['attack_type']
    w_str = "cumulative" if window == -1 else f"w={window}"
    ax.set_title(f'Score Distribution: {at} attack ({w_str}, {method})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] Distribution → {output_path}")



def save_scores_data(data, output_path):
    max_T = max(s.shape[0] for s in data['scores'])
    N = data['scores'][0].shape[1]
    n_ep = len(data['scores'])

    scores_padded = np.zeros((n_ep, max_T, N, N), dtype=np.float32)
    for i, s in enumerate(data['scores']):
        scores_padded[i, :s.shape[0]] = s

    np.savez_compressed(
        output_path,
        scores=scores_padded,
        lengths=np.array(data['lengths']),
        rewards=np.array(data['rewards']),
        attack_type=np.array(data['attack_type']),
        victim_idx=np.array(data['victim_idx']),
    )
    print(f"[Data] Saved → {output_path}")


def load_scores_data(input_path):
    raw = np.load(input_path, allow_pickle=True)

    lengths = raw['lengths']
    scores_padded = raw['scores']

    scores = []
    for i in range(len(lengths)):
        scores.append(scores_padded[i, :lengths[i]])

    return {
        'scores': scores,
        'rewards': raw['rewards'].tolist(),
        'lengths': lengths.tolist(),
        'attack_type': str(raw['attack_type']),
        'victim_idx': int(raw['victim_idx']),
    }



def generate_all_plots(all_data, normal_data, victim_idx, windows, roc_method,
                        output_dir):

    attack_types = [k for k in all_data.keys() if k != 'none']

    if not attack_types:
        print("[Warn] No attack data found. Only baseline collected.")
        return

    print("\n[3/5] Computing episode-level ROC curves...")
    roc_results = {}
    for at in attack_types:
        roc_results[at] = {}
        for window in windows:
            fpr, tpr, th, auc_val = compute_roc_curve(
                all_data[at], normal_data, victim_idx, window, roc_method)
            roc_results[at][window] = (fpr, tpr, auc_val)
            print(f"  {at:>10}, window={window:>3}: AUC = {auc_val:.4f}")

    plot_roc_curves(
        roc_results, os.path.join(output_dir, "roc_episode_level.png"),
        title_prefix="Episode-Level ")

    print("\n[3.5/5] Computing step-level ROC curves...")
    best_window = windows[-1]
    step_roc = {}
    for at in attack_types:
        fpr, tpr, th, auc_val = compute_step_level_roc(
            all_data[at], normal_data, victim_idx, best_window)
        step_roc[at] = (fpr, tpr, auc_val)
        print(f"  {at:>10}: Step-level AUC = {auc_val:.4f}")

    plot_step_level_roc(step_roc, os.path.join(output_dir, "roc_step_level.png"))

    print("\n[4/5] Computing detection time curves...")
    score_thresholds = np.linspace(-150, 0, 300).tolist()

    dt_results = {}
    for at in attack_types:
        fprs, times, rates = compute_detection_time_vs_fpr(
            all_data[at], normal_data, victim_idx, best_window, score_thresholds)
        dt_results[at] = (fprs, times, rates)

        for target_fpr in [0.05, 0.10, 0.20]:
            idx = np.argmin(np.abs(fprs - target_fpr))
            print(f"  {at:>10} @ FPR≈{fprs[idx]:.2f}: "
                  f"detect_time={times[idx]:.1f}, detect_rate={rates[idx]:.2f}")

    plot_detection_time_vs_fpr(
        dt_results, os.path.join(output_dir, "detection_time_vs_fpr.png"))

    print("\n[5/5] Generating visualizations...")
    for at in attack_types:
        plot_score_trajectories(
            all_data[at], normal_data, victim_idx, best_window,
            os.path.join(output_dir, f"trajectories_{at}.png"))

        plot_score_distributions(
            all_data[at], normal_data, victim_idx, best_window,
            os.path.join(output_dir, f"distribution_{at}.png"),
            method=roc_method)

    summary = {
        'victim_idx': victim_idx,
        'num_episodes_baseline': len(normal_data['scores']),
        'avg_reward_baseline': float(np.mean(normal_data['rewards'])),
        'windows': windows,
        'roc_method': roc_method,
        'attacks': {},
    }

    for at in attack_types:
        attack_summary = {
            'num_episodes': len(all_data[at]['scores']),
            'avg_reward': float(np.mean(all_data[at]['rewards'])),
            'reward_change_pct': float(
                (np.mean(all_data[at]['rewards']) - np.mean(normal_data['rewards']))
                / abs(np.mean(normal_data['rewards'])) * 100),
            'episode_level_auc': {
                str(w): float(roc_results[at][w][2]) for w in windows},
            'step_level_auc': float(step_roc[at][2]) if at in step_roc else None,
        }
        summary['attacks'][at] = attack_summary

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Baseline avg reward:  {summary['avg_reward_baseline']:.2f}")
    for at in attack_types:
        s = summary['attacks'][at]
        print(f"\n  [{at}]")
        print(f"    Avg reward:         {s['avg_reward']:.2f} "
              f"({s['reward_change_pct']:+.1f}%)")
        for w in windows:
            w_str = "cum" if w == -1 else f"w={w}"
            print(f"    ROC AUC ({w_str:>5}):  {s['episode_level_auc'][str(w)]:.4f}")
        if s['step_level_auc']:
            print(f"    Step-level AUC:     {s['step_level_auc']:.4f}")

    print(f"\n  Output directory: {output_dir}/")
    print(f"{'=' * 65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tracker Performance Evaluation — ROC & Detection Time")

    parser.add_argument("--plot_only", action="store_true",
                        help="Generate plots from saved .npz files")
    parser.add_argument("--data_files", nargs="+", default=None,
                        help=".npz files for --plot_only mode")

    parser.add_argument("--phase1_model_dir", type=str, default=None)
    parser.add_argument("--phase2_model_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)

    parser.add_argument("--attack_type", type=str, default="ACT")
    parser.add_argument("--attack_model_path", type=str, default=None)
    parser.add_argument("--victim_idx", type=int, default=3)
    parser.add_argument("--budget", type=float, default=0.35)

    parser.add_argument("--scenario", type=str, default="simple_spread_v2",
                        help="Environment scenario: simple_spread_v2 or simple_tag_v2")
    parser.add_argument("--num_agents", type=int, default=6)
    parser.add_argument("--max_cycles", type=int, default=25,
                        help="Max steps per episode (simple_spread=25, simple_tag=50)")
    parser.add_argument("--num_episodes", type=int, default=100)

    parser.add_argument("--windows", nargs="+", type=int, default=[-1, 5, 10])
    parser.add_argument("--roc_method", type=str, default="final",
                        choices=["final", "mean", "min"])

    parser.add_argument("--output_dir", type=str, default="eval_tracker_results_DYN0.02")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--auto_calibrate", action="store_true", default=True,
                        help="Auto-calibrate cluster (fixes obs pipeline inconsistency)")
    parser.add_argument("--no_auto_calibrate", action="store_false", dest="auto_calibrate")
    parser.add_argument("--num_calib_episodes", type=int, default=20,
                        help="Number of episodes used for calibration")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        if not args.data_files:
            parser.error("--data_files required in --plot_only mode")

        all_data = {}
        normal_data = None
        victim_idx = args.victim_idx

        for f in args.data_files:
            data = load_scores_data(f)
            at = data['attack_type']
            all_data[at] = data
            victim_idx = data['victim_idx']
            if at == 'none':
                normal_data = data
            print(f"  Loaded {f}: type={at}, episodes={len(data['scores'])}")

        if normal_data is None:
            print("[Warn] No 'none' baseline in data files. "
                  "Using first file as baseline reference.")
            normal_data = list(all_data.values())[0]

        generate_all_plots(all_data, normal_data, victim_idx,
                           args.windows, args.roc_method, args.output_dir)
        return

    if args.phase1_model_dir is None:
        parser.error("--phase1_model_dir required (or use --plot_only)")

    if args.phase2_model_dir is None:
        args.phase2_model_dir = args.phase1_model_dir

    print(f"\n{'=' * 65}")
    print(f"  Tracker Performance Evaluation")
    print(f"  Scenario: {args.scenario}")
    print(f"  Attack: {args.attack_type}  |  Victim: agent_{args.victim_idx}")
    print(f"  Episodes: {args.num_episodes} attack + {args.num_episodes} baseline")
    print(f"  Windows: {args.windows}  |  Method: {args.roc_method}")
    print(f"{'=' * 65}\n")

    print("[1/2] Collecting baseline (no attack) scores...")
    normal_data = collect_episode_scores(
        args.phase1_model_dir, args.phase2_model_dir,
        attack_type="none", attack_model_path=None,
        victim_idx=args.victim_idx, num_agents=args.num_agents,
        num_episodes=args.num_episodes, max_cycles=args.max_cycles,
        config_path=args.config_path, device=args.device,
        scenario=args.scenario, auto_calibrate=True, num_calib_episodes=20)
    save_scores_data(normal_data,
                     os.path.join(args.output_dir, "scores_none.npz"))

    print(f"\n[2/2] Collecting {args.attack_type} attack scores...")
    attack_data = collect_episode_scores(
        args.phase1_model_dir, args.phase2_model_dir,
        attack_type=args.attack_type,
        attack_model_path=args.attack_model_path,
        victim_idx=args.victim_idx, num_agents=args.num_agents,
        num_episodes=args.num_episodes, max_cycles=args.max_cycles,
        budget=args.budget, config_path=args.config_path, device=args.device,
        scenario=args.scenario, auto_calibrate=True,
        _cached_calibration=normal_data['_calibration'])
    save_scores_data(attack_data,
                     os.path.join(args.output_dir,
                                  f"scores_{args.attack_type}.npz"))

    all_data = {'none': normal_data, args.attack_type: attack_data}
    generate_all_plots(all_data, normal_data, args.victim_idx,
                       args.windows, args.roc_method, args.output_dir)


if __name__ == "__main__":
    main()
