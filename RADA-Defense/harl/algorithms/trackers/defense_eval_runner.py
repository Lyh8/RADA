import numpy as np
import torch
import os
import json
import time
import types

from harl.utils.trans_tools import _t2n

from harl.algorithms.trackers.defense import (
    ContinuousDefenseModule, DefenseConfig, ThresholdCalibrator,
    ALL_METHODS,
    METHOD_NONE, METHOD_NO_DEFENSE, METHOD_ORACLE,
)


_VERSION = "2026-03-03-v1-defense-eval"


class DefenseEvalMixin:


    @torch.no_grad()
    def eval_with_defense(self, attacker, defense_config, shadow_envs=None):
        cfg = defense_config
        victim = cfg.victim_agent_id
        N = self.num_agents
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]

        act_space = self.eval_envs.action_space[0]
        action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 5
        cfg.action_low = float(getattr(act_space, 'low', np.zeros(1))[0])
        cfg.action_high = float(getattr(act_space, 'high', np.ones(1))[0])
        cfg.disable_clip = True

        method = cfg.internal_method
        is_attack = (method != METHOD_NONE)
        use_defense = method not in (METHOD_NONE, METHOD_NO_DEFENSE)
        use_shadow = (shadow_envs is not None)

        defense = ContinuousDefenseModule(cfg, N, action_dim, self.device)
        defense.reset_episode(n_threads)

        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        shadow_obs = None
        if use_shadow:
            shadow_obs, _, _ = shadow_envs.reset()

        eval_rnn = np.zeros(
            (n_threads, N, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        shadow_rnn = np.zeros_like(eval_rnn) if use_shadow else None
        eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)

        self.tracker.prep_rollout()
        self.tracker.init_hidden(n_threads)
        if hasattr(self, 'recl') and self.recl is not None:
            self.recl.prep_rollout()
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((n_threads, N, action_dim), dtype=np.float32)

        episode_count = 0
        ep_rewards = np.zeros(n_threads)
        ep_steps = np.zeros(n_threads, dtype=int)
        ep_caught = np.zeros(n_threads, dtype=bool)

        max_episodes = cfg.eval_episodes

        print(f"  [Eval] method={cfg.defense_method}({method}), "
              f"attack={'ON' if is_attack else 'OFF'}, "
              f"defense={'ON' if use_defense else 'OFF'}, "
              f"shadow={'ON' if use_shadow else 'OFF'}, "
              f"episodes={max_episodes}")

        while episode_count < max_episodes:
            acts_list = []
            for aid in range(N):
                act_out, rnn_out = self.actor[aid].act(
                    eval_obs[:, aid],
                    eval_rnn[:, aid],
                    eval_masks[:, aid],
                    eval_avail[:, aid] if eval_avail[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn[:, aid] = _t2n(rnn_out)
                acts_list.append(_t2n(act_out))
            eval_actions = np.array(acts_list).transpose(1, 0, 2)

            shadow_actions = None
            if use_shadow and shadow_obs is not None:
                s_list = []
                for aid in range(N):
                    s_act, s_rnn = self.actor[aid].act(
                        shadow_obs[:, aid],
                        shadow_rnn[:, aid],
                        eval_masks[:, aid],
                        None,
                        deterministic=True,
                    )
                    shadow_rnn[:, aid] = _t2n(s_rnn)
                    s_list.append(_t2n(s_act))
                shadow_actions = np.array(s_list).transpose(1, 0, 2)

            victim_normal = eval_actions[:, victim].copy()

            final_actions = eval_actions.copy()
            if is_attack:
                atk_acts = attacker.get_batch_attack_actions(
                    eval_obs[:, victim],
                    eval_actions[:, victim],
                    deterministic=True,
                )
                final_actions[:, victim] = atk_acts

            obs_t = torch.as_tensor(
                eval_obs, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(
                last_actions, dtype=torch.float32, device=self.device)

            role_embed = self._defense_get_role_embed(obs_t, n_threads)

            mu, std, _ = self.tracker.forward(obs_t, act_t, role_embed)
            mu_np = mu.cpu().numpy()
            std_np = std.cpu().numpy()

            act_score = torch.as_tensor(
                final_actions, dtype=torch.float32, device=self.device)
            scores_all = self.tracker.compute_scores(
                mu, std, act_score,
                eval_masks.squeeze(-1),
            )

            defense.update_detection(scores_all)

            if defense.current_step <= 50 and defense.current_step % 10 == 0:
                v = cfg.victim_agent_id
                locked = defense.defense_locked[0, v] if defense.defense_locked is not None else False
                if defense.current_step <= 10 or locked:
                    all_s = []
                    n_nan, n_stage1 = 0, 0
                    for i in range(N):
                        if i == v:
                            continue
                        s = scores_all[0, i, v]
                        if np.isnan(s):
                            n_nan += 1
                        elif s < -400:
                            n_stage1 += 1
                        else:
                            all_s.append(s)
                    agg_str = f"{np.mean(all_s):.2f}" if all_s else "N/A"
                    print(f"  [DEBUG] step={defense.current_step:>3d} | "
                          f"valid={len(all_s)} nan={n_nan} stage1={n_stage1} | "
                          f"agg={agg_str:>8s} | locked={locked}")

            labels_np = None
            if self.tracker.current_labels is not None:
                labels_np = self.tracker.current_labels.cpu().numpy()

            oracle_actions = None
            if method == METHOD_ORACLE:
                oracle_actions = eval_actions.copy()

            if use_defense:
                final_actions, applied = defense.get_corrected_actions(
                    final_actions, mu_np, std_np, labels_np, oracle_actions,
                )

                for b in range(n_threads):
                    if applied[b, victim]:
                        defense.record_correction(
                            b, final_actions[b, victim], victim_normal[b])
            else:
                for b in range(n_threads):
                    defense._ep_total_steps[b] += 1

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_avail) = self.eval_envs.step(final_actions)

            if use_shadow and shadow_obs is not None and shadow_actions is not None:
                (shadow_obs, _, _, _, _, _) = shadow_envs.step(shadow_actions)

            last_actions = final_actions.copy()

            for b in range(n_threads):
                ep_rewards[b] += float(eval_rewards[b].mean())
                ep_steps[b] += 1

                if eval_infos is not None:
                    for info in eval_infos[b]:
                        if isinstance(info, dict):
                            if info.get("catch", False) or info.get("caught", False):
                                ep_caught[b] = True

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn[eval_dones_env] = 0
            eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)
            eval_masks[eval_dones_env] = 0

            for b in range(n_threads):
                if not eval_dones_env[b]:
                    continue

                episode_count += 1

                defense.end_episode(b, ep_rewards[b], ep_caught[b])
                v = cfg.victim_agent_id
                ep_rewards[b] = 0.0
                ep_steps[b] = 0
                ep_caught[b] = False
                last_actions[b] = 0.0
                self.tracker.hidden[b] = 0.0
                if use_shadow and shadow_rnn is not None:
                    shadow_rnn[b] = 0.0

                if episode_count % 100 == 0:
                    s = defense.get_summary()
                    print(f"    [{episode_count}/{max_episodes}] "
                          f"R={s['mean_reward']:.2f}±{s['std_reward']:.2f}")

                if episode_count >= max_episodes:
                    break

        return defense

    def _defense_get_role_embed(self, obs_tensor, n_threads):
        use_recl = (hasattr(self, 'recl')
                    and self.recl is not None
                    and hasattr(self, 'use_recl')
                    and self.use_recl)
        if use_recl:
            ae = self.recl.embedding_net.agent_embed_forward(
                obs_tensor.reshape(-1, self.tracker.obs_dim), detach=True)
            re = self.recl.embedding_net.role_embed_forward(
                ae, detach=True, ema=False)
            return re.reshape(n_threads, self.num_agents, -1)
        else:
            return torch.zeros(
                n_threads, self.num_agents,
                self.tracker.role_embedding_dim,
                dtype=torch.float32, device=self.device,
            )


    @torch.no_grad()
    def calibrate_threshold(self, n_episodes=100, ema_window=5,
                             target_fpr=0.05):
        N = self.num_agents
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]

        act_space = self.eval_envs.action_space[0]
        action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 5

        calibrator = ThresholdCalibrator(N, ema_window=ema_window)

        print(f"\n[Calibrate] Running {n_episodes} normal episodes "
              f"(ema_window={ema_window}, target_fpr={target_fpr})")

        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        eval_rnn = np.zeros(
            (n_threads, N, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)

        self.tracker.prep_rollout()
        self.tracker.init_hidden(n_threads)
        if hasattr(self, 'recl') and self.recl is not None:
            self.recl.prep_rollout()
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((n_threads, N, action_dim), dtype=np.float32)
        calibrator.reset_episode(n_threads)

        episode_count = 0

        while episode_count < n_episodes:
            acts_list = []
            for aid in range(N):
                act_out, rnn_out = self.actor[aid].act(
                    eval_obs[:, aid],
                    eval_rnn[:, aid],
                    eval_masks[:, aid],
                    eval_avail[:, aid] if eval_avail[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn[:, aid] = _t2n(rnn_out)
                acts_list.append(_t2n(act_out))
            eval_actions = np.array(acts_list).transpose(1, 0, 2)

            obs_t = torch.as_tensor(
                eval_obs, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(
                last_actions, dtype=torch.float32, device=self.device)
            role_embed = self._defense_get_role_embed(obs_t, n_threads)

            mu, std, _ = self.tracker.forward(obs_t, act_t, role_embed)
            act_score = torch.as_tensor(
                eval_actions, dtype=torch.float32, device=self.device)
            scores_all = self.tracker.compute_scores(
                mu, std, act_score, eval_masks.squeeze(-1))

            calibrator.collect_step(scores_all)

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_avail) = self.eval_envs.step(eval_actions)

            last_actions = eval_actions.copy()

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn[eval_dones_env] = 0
            eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)
            eval_masks[eval_dones_env] = 0

            for b in range(n_threads):
                if eval_dones_env[b]:
                    episode_count += 1
                    calibrator.end_episode()
                    last_actions[b] = 0.0
                    self.tracker.hidden[b] = 0.0

                    calibrator.score_ema[b] = 0.0

                    if episode_count % 50 == 0:
                        print(f"  [Calibrate] {episode_count}/{n_episodes}")

                    if episode_count >= n_episodes:
                        break

        diag = calibrator.print_report()
        threshold = calibrator.compute_threshold(target_fpr)

        print(f"\n  >>> Recommended η* = {threshold:.4f} "
              f"(FPR={target_fpr:.0%})")

        if hasattr(self, 'log_dir') and self.log_dir:
            cal_path = os.path.join(
                str(self.log_dir), "calibration_result.json")
            os.makedirs(os.path.dirname(cal_path), exist_ok=True)
            result = {
                "threshold": threshold,
                "target_fpr": target_fpr,
                "ema_window": ema_window,
                "n_episodes": n_episodes,
                "diagnostics": diag,
            }
            with open(cal_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  [Calibrate] Saved → {cal_path}")

        return threshold


    def run_main_comparison(self, attacker, attack_name="ACT",
                            methods=None, n_episodes=500,
                            victim_id=0, detection_threshold=-11.0,
                            shadow_envs=None,
                            w_same=3.0, w_diff=0.0):
        if methods is None:
            methods = ALL_METHODS

        results = {}
        t0 = time.time()

        for method_label in methods:
            print(f"\n{'=' * 60}")
            print(f"[Exp1] {method_label} vs {attack_name}")
            print(f"{'=' * 60}")

            cfg = DefenseConfig(
                defense_method=method_label,
                victim_agent_id=victim_id,
                eval_episodes=n_episodes,
                detection_threshold=detection_threshold,
                w_same=w_same,
                w_diff=w_diff,
            )

            defense_mod = self.eval_with_defense(
                attacker=attacker,
                defense_config=cfg,
                shadow_envs=shadow_envs,
            )

            summary = defense_mod.get_summary()
            results[method_label] = summary
            _print_method_result(method_label, summary)

        if "B0" in results and "B1" in results:
            r_n = results["B0"]["mean_reward"]
            r_a = results["B1"]["mean_reward"]
            for m in results:
                if m in ("B0", "B1"):
                    continue
                r_d = results[m]["mean_reward"]
                results[m]["recovery_rate"] = \
                    ContinuousDefenseModule.compute_recovery_rate(r_d, r_a, r_n)

        _print_result_table(results, attack_name, victim_id)

        results["_meta"] = {
            "attack": attack_name,
            "victim": victim_id,
            "n_episodes": n_episodes,
            "threshold": detection_threshold,
            "w_same": w_same,
            "w_diff": w_diff,
            "elapsed_s": time.time() - t0,
            "version": _VERSION,
        }
        if hasattr(self, 'log_dir') and self.log_dir:
            path = os.path.join(
                str(self.log_dir),
                f"defense_exp1_{attack_name}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[Result] Saved → {path}")

        return results


    def run_delay_experiment(self, attacker, attack_name="ACT",
                             delays=None, n_episodes=200,
                             victim_id=3, detection_threshold=-8.94,
                             defense_method="C4"):
        if delays is None:
            delays = [0, 2, 5, 10, 15]

        results = {}
        t0 = time.time()

        for delay in delays:
            print(f"\n[Delay Exp] delay={delay}, method={defense_method}")
            cfg = DefenseConfig(
                defense_method=defense_method,
                victim_agent_id=victim_id,
                eval_episodes=n_episodes,
                detection_threshold=detection_threshold,
                forced_delay=delay,
            )
            dm = self.eval_with_defense(attacker, cfg)
            results[delay] = dm.get_summary()
            print(f"  delay={delay}: R={results[delay]['mean_reward']:.2f}")

        print(f"\n[Delay Exp] No Defense baseline")
        cfg_nd = DefenseConfig(
            defense_method="B1",
            victim_agent_id=victim_id,
            eval_episodes=n_episodes,
        )
        dm_nd = self.eval_with_defense(attacker, cfg_nd)
        results["no_defense"] = dm_nd.get_summary()

        results["_meta"] = {
            "attack": attack_name,
            "victim": victim_id,
            "delays": delays,
            "elapsed_s": time.time() - t0,
        }
        if hasattr(self, 'log_dir') and self.log_dir:
            path = os.path.join(
                str(self.log_dir),
                f"defense_exp4_delay_{attack_name}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[Result] Saved → {path}")

        return results


    @torch.no_grad()
    def eval_single_trajectory(self, attacker, defense_config,
                                pos_indices=(2, 3)):
        from harl.algorithms.trackers.defense_trajectory import TrajectoryData

        cfg = defense_config
        victim = cfg.victim_agent_id
        N = self.num_agents
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]

        act_space = self.eval_envs.action_space[0]
        action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 5
        cfg.action_low = float(getattr(act_space, 'low', np.zeros(1))[0])
        cfg.action_high = float(getattr(act_space, 'high', np.ones(1))[0])
        cfg.disable_clip = True

        method = cfg.internal_method
        is_attack = (method != METHOD_NONE)
        use_defense = method not in (METHOD_NONE, METHOD_NO_DEFENSE)

        defense = ContinuousDefenseModule(cfg, N, action_dim, self.device)
        defense.reset_episode(n_threads)

        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        eval_rnn = np.zeros(
            (n_threads, N, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32)
        eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)

        self.tracker.prep_rollout()
        self.tracker.init_hidden(n_threads)
        if hasattr(self, 'recl') and self.recl is not None:
            self.recl.prep_rollout()
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((n_threads, N, action_dim), dtype=np.float32)

        pidx = list(pos_indices)
        traj_positions = [eval_obs[0, :, pidx].copy()]
        traj_defense_mask = []
        ep_reward = 0.0
        ep_caught = False
        thread0_done = False

        print(f"  [Traj] Collecting 1 episode: method={cfg.defense_method}, "
              f"attack={'ON' if is_attack else 'OFF'}")

        while not thread0_done:
            acts_list = []
            for aid in range(N):
                act_out, rnn_out = self.actor[aid].act(
                    eval_obs[:, aid], eval_rnn[:, aid],
                    eval_masks[:, aid],
                    eval_avail[:, aid] if eval_avail[0] is not None else None,
                    deterministic=True)
                eval_rnn[:, aid] = _t2n(rnn_out)
                acts_list.append(_t2n(act_out))
            eval_actions = np.array(acts_list).transpose(1, 0, 2)

            victim_normal = eval_actions[:, victim].copy()

            final_actions = eval_actions.copy()
            if is_attack:
                atk_acts = attacker.get_batch_attack_actions(
                    eval_obs[:, victim], eval_actions[:, victim],
                    deterministic=True)
                final_actions[:, victim] = atk_acts

            obs_t = torch.as_tensor(eval_obs, dtype=torch.float32,
                                    device=self.device)
            act_t = torch.as_tensor(last_actions, dtype=torch.float32,
                                    device=self.device)
            role_embed = self._defense_get_role_embed(obs_t, n_threads)
            mu, std, _ = self.tracker.forward(obs_t, act_t, role_embed)
            mu_np = mu.cpu().numpy()
            std_np = std.cpu().numpy()

            act_score = torch.as_tensor(final_actions, dtype=torch.float32,
                                        device=self.device)
            scores_all = self.tracker.compute_scores(
                mu, std, act_score, eval_masks.squeeze(-1))

            defense.update_detection(scores_all)

            labels_np = None
            if self.tracker.current_labels is not None:
                labels_np = self.tracker.current_labels.cpu().numpy()

            oracle_actions = None
            if method == METHOD_ORACLE:
                oracle_actions = eval_actions.copy()

            if use_defense:
                final_actions, applied = defense.get_corrected_actions(
                    final_actions, mu_np, std_np, labels_np, oracle_actions)
                for b in range(n_threads):
                    if applied[b, victim]:
                        defense.record_correction(
                            b, final_actions[b, victim], victim_normal[b])
            else:
                for b in range(n_threads):
                    defense._ep_total_steps[b] += 1

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_avail) = self.eval_envs.step(final_actions)

            last_actions = final_actions.copy()

            traj_positions.append(eval_obs[0, :, pidx].copy())
            dmask = np.zeros(N, dtype=bool)
            if defense.defense_locked is not None:
                dmask = defense.defense_locked[0].copy()
            traj_defense_mask.append(dmask)

            ep_reward += float(eval_rewards[0].mean())
            if eval_infos is not None:
                for info in eval_infos[0]:
                    if isinstance(info, dict):
                        if info.get("catch", False) or info.get("caught", False):
                            ep_caught = True

            eval_dones_env = np.all(eval_dones, axis=1)
            if eval_dones_env[0]:
                thread0_done = True

            eval_rnn[eval_dones_env] = 0
            eval_masks = np.ones((n_threads, N, 1), dtype=np.float32)
            eval_masks[eval_dones_env] = 0
            for b in range(n_threads):
                if eval_dones_env[b]:
                    last_actions[b] = 0.0
                    self.tracker.hidden[b] = 0.0

        det = -1
        if defense.detection_step is not None:
            det = int(defense.detection_step[0, victim])

        from harl.algorithms.trackers.defense_trajectory import METHOD_DISPLAY
        traj = TrajectoryData(
            method=cfg.defense_method,
            method_label=METHOD_DISPLAY.get(cfg.defense_method,
                                            cfg.defense_method),
            victim_id=victim,
            num_agents=N,
            positions=traj_positions,
            detection_step=det,
            defense_active_mask=traj_defense_mask,
            total_reward=ep_reward,
            caught=ep_caught,
        )
        print(f"  [Traj] Done: {len(traj_positions)} steps, "
              f"R={ep_reward:.1f}, det@{det}")
        return traj

    def run_trajectory_experiment(self, attacker, attack_name="ACT",
                                   victim_id=3, detection_threshold=-8.94,
                                   methods=None, pos_indices=(2, 3),
                                   n_candidates=3, output_dir=None,
                                   w_same=3.0, w_diff=0.0,
                                   prey_agent_ids=None):
        from harl.algorithms.trackers.defense_trajectory import (
            save_trajectories, save_trajectories_npz,
            plot_trajectory_comparison,
        )

        if methods is None:
            methods = ["B0", "B1", "C4"]
        if output_dir is None:
            output_dir = getattr(self, 'log_dir', '.') or '.'
        os.makedirs(output_dir, exist_ok=True)

        best_trajectories = []

        for method_label in methods:
            print(f"\n{'=' * 50}")
            print(f"[Traj] Collecting {n_candidates} candidate(s) "
                  f"for {method_label}")
            print(f"{'=' * 50}")

            candidates = []
            for ep_i in range(n_candidates):
                cfg = DefenseConfig(
                    defense_method=method_label,
                    victim_agent_id=victim_id,
                    eval_episodes=1,
                    detection_threshold=detection_threshold,
                    w_same=w_same,
                    w_diff=w_diff,
                )
                traj = self.eval_single_trajectory(
                    attacker, cfg, pos_indices=pos_indices)
                candidates.append(traj)
                print(f"    candidate {ep_i}: R={traj.total_reward:.1f}, "
                      f"det={traj.detection_step}, "
                      f"steps={len(traj.positions)}")

            candidates.sort(key=lambda t: t.total_reward)
            best = candidates[len(candidates) // 2]
            print(f"  → Selected: R={best.total_reward:.1f}")
            best_trajectories.append(best)

        json_path = os.path.join(output_dir,
                                 f"trajectory_data_{attack_name}.json")
        npz_path = os.path.join(output_dir,
                                f"trajectory_data_{attack_name}.npz")
        pdf_path = os.path.join(output_dir,
                                f"trajectory_comparison_{attack_name}.pdf")

        save_trajectories(best_trajectories, json_path)
        save_trajectories_npz(best_trajectories, npz_path)
        plot_trajectory_comparison(best_trajectories, pdf_path,
                                    prey_agent_ids=prey_agent_ids)

        all_json_path = os.path.join(
            output_dir, f"trajectory_all_candidates_{attack_name}.json")
        all_candidates = []
        save_trajectories(best_trajectories, all_json_path)

        print(f"\n[Traj] Experiment complete:")
        print(f"  Data  → {json_path}")
        print(f"  NPZ   → {npz_path}")
        print(f"  PDF   → {pdf_path}")

        return best_trajectories



def _print_method_result(method, summary):
    r = summary.get('mean_reward', 0)
    s = summary.get('std_reward', 0)
    c = summary.get('catch_rate', 0)
    line = f"  {method}: R={r:.2f}±{s:.2f}, catch={c:.3f}"
    if 'mean_action_mse' in summary:
        line += f", MSE={summary['mean_action_mse']:.4f}"
    if 'mean_cosine_sim' in summary:
        line += f", cos={summary['mean_cosine_sim']:.4f}"
    if 'mean_detection_step' in summary:
        line += f", det_t={summary['mean_detection_step']:.1f}"
    print(line)

def _print_result_table(results, attack_name, victim_id):
    print(f"\n{'=' * 85}")
    print(f"  Experiment 1 Results | Attack: {attack_name} | Victim: {victim_id}")
    print(f"{'=' * 85}")
    header = (f"  {'Method':<10} {'Reward':>10} {'±std':>8} {'η(%)':>8} "
              f"{'Catch':>8} {'MSE':>10} {'CosSim':>8} {'DetStep':>8}")
    print(header)
    print(f"  {'-' * 80}")

    for m in ALL_METHODS:
        if m not in results:
            continue
        r = results[m]
        row = f"  {m:<10}"
        row += f" {r.get('mean_reward', 0):>10.2f}"
        row += f" {r.get('std_reward', 0):>8.2f}"
        eta = r.get('recovery_rate')
        if isinstance(eta, (int, float)):
            row += f" {eta:>8.1f}"
        else:
            row += f" {'—':>8}"
        row += f" {r.get('catch_rate', 0):>8.3f}"
        mse = r.get('mean_action_mse')
        if isinstance(mse, float):
            row += f" {mse:>10.4f}"
        else:
            row += f" {'—':>10}"
        cos = r.get('mean_cosine_sim')
        if isinstance(cos, float):
            row += f" {cos:>8.4f}"
        else:
            row += f" {'—':>8}"
        det = r.get('mean_detection_step')
        if isinstance(det, float):
            row += f" {det:>8.1f}"
        else:
            row += f" {'—':>8}"
        print(row)

    print(f"  {'=' * 80}\n")



def patch_runner_with_defense(runner):
    methods_to_patch = [
        'eval_with_defense',
        '_defense_get_role_embed',
        'calibrate_threshold',
        'run_main_comparison',
        'run_delay_experiment',
        'eval_single_trajectory',
        'run_trajectory_experiment',
    ]
    for name in methods_to_patch:
        method = getattr(DefenseEvalMixin, name)
        bound = types.MethodType(method, runner)
        setattr(runner, name, bound)

    print(f"[Defense] Patched {len(methods_to_patch)} methods onto runner")
    return runner
