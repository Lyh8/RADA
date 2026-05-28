import numpy as np
import torch
import os
import json

from harl.utils.trans_tools import _t2n

def inject_attack_eval_into_runner(runner_class):

    def set_attacker(self, attacker, victim_idx=0):
        self._attacker = attacker
        self._victim_idx = victim_idx
        print(f"[Eval] Attacker set: type={attacker.attack_type}, victim={victim_idx}")

    @torch.no_grad()
    def eval_with_attack(self):
        if not hasattr(self, '_attacker') or self._attacker is None:
            print("[Eval] No attacker set, using standard evaluation.")
            self.eval()
            return

        attacker = self._attacker
        victim_idx = self._victim_idx

        self.logger.eval_init()
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        n_eval_threads = self.algo_args["eval"]["n_eval_rollout_threads"]

        eval_rnn_states = np.zeros((
            n_eval_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size,
        ), dtype=np.float32)
        eval_masks = np.ones((n_eval_threads, self.num_agents, 1), dtype=np.float32)

        if self.use_tracker:
            self.tracker.init_hidden(n_eval_threads)
        if hasattr(self, 'use_recl') and self.use_recl:
            self.recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((n_eval_threads, self.num_agents,
                                 self.tracker.action_dim if self.use_tracker else 5))
        episode_rewards = [0.0] * n_eval_threads
        episode_rewards_no_attack = [0.0] * n_eval_threads
        episode_steps = [0] * n_eval_threads

        all_episode_stats = []

        while True:
            eval_actions_collector = []
            normal_actions_collector = []

            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                normal_action = _t2n(eval_actions)
                normal_actions_collector.append(normal_action.copy())

                if agent_id == victim_idx:
                    attacked_action = attacker.get_batch_attack_actions(
                        eval_obs[:, agent_id],
                        normal_action,
                        deterministic=True,
                    )
                    eval_actions_collector.append(attacked_action)
                else:
                    eval_actions_collector.append(normal_action)

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            normal_actions = np.array(normal_actions_collector).transpose(1, 0, 2)

            if self.use_tracker and hasattr(self, 'use_recl') and self.use_recl:
                obs_tensor = torch.tensor(
                    eval_obs, dtype=torch.float32, device=self.device)
                act_tensor = torch.tensor(
                    last_actions, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    agent_embed = self.recl.embedding_net.agent_embed_forward(
                        obs_tensor.reshape(-1, self.tracker.obs_dim), detach=True)
                    role_embed = self.recl.embedding_net.role_embed_forward(
                        agent_embed, detach=True, ema=False)
                    role_embed = role_embed.reshape(
                        n_eval_threads, self.num_agents, -1)

                mu, std, _ = self.tracker.forward(
                    obs_tensor, act_tensor, role_embed)
                action_tensor = torch.tensor(
                    eval_actions, dtype=torch.float32, device=self.device)
                scores = self.tracker.compute_scores(
                    mu, std, action_tensor, eval_masks.squeeze(-1))

                self.tracker.update_detection_stats(scores[0])

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_available_actions) = self.eval_envs.step(eval_actions)

            self.logger.eval_per_step((
                eval_obs, eval_share_obs, eval_rewards,
                eval_dones, eval_infos, eval_available_actions))

            last_actions = eval_actions.copy()

            for t in range(n_eval_threads):
                episode_rewards[t] += eval_rewards[t].mean()
                episode_steps[t] += 1

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = 0
            eval_masks = np.ones(
                (n_eval_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(n_eval_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

                    stats = {
                        'reward': episode_rewards[eval_i],
                        'length': episode_steps[eval_i],
                        'attack_type': attacker.attack_type,
                        'victim_idx': victim_idx,
                    }
                    all_episode_stats.append(stats)

                    if self.use_tracker:
                        self.tracker.end_episode(
                            reward=episode_rewards[eval_i],
                            episode_length=episode_steps[eval_i],
                            attacked_agent=victim_idx,
                        )

                    episode_rewards[eval_i] = 0.0
                    episode_steps[eval_i] = 0
                    last_actions[eval_i] = 0.0

                    if self.use_tracker:
                        self.tracker.hidden[eval_i] = 0.0

                    if hasattr(self, 'use_recl') and self.use_recl:
                        pass

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)

                if self.use_tracker:
                    self._print_attack_detection_summary(
                        attacker.attack_type, victim_idx, all_episode_stats)

                    log_path = os.path.join(
                        str(self.log_dir),
                        f"tracker_eval_{attacker.attack_type}_victim{victim_idx}.json"
                    )
                    self.tracker.save_episode_logs(log_path)

                break

    def _print_attack_detection_summary(self, attack_type, victim_idx, episode_stats):
        avg_reward = np.mean([s['reward'] for s in episode_stats])
        avg_length = np.mean([s['length'] for s in episode_stats])

        print(f"\n{'=' * 60}")
        print(f"[Attack Eval] Attack type: {attack_type}, victim: agent_{victim_idx}")
        print(f"[Attack Eval] Avg reward: {avg_reward:.3f}, avg length: {avg_length:.1f}")
        print(f"{'=' * 60}")

        print(f"[Tracker] Detection Summary (victim=agent_{victim_idx})")
        for w_idx, window in enumerate(self.tracker.windows):
            for th_idx, threshold in enumerate(self.tracker.thresholds):
                result = self.tracker.get_detection_result(w_idx, th_idx)
                t_detect = result['t_detect']

                observer_detections = t_detect[:, victim_idx].copy()
                observer_detections[victim_idx] = 1e6
                detected_count = (observer_detections < 1e5).sum()
                total_observers = self.num_agents - 1

                if detected_count > 0:
                    avg_t = observer_detections[observer_detections < 1e5].mean()
                    print(f"  W={window:>3}, Th={threshold:>6.1f}: "
                          f"{detected_count}/{total_observers} observers detected victim, "
                          f"avg_t={avg_t:.1f}")
        print(f"{'=' * 60}\n")

    runner_class.set_attacker = set_attacker
    runner_class.eval_with_attack = eval_with_attack
    runner_class._print_attack_detection_summary = _print_attack_detection_summary

    return runner_class



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



def standalone_eval_with_attack(
        phase1_model_dir,
        phase2_model_dir=None,
        attack_model_path=None,
        attack_type="none",
        victim_idx=0,
        num_agents=6,
        max_cycles=25,
        num_episodes=50,
        budget=0.35,
        config_path=None,
        device="cpu",
        scenario="simple_spread_v2",
):
    from harl.utils.attacked_env_harl import (
        create_pettingzoo_env, load_harl_actor, load_recl, load_tracker
    )
    from harl.algorithms.attackers import HARLAttacker

    obs_dim, action_dim = _get_env_dims(scenario)
    role_dim = 32
    env_args = _build_env_args(scenario, num_agents)

    print(f"\n{'=' * 60}")
    print(f"[Standalone Eval] Scenario: {scenario}")
    print(f"[Standalone Eval] obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[Standalone Eval] Loading models...")
    print(f"{'=' * 60}")

    actor = load_harl_actor(
        phase1_model_dir, config_path=config_path,
        obs_dim=obs_dim, action_dim=action_dim, device=device
    )

    tracker = None
    recl = None
    if phase2_model_dir is not None:
        recl = load_recl(
            phase1_model_dir, obs_dim=obs_dim, num_agents=num_agents,
            device=device, role_embedding_dim=role_dim,
        )

        tracker_dir = os.path.join(phase2_model_dir, "tracker")
        if os.path.exists(tracker_dir):
            tracker = load_tracker(
                tracker_dir, num_agents=num_agents, obs_dim=obs_dim,
                action_dim=action_dim, role_embedding_dim=role_dim, device=device,
            )

    attacker = HARLAttacker(
        attack_type=attack_type,
        model_path=attack_model_path if attack_type not in ["none", "random"] else None,
        budget=budget,
        action_dim=action_dim,
    )

    env = create_pettingzoo_env(env_args=env_args, N=num_agents, max_cycles=max_cycles)
    agent_names = env.possible_agents

    recurrent_n = 1
    rnn_hidden_size = 128

    print(f"\n[Eval] Starting evaluation: {num_episodes} episodes")
    print(f"[Eval] Attack type={attack_type}, victim=agent_{victim_idx}")

    all_rewards = []
    all_reward_per_agent = []

    for ep in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs_dict = reset_result[0]
        else:
            obs_dict = reset_result

        rnn_states = np.zeros((1, recurrent_n, rnn_hidden_size), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)

        if tracker is not None:
            tracker.init_hidden(1)
            tracker.reset_detection_stats()
        if recl is not None:
            recl.embedding_net.agent_embedding_net.rnn_hidden = None

        last_actions = np.zeros((1, num_agents, action_dim))
        ep_reward = 0.0
        ep_reward_per_agent = np.zeros(num_agents)
        terminated = False
        truncated = False
        t = 0

        while not (terminated or truncated):
            team_action = {}
            all_actions = []

            for agent_num, agent_name in enumerate(agent_names):
                agent_obs = obs_dict[agent_name].reshape(1, -1).astype(np.float32)

                with torch.no_grad():
                    action_t, _ = actor.act(
                        agent_obs, rnn_states, masks, None, deterministic=True)
                normal_action = action_t.cpu().numpy().flatten()

                if agent_num == victim_idx:
                    agent_action = attacker.get_attack_action(
                        agent_obs.flatten(), normal_action, deterministic=True
                    )
                else:
                    agent_action = normal_action

                agent_action = np.clip(agent_action, 0.0, 1.0)
                team_action[agent_name] = agent_action
                all_actions.append(agent_action)

            if tracker is not None and recl is not None:
                obs_array = np.stack([
                    obs_dict[a] for a in agent_names
                ]).reshape(1, num_agents, -1).astype(np.float32)
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
                scores = tracker.compute_scores(mu, std, act_t)
                tracker.update_detection_stats(scores[0])

                last_actions = act_array

            step_result = env.step(team_action)
            if len(step_result) == 5:
                obs_dict, rewards, terms, truncs, infos = step_result
            else:
                obs_dict, rewards, dones, infos = step_result
                terms = dones
                truncs = {a: False for a in dones}

            for agent_num, agent_name in enumerate(agent_names):
                ep_reward += rewards[agent_name]
                ep_reward_per_agent[agent_num] += rewards[agent_name]

            terminated = any(terms.values())
            truncated = any(truncs.values())
            t += 1

        all_rewards.append(ep_reward)
        all_reward_per_agent.append(ep_reward_per_agent.copy())

        if tracker is not None:
            tracker.end_episode(
                reward=ep_reward, episode_length=t, attacked_agent=victim_idx)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}: reward={ep_reward:.2f}")

    env.close()

    avg_reward = np.mean(all_rewards)
    avg_per_agent = np.mean(all_reward_per_agent, axis=0)

    print(f"\n{'=' * 60}")
    print(f"[Results] Scenario: {scenario}")
    print(f"[Results] Attack type: {attack_type}, victim: agent_{victim_idx}")
    print(f"[Results] Avg team reward: {avg_reward:.3f}")
    print(f"[Results] Per-agent avg reward: {avg_per_agent}")

    if tracker is not None:
        print(f"\n[Detection Results]")
        for w_idx, window in enumerate(tracker.windows):
            for th_idx, threshold in enumerate(tracker.thresholds):
                result = tracker.get_detection_result(w_idx, th_idx)
                t_det = result['t_detect']
                obs_det = t_det[:, victim_idx].copy()
                obs_det[victim_idx] = 1e6
                n_detected = (obs_det < 1e5).sum()
                if n_detected > 0:
                    avg_t = obs_det[obs_det < 1e5].mean()
                    print(f"  W={window:>3}, Th={threshold:>6.1f}: "
                          f"{n_detected}/{num_agents - 1} detected, avg_t={avg_t:.1f}")

    print(f"{'=' * 60}\n")

    return {
        'scenario': scenario,
        'attack_type': attack_type,
        'victim_idx': victim_idx,
        'avg_reward': avg_reward,
        'avg_reward_per_agent': avg_per_agent.tolist(),
        'all_rewards': all_rewards,
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation with attack")
    parser.add_argument("--phase1_model_dir", type=str, required=True)
    parser.add_argument("--phase2_model_dir", type=str, default=None)
    parser.add_argument("--attack_model_path", type=str, default=None)
    parser.add_argument("--attack_type", type=str, default="none",
                        choices=["ACT", "DYN", "grad", "random", "none"])
    parser.add_argument("--victim_idx", type=int, default=3)
    parser.add_argument("--scenario", type=str, default="simple_spread_v2",
                        help="Environment scenario: simple_spread_v2 or simple_tag_v2")
    parser.add_argument("--num_agents", type=int, default=6)
    parser.add_argument("--max_cycles", type=int, default=25,
                        help="Max steps per episode (simple_spread=25, simple_tag=50)")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--budget", type=float, default=0.35)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    results = standalone_eval_with_attack(
        phase1_model_dir=args.phase1_model_dir,
        phase2_model_dir=args.phase2_model_dir,
        attack_model_path=args.attack_model_path,
        attack_type=args.attack_type,
        victim_idx=args.victim_idx,
        num_agents=args.num_agents,
        max_cycles=args.max_cycles,
        num_episodes=args.num_episodes,
        budget=args.budget,
        config_path=args.config_path,
        device=args.device,
        scenario=args.scenario,
    )

    scenario_short = args.scenario.replace("simple_", "").replace("_v2", "").replace("_v3", "")
    output_path = f"eval_results_{scenario_short}_{args.attack_type}_victim{args.victim_idx}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved to: {output_path}")
