import pprint
import datetime
import os
import re
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import numpy as np
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from learners.decentralizedTracker import Tracker
from learners.single_agent_rdqn import RDQNAgent


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]

    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.timestamp_str = timestamp_str

    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{timestamp_str}"
    args.unique_token = unique_token

    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(abspath(__file__)), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, unique_token)
        logger.setup_tb(tb_exp_direc)

    logger.setup_sacred(_run.info)

    if args.evaluate:
        run_evaluation(args, logger)
    else:
        run_sequential(args=args, logger=logger)


    for t in threading.enumerate():
        if t.name != "MainThread":
            t.join(timeout=1)


def run_evaluation(args, logger):
    """Used only for evaluating an already-trained model"""
    logger.console_logger.info("Running in evaluation mode")
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    mac = mac_REGISTRY[args.mac](ReplayBuffer(scheme, groups, 1, 1, preprocess=preprocess).scheme, groups, args)

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    learner = le_REGISTRY[args.learner](mac, ReplayBuffer(scheme, groups, 1, 1).scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        logger.console_logger.info(f"Loading model from {args.checkpoint_path}")
        learner.load_models(args.checkpoint_path)

    evaluate_sequential(args, runner, logger)
    runner.close_env()


def evaluate_sequential(args, runner, logger):
    advagent = None
    if args.attack_active:
        if args.attack_type == "DAA" or args.attack_type == "OA":
            adv_obs_size = runner.env.get_obs_size()
            adv_actions_size = runner.env.get_total_actions()
            adv_args = copy.deepcopy(args)
            adv_args.batch_size = getattr(args, 'adv_batch_size', 32)
            adv_args.buffer_size = getattr(args, 'adv_buffer_size', 2000)

            if args.attack_type == "OA":
                print("[Info] Initializing OA Attacker: Lambda forced to 0.0 (Pure Damage)")
                lambda_init = [0.0] * (args.n_agents - 1)
            elif args.attack_type == "DAA":
                target_lambda = getattr(args, "adv_lambda", 0.025)
                print(f"[Info] Initializing DAA Attacker: Lambda = {target_lambda}")
                lambda_init = [target_lambda] * (args.n_agents - 1)

            advagent = RDQNAgent(adv_obs_size, adv_actions_size, adv_args, lambda_init=lambda_init)

            if getattr(args, "adv_load_adr", ""):
                logger.console_logger.info(f"Loading Adversary from {args.adv_load_adr}")
                advagent.load_model(args.adv_load_adr)

    trackers = Tracker(args.n_agents, runner.env.get_obs_size(), args)


    if args.checkpoint_path:
        logger.console_logger.info(f"[Info] Loading RECL & Clustering from checkpoint: {args.checkpoint_path}")
        trackers.load_recl_weights(args.checkpoint_path)
        trackers.load_clustering_info(args.checkpoint_path)
    else:
        print("[Warning] No checkpoint_path provided. RECL and Clustering info might be missing!")

    tracker_load_path = getattr(args, 'tracker_load_adr', "")
    if tracker_load_path:
        logger.console_logger.info(f"[Info] Loading Tracker model from: {tracker_load_path}")
        trackers.load_tracker_weights(tracker_load_path)
    else:
        print("[Warning] No tracker_load_adr provided. Using initialized Tracker weights.")

    start_episode = 0
    adv_load_path = getattr(args, "adv_load_adr", "")
    if adv_load_path and advagent is not None and not advagent.test_mode:
        match = re.search(r'ep_(\d+)', adv_load_path)
        if match:
            start_episode = int(match.group(1))
            logger.console_logger.info(
                f"[Resume] Resuming adversary training from episode {start_episode} "
                f"(epsilon={advagent.exploration_proba:.6f}, "
                f"training_steps={advagent.training_steps})"
            )
        else:
            logger.console_logger.info(
                f"[Resume] Could not parse episode number from {adv_load_path}, "
                f"starting from episode 0"
            )

    print(f"Starting Evaluation. Attack Active: {args.attack_active}, Type: {args.attack_type}, "
          f"Start Episode: {start_episode}, Total: {args.test_nepisode}")

    for episode in range(start_episode, args.test_nepisode):
        runner.run(advagent=advagent, tracker=trackers, test_mode=True)
        runner.t_env = episode + 1

        trackers.output_statistics(None, None, None, reset=True)
        if True:
            trackers.out_dict["attacked"].append(runner.adv_active)
            trackers.out_dict["battle_won"].append(runner.episode_result)
            trackers.out_dict["ep_length"].append(runner.episode_len)
            trackers.out_dict["t_start"].append(runner.attack_start_t + 1 if runner.adv_active else 1)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{args.test_nepisode} finished.")

        if advagent and not advagent.test_mode:
            if (episode + 1) % 500 == 0:
                won_recent = trackers.out_dict["battle_won"][-500:]
                win_rate = np.mean([1 if x == 1 else 0 for x in won_recent])
                print(f"  [ADV Train] ep={episode+1}  "
                      f"win_rate(last500)={win_rate:.3f}  "
                      f"epsilon={advagent.exploration_proba:.4f}  "
                      f"buffer_size={advagent.buffer.num_episodes}  "
                      f"training_steps={advagent.training_steps}")

            if (episode + 1) % 1000 == 0 or (episode + 1) == args.test_nepisode:
                map_name = args.env_args.get("map_name", "unknown")
                timestamp_str = getattr(args, "timestamp_str", "unknown_time")
                lambda_val = getattr(args, "adv_lambda", "0.0")

                folder_name = f"{timestamp_str}_{args.seed}_{lambda_val}"
                save_dir = os.path.join(args.local_results_path, "adv", map_name, args.attack_type, folder_name, f"ep_{episode + 1}")

                try:
                    advagent.save_model(save_dir)
                except Exception as e:
                    print(f"[Error] Failed to save adversary model: {e}")

    base_root = "test"
    map_name = args.env_args.get("map_name", "unknown_map")
    timestamp_str = getattr(args, "timestamp_str", "unknown_time")
    lambda_val = getattr(args, "adv_lambda", "0.0")

    if args.attack_active:
        folder_suffix = f"{timestamp_str}_{args.seed}_{lambda_val}"
    else:
        folder_suffix = f"{timestamp_str}_{args.seed}_normal"

    folder_name = folder_suffix
    save_dir = os.path.join(base_root, map_name, args.attack_type, f"ep_{args.test_nepisode}", folder_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    json_filename = "results.json"
    full_path = os.path.join(save_dir, json_filename)

    print(f"[Info] Saving detection results to: {full_path}")
    logger.console_logger.info(f"Saving detection results to: {full_path}")
    trackers.save_stats(full_path)



def run_sequential(args, logger):
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    qmix_model_path_for_tracker = ""

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if len(timesteps) > 0:
            if args.load_step == 0:
                timestep_to_load = max(timesteps)
            else:
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        else:
            model_path = args.checkpoint_path
            logger.console_logger.info(f"No numeric subdirectories found. Assuming {args.checkpoint_path} is the direct model path.")
            try:
                timestep_to_load = int(os.path.basename(args.checkpoint_path))
            except ValueError:
                timestep_to_load = 0

        qmix_model_path_for_tracker = model_path

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")

    while runner.t_env <= args.t_max:
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            last_test_T = runner.t_env
            for _ in range(max(1, args.test_nepisode // runner.batch_size)):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env

            map_name = args.env_args.get("map_name", "unknown")
            timestamp_str = getattr(args, "timestamp_str", "unknown_time")
            folder_name = f"{args.name}_{timestamp_str}_{args.seed}"

            save_path = os.path.join(args.local_results_path, "models", map_name, folder_name, str(runner.t_env))

            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")
            learner.save_models(save_path)

            qmix_model_path_for_tracker = save_path

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    logger.console_logger.info("Finished QMIX+RECL Training")

    if getattr(args, "tracker_train", False):
        logger.console_logger.info("--- Starting Stage 2: Tracker Training ---")

        args.episode_limit = runner.episode_limit
        if hasattr(runner, 'env'):
            obs_size = runner.env.get_obs_size()
        else:
            obs_size = runner.env_info["obs_shape"]
        tracker = Tracker(args.n_agents, obs_size, args)

        recl_load_path = qmix_model_path_for_tracker
        if not recl_load_path:
            pass

        if os.path.exists(recl_load_path):
            logger.console_logger.info(f"Loading RECL and Clustering from {recl_load_path} for tracker training.")
            tracker.load_recl_weights(recl_load_path)
            tracker.load_clustering_info(recl_load_path)
        else:
            logger.console_logger.warning(f"Could not find model path {recl_load_path} to load RECL net. Tracker might not work.")
        tracker_resume_path = getattr(args, "tracker_load_adr", "")
        if tracker_resume_path and os.path.exists(tracker_resume_path):
            logger.console_logger.info(f"[Resume] Resuming Tracker training from: {tracker_resume_path}")
            tracker.load_tracker_weights(tracker_resume_path)
        logger.console_logger.info(f"Collecting data for {args.tracker_train_episodes} episodes...")

        runner.t_env = 0

        episodes_per_run = getattr(runner, 'batch_size', 1)
        total_episodes_collected = 0
        run_count = 0

        import time as _time
        _t_start = _time.time()

        while total_episodes_collected < args.tracker_train_episodes:
            runner.run(tracker=tracker, test_mode=True)
            total_episodes_collected += episodes_per_run
            run_count += 1

            if tracker.buffer.num_episodes >= args.batch_size:
                for _ in range(episodes_per_run):
                    tracker.train(logger, total_episodes_collected)

            if total_episodes_collected % 100 < episodes_per_run:
                elapsed = _time.time() - _t_start
                eps_per_sec = total_episodes_collected / max(elapsed, 1e-6)
                remaining = (args.tracker_train_episodes - total_episodes_collected) / max(eps_per_sec, 1e-6)
                logger.console_logger.info(
                    f"Tracker training: {total_episodes_collected}/{args.tracker_train_episodes} episodes "
                    f"({eps_per_sec:.1f} ep/s, ETA {remaining/60:.1f} min)"
                )

            if total_episodes_collected % 10000 < episodes_per_run:
                map_name = args.env_args.get("map_name", "unknown")
                timestamp_str = getattr(args, "timestamp_str", "unknown_time")
                folder_name = f"{timestamp_str}_{args.seed}"

                save_path = os.path.join(args.local_results_path, "tracker", map_name, folder_name, f"ep_{total_episodes_collected}")

                logger.console_logger.info(f"Saving intermediate tracker to {save_path}")
                tracker.save_model(save_path)


        final_ep = args.tracker_train_episodes
        map_name = args.env_args.get("map_name", "unknown")
        timestamp_str = getattr(args, "timestamp_str", "unknown_time")
        folder_name = f"{timestamp_str}_{args.seed}"

        save_path = os.path.join(args.local_results_path, "tracker", map_name, folder_name, f"ep_{final_ep}")

        logger.console_logger.info(f"Saving trained tracker to {save_path}")
        tracker.save_model(save_path)

        logger.console_logger.info("Finished Tracker Training")


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    return config
