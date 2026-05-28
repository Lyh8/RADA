import numpy as np
import torch
from dataclasses import dataclass


METHOD_NONE = "none"
METHOD_NO_DEFENSE = "no_defense"
METHOD_ZERO = "zero"
METHOD_RANDOM = "random"
METHOD_TRIMMED_MEAN = "trimmed_mean"
METHOD_COORD_MEDIAN = "coord_median"
METHOD_GEO_MEDIAN = "geo_median"
METHOD_W_GEO_MEDIAN = "weighted_geo_median"
METHOD_ORACLE = "oracle"

SCHEME_TO_METHOD = {
    "B0": METHOD_NONE,
    "B1": METHOD_NO_DEFENSE,
    "B2": METHOD_ZERO,
    "B3": METHOD_RANDOM,
    "C1": METHOD_COORD_MEDIAN,
    "C2": METHOD_TRIMMED_MEAN,
    "C3": METHOD_GEO_MEDIAN,
    "C4": METHOD_W_GEO_MEDIAN,
    "O1": METHOD_ORACLE,
}

ALL_METHODS = list(SCHEME_TO_METHOD.keys())


@dataclass
class DefenseConfig:

    defense_method: str = "C4"

    detection_window: int = 5
    detection_threshold: float = -11.0

    weiszfeld_max_iter: int = 20
    weiszfeld_tol: float = 1e-5
    trim_m: int = 1

    w_same: float = 3.0
    w_diff: float = 1.0
    use_conf_weight: bool = True
    conf_eps: float = 1e-6

    action_low: float = 0.0
    action_high: float = 1.0
    disable_clip: bool = False

    victim_agent_id: int = 3

    eval_episodes: int = 500

    forced_delay: int = -1

    @property
    def internal_method(self):
        return SCHEME_TO_METHOD.get(self.defense_method, self.defense_method)



def trimmed_mean_np(points, m=1):
    K, d = points.shape
    if K <= 2 * m:
        return points.mean(axis=0)

    sorted_pts = np.sort(points, axis=0)
    trimmed = sorted_pts[m:K - m]
    return trimmed.mean(axis=0)


def geometric_median_np(points, weights=None, max_iter=20, tol=1e-5):
    K, d = points.shape
    if K == 0:
        return np.zeros(d, dtype=np.float32)
    if K == 1:
        return points[0].copy()

    if weights is None:
        weights = np.ones(K, dtype=np.float64)

    w_sum = weights.sum()
    if w_sum < 1e-12:
        weights = np.ones(K, dtype=np.float64)
        w_sum = float(K)

    w_norm = weights / w_sum
    y = (w_norm[:, None] * points).sum(axis=0)

    for _ in range(max_iter):
        dists = np.linalg.norm(points - y[None, :], axis=1)
        dists = np.maximum(dists, 1e-8)

        inv_d = weights / dists
        y_new = (inv_d[:, None] * points).sum(axis=0) / inv_d.sum()

        if np.linalg.norm(y_new - y) < tol:
            break
        y = y_new

    return y.astype(np.float32)



class ContinuousDefenseModule:

    def __init__(self, config, num_agents, action_dim, device=torch.device("cpu")):
        self.config = config
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device

        self.method = config.internal_method

        self.defense_locked = None
        self.detection_step = None
        self.thread_step = None

        self.score_ema = None
        self.score_count = 0

        self._ep_action_mse_sum = None
        self._ep_cosine_sum = None
        self._ep_defense_steps = None
        self._ep_total_steps = None

        self.all_rewards = []
        self.all_action_mse = []
        self.all_cosine_sim = []
        self.all_defense_ratios = []
        self.all_catch = []
        self.all_detection_steps = []

        self._pending_det_step = {}

        print(f"[Defense] Init: method={config.defense_method} "
              f"({self.method}), N={num_agents}, d={action_dim}")


    def reset_episode(self, n_threads=1):
        N = self.num_agents
        self.n_threads = n_threads
        self.defense_locked = np.zeros((n_threads, N), dtype=bool)
        self.detection_step = np.full((n_threads, N), -1, dtype=int)
        self.thread_step = np.zeros(n_threads, dtype=int)
        self.score_ema = np.zeros((n_threads, N, N))
        self.score_count = 0

        self._ep_action_mse_sum = np.zeros(n_threads)
        self._ep_cosine_sum = np.zeros(n_threads)
        self._ep_defense_steps = np.zeros(n_threads, dtype=int)
        self._ep_total_steps = np.zeros(n_threads, dtype=int)

    @property
    def current_step(self):
        if self.thread_step is not None and len(self.thread_step) > 0:
            return int(self.thread_step.max())
        return 0

    def reset_thread(self, tid):
        N = self.num_agents
        self.defense_locked[tid] = False
        self.detection_step[tid] = -1
        self.thread_step[tid] = 0
        self.score_ema[tid] = 0.0

        for j in range(N):
            self._pending_det_step.pop((tid, j), None)

        self._ep_action_mse_sum[tid] = 0.0
        self._ep_cosine_sum[tid] = 0.0
        self._ep_defense_steps[tid] = 0
        self._ep_total_steps[tid] = 0

    def end_episode(self, tid, reward, caught=False):
        total = max(self._ep_total_steps[tid], 1)
        d_steps = self._ep_defense_steps[tid]
        v = self.config.victim_agent_id

        self.all_rewards.append(float(reward))
        self.all_catch.append(bool(caught))
        self.all_defense_ratios.append(float(d_steps) / float(total))

        key = (tid, v)
        det = self._pending_det_step.pop(key, -1)
        self.all_detection_steps.append(int(det))

        if d_steps > 0:
            self.all_action_mse.append(
                float(self._ep_action_mse_sum[tid] / d_steps))
            self.all_cosine_sim.append(
                float(self._ep_cosine_sum[tid] / d_steps))

        self.reset_thread(tid)


    def _record_detection(self, tid, agent_id, step):
        key = (tid, agent_id)
        if key not in self._pending_det_step:
            self._pending_det_step[key] = int(step)

    def update_detection(self, raw_scores):
        n_threads = raw_scores.shape[0]
        N = self.num_agents
        alpha = 2.0 / (self.config.detection_window + 1)
        self.score_count += 1

        for b in range(n_threads):
            self.thread_step[b] += 1

        if self.config.forced_delay >= 0:
            v = self.config.victim_agent_id
            for b in range(n_threads):
                if self.thread_step[b] >= self.config.forced_delay and not self.defense_locked[b, v]:
                    valid = []
                    for i in range(N):
                        if i == v:
                            continue
                        s = raw_scores[b, i, v]
                        if not np.isnan(s) and s > -400:
                            valid.append(s)
                    if len(valid) > 0 and np.mean(valid) < self.config.detection_threshold:
                        self.defense_locked[b, v] = True
                        self.detection_step[b, v] = self.thread_step[b]
                        self._record_detection(b, v, self.thread_step[b])
            return

        v = self.config.victim_agent_id
        for b in range(n_threads):
            if self.defense_locked[b, v]:
                continue

            for i in range(N):
                if i == v:
                    continue
                s = raw_scores[b, i, v]
                if np.isnan(s):
                    continue
                if s < -400:
                    self.defense_locked[b, v] = True
                    self.detection_step[b, v] = self.thread_step[b]
                    self._record_detection(b, v, self.thread_step[b])
                    break
                self.score_ema[b, i, v] = alpha * s + (1 - alpha) * self.score_ema[b, i, v]

            if self.defense_locked[b, v]:
                continue

            valid = []
            for i in range(N):
                if i == v:
                    continue
                if not np.isnan(raw_scores[b, i, v]) and raw_scores[b, i, v] > -400:
                    valid.append(self.score_ema[b, i, v])

            if len(valid) == 0:
                continue

            agg = np.mean(valid)
            if agg < self.config.detection_threshold:
                self.defense_locked[b, v] = True
                self.detection_step[b, v] = self.thread_step[b]
                self._record_detection(b, v, self.thread_step[b])


    def get_corrected_actions(self, actions, mu_np, std_np, labels_np,
                              oracle_actions=None):
        n_threads, N, d = actions.shape
        corrected = actions.copy()
        applied = np.zeros((n_threads, N), dtype=bool)

        method = self.method

        if method in (METHOD_NONE, METHOD_NO_DEFENSE):
            for b in range(n_threads):
                self._ep_total_steps[b] += 1
            return corrected, applied

        for b in range(n_threads):
            self._ep_total_steps[b] += 1

            for v in range(N):
                if not self.defense_locked[b, v]:
                    continue

                applied[b, v] = True
                self._ep_defense_steps[b] += 1

                normal_ids = [
                    i for i in range(N)
                    if i != v and not self.defense_locked[b, i]
                ]

                if method == METHOD_ZERO:
                    corrected[b, v] = np.zeros(d, dtype=np.float32)

                elif method == METHOD_RANDOM:
                    corrected[b, v] = np.random.uniform(
                        self.config.action_low,
                        self.config.action_high,
                        size=d,
                    ).astype(np.float32)

                elif method == METHOD_ORACLE:
                    if oracle_actions is not None:
                        corrected[b, v] = oracle_actions[b, v]

                elif method in (METHOD_COORD_MEDIAN, METHOD_TRIMMED_MEAN, METHOD_GEO_MEDIAN, METHOD_W_GEO_MEDIAN):
                    if len(normal_ids) == 0:
                        corrected[b, v] = np.zeros(d, dtype=np.float32)
                        continue

                    points = np.stack([mu_np[b, i, v] for i in normal_ids])

                    if method == METHOD_COORD_MEDIAN:
                        result = np.median(points, axis=0)

                    elif method == METHOD_TRIMMED_MEAN:
                        result = trimmed_mean_np(points, m=self.config.trim_m)

                    elif method == METHOD_GEO_MEDIAN:
                        result = geometric_median_np(
                            points,
                            max_iter=self.config.weiszfeld_max_iter,
                            tol=self.config.weiszfeld_tol,
                        )
                    elif method == METHOD_W_GEO_MEDIAN:
                        weights = self._compute_weights(
                            normal_ids, v,
                            std_np[b],
                            labels_np[b] if labels_np is not None else None,
                        )
                        result = geometric_median_np(
                            points,
                            weights=weights,
                            max_iter=self.config.weiszfeld_max_iter,
                            tol=self.config.weiszfeld_tol,
                        )

                    if not self.config.disable_clip:
                        result = np.clip(result, self.config.action_low, self.config.action_high)
                    corrected[b, v] = result.astype(np.float32)

                else:
                    raise ValueError(f"Unknown defense method: {method}")

        return corrected, applied

    def _compute_weights(self, normal_ids, victim_id, std_matrix, labels):
        K = len(normal_ids)
        weights = np.ones(K, dtype=np.float64)

        if labels is not None:
            v_label = labels[victim_id]
            for k, i in enumerate(normal_ids):
                if labels[i] == v_label:
                    weights[k] *= self.config.w_same
                else:
                    weights[k] *= self.config.w_diff

        if self.config.use_conf_weight:
            for k, i in enumerate(normal_ids):
                std_iv = std_matrix[i, victim_id]
                trace_sigma = np.sum(std_iv ** 2)
                weights[k] *= 1.0 / (trace_sigma + self.config.conf_eps)

        return weights




    def record_correction(self, tid, corrected_action, oracle_action):
        if oracle_action is None or corrected_action is None:
            return

        mse = np.sum((corrected_action - oracle_action) ** 2)
        self._ep_action_mse_sum[tid] += mse

        nc = np.linalg.norm(corrected_action)
        no = np.linalg.norm(oracle_action)
        if nc > 1e-8 and no > 1e-8:
            cos = np.dot(corrected_action, oracle_action) / (nc * no)
        else:
            cos = 0.0
        self._ep_cosine_sum[tid] += cos


    def get_summary(self):
        n = len(self.all_rewards)
        if n == 0:
            return {"n_episodes": 0}

        rewards = np.array(self.all_rewards)
        catches = np.array(self.all_catch, dtype=float)
        ratios = np.array(self.all_defense_ratios)
        det_steps = np.array(self.all_detection_steps)

        valid_mse = np.array(self.all_action_mse) if self.all_action_mse else np.array([])
        valid_cos = np.array(self.all_cosine_sim) if self.all_cosine_sim else np.array([])
        valid_det = det_steps[det_steps >= 0]

        summary = {
            "n_episodes": n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "catch_rate": float(np.mean(catches)),
            "mean_defense_ratio": float(np.mean(ratios)),
        }

        if len(valid_mse) > 0:
            summary["mean_action_mse"] = float(np.mean(valid_mse))
        if len(valid_cos) > 0:
            summary["mean_cosine_sim"] = float(np.mean(valid_cos))
        if len(valid_det) > 0:
            summary["mean_detection_step"] = float(np.mean(valid_det))
            summary["detection_rate"] = float(len(valid_det)) / n

        return summary

    @staticmethod
    def compute_recovery_rate(R_defense, R_attack, R_normal):
        denom = R_normal - R_attack
        if abs(denom) < 1e-10:
            return 0.0
        return float((R_defense - R_attack) / denom * 100.0)



class ThresholdCalibrator:

    def __init__(self, num_agents, ema_window=5):
        self.num_agents = num_agents
        self.ema_window = ema_window
        self.alpha = 2.0 / (ema_window + 1)

        self.score_ema = None

        self.all_agg_scores = []

        self.all_raw_scores = []

        self.total_steps = 0
        self.total_episodes = 0

    def reset_episode(self, n_threads):
        N = self.num_agents
        self.score_ema = np.zeros((n_threads, N, N))

    def collect_step(self, raw_scores):
        n_threads = raw_scores.shape[0]
        N = self.num_agents
        alpha = self.alpha
        self.total_steps += 1

        for b in range(n_threads):
            for j in range(N):
                for i in range(N):
                    if i == j:
                        continue
                    s = raw_scores[b, i, j]
                    if np.isnan(s) or s < -400:
                        continue
                    self.score_ema[b, i, j] = (
                        alpha * s + (1 - alpha) * self.score_ema[b, i, j]
                    )

                valid = []
                for i in range(N):
                    if i == j:
                        continue
                    s = raw_scores[b, i, j]
                    if not np.isnan(s) and s > -400:
                        valid.append(self.score_ema[b, i, j])

                if len(valid) > 0:
                    agg = float(np.mean(valid))
                    self.all_agg_scores.append(agg)

        self.all_raw_scores.append(raw_scores.copy())

    def end_episode(self):
        self.total_episodes += 1

    def compute_threshold(self, target_fpr=0.05):
        if len(self.all_agg_scores) == 0:
            print("[Calibrator] Warning: No scores collected, returning default -11.0")
            return -11.0

        scores = np.array(self.all_agg_scores)
        threshold = float(np.percentile(scores, target_fpr * 100))

        return threshold

    def get_diagnostics(self, target_fprs=None):
        if target_fprs is None:
            target_fprs = [0.01, 0.02, 0.05, 0.10, 0.20]

        if len(self.all_agg_scores) == 0:
            return {"error": "no scores collected"}

        scores = np.array(self.all_agg_scores)

        diag = {
            "n_samples": len(scores),
            "n_episodes": self.total_episodes,
            "n_steps": self.total_steps,
            "ema_window": self.ema_window,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "percentiles": {},
            "thresholds": {},
        }

        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            diag["percentiles"][f"p{p}"] = float(np.percentile(scores, p))

        for fpr in target_fprs:
            eta = float(np.percentile(scores, fpr * 100))
            diag["thresholds"][f"fpr_{fpr:.2f}"] = eta

        return diag

    def print_report(self, target_fprs=None):
        diag = self.get_diagnostics(target_fprs)

        if "error" in diag:
            print(f"[Calibrator] {diag['error']}")
            return diag

        print(f"\n{'=' * 65}")
        print(f"  Threshold Calibration Report")
        print(f"{'=' * 65}")
        print(f"  EMA window:  {diag['ema_window']}")
        print(f"  Episodes:    {diag['n_episodes']}")
        print(f"  Samples:     {diag['n_samples']}")
        print(f"  Score dist:  mean={diag['mean']:.3f}, "
              f"std={diag['std']:.3f}, "
              f"range=[{diag['min']:.3f}, {diag['max']:.3f}]")
        print(f"\n  Percentiles:")
        for k, v in diag["percentiles"].items():
            print(f"    {k:>5s}: {v:.4f}")
        print(f"\n  Recommended thresholds (η*):")
        for k, v in diag["thresholds"].items():
            fpr_val = k.replace("fpr_", "")
            print(f"    FPR={fpr_val}:  η* = {v:.4f}")
        print(f"{'=' * 65}\n")

        return diag
