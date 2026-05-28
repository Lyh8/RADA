import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional



METHOD_DISPLAY = {
    "B0": "Normal (No Attack)",
    "B1": "No Defense (Attack Only)",
    "C4": "Weighted Geo. Median (C4)",
    "C3": "Geo. Median (C3)",
    "C2": "Trimmed Mean (C2)",
    "C1": "Coord. Median (C1)",
    "O1": "Oracle (O1)",
}


@dataclass
class TrajectoryData:
    method: str
    method_label: str
    victim_id: int
    num_agents: int
    positions: List[np.ndarray]
    detection_step: int = -1
    defense_active_mask: List[np.ndarray] = field(default_factory=list)
    total_reward: float = 0.0
    caught: bool = False

    def to_dict(self):
        return {
            "method": self.method,
            "method_label": self.method_label,
            "victim_id": self.victim_id,
            "num_agents": self.num_agents,
            "positions": [p.tolist() for p in self.positions],
            "detection_step": self.detection_step,
            "defense_active_mask": [d.tolist() for d in self.defense_active_mask],
            "total_reward": float(self.total_reward),
            "caught": bool(self.caught),
            "n_steps": len(self.positions),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            method=d["method"],
            method_label=d["method_label"],
            victim_id=d["victim_id"],
            num_agents=d["num_agents"],
            positions=[np.array(p) for p in d["positions"]],
            detection_step=d["detection_step"],
            defense_active_mask=[np.array(a, dtype=bool) for a in d["defense_active_mask"]],
            total_reward=d["total_reward"],
            caught=d.get("caught", False),
        )



def save_trajectories(trajectories, path):
    data = {"trajectories": [t.to_dict() for t in trajectories]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Trajectory] JSON saved → {path}")


def save_trajectories_npz(trajectories, path):
    arrays = {}
    for i, t in enumerate(trajectories):
        prefix = f"traj{i}"
        pos = np.array([p for p in t.positions])
        arrays[f"{prefix}_positions"] = pos
        arrays[f"{prefix}_defense_mask"] = np.array(
            [d for d in t.defense_active_mask]) if t.defense_active_mask else np.array([])
        arrays[f"{prefix}_meta"] = np.array([
            t.victim_id, t.num_agents, t.detection_step,
            t.total_reward, int(t.caught),
        ])
    np.savez_compressed(path, **arrays)
    print(f"[Trajectory] NPZ saved → {path}")


def load_trajectories(path):
    with open(path) as f:
        data = json.load(f)
    return [TrajectoryData.from_dict(d) for d in data["trajectories"]]



def plot_trajectory_comparison(trajectories, output_path,
                                figsize=(20, 7), prey_agent_ids=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection

    n_panels = len(trajectories)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    C_VICTIM_NORMAL  = '#1976D2'
    C_VICTIM_ATTACK  = '#D32F2F'
    C_VICTIM_DEFENSE = '#388E3C'
    C_PREY           = '#F57C00'
    C_DETECTION_STAR = '#FFD600'
    AGENT_CMAP = ['#78909C', '#90A4AE', '#B0BEC5', '#A1887F',
                  '#AED581', '#4DD0E1', '#CE93D8', '#FFAB91']

    if prey_agent_ids is None:
        N0 = trajectories[0].num_agents
        n_prey = max(1, N0 // 4)
        prey_agent_ids = list(range(N0 - n_prey, N0))

    prey_set = set(prey_agent_ids)

    all_x, all_y = [], []
    for traj in trajectories:
        for pos in traj.positions:
            all_x.extend(pos[:, 0].tolist())
            all_y.extend(pos[:, 1].tolist())
    margin = 0.15
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    x_pad = x_range * margin
    y_pad = y_range * margin

    for panel_idx, (traj, ax) in enumerate(zip(trajectories, axes)):
        v = traj.victim_id
        N = traj.num_agents
        T = len(traj.positions)
        det = traj.detection_step

        agent_xy = {}
        for a in range(N):
            xy_list = []
            for t in range(T):
                p = np.asarray(traj.positions[t], dtype=np.float32)
                if p.ndim == 2 and p.shape == (N, 2):
                    xy_list.append(p[a])
                elif p.ndim == 2 and p.shape == (2, N):
                    xy_list.append(p[:, a])
                elif p.ndim == 1 and len(p) == 2 * N:
                    xy_list.append(p[2*a:2*a+2])
                else:
                    xy_list.append(np.zeros(2))
            agent_xy[a] = np.array(xy_list)

        for a in range(N):
            if a == v:
                continue
            xy = agent_xy[a]
            is_prey = (a in prey_set)
            color = C_PREY if is_prey else AGENT_CMAP[a % len(AGENT_CMAP)]
            lw = 1.8 if is_prey else 1.0
            alpha = 0.75 if is_prey else 0.45
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, zorder=1)
            ax.scatter(xy[0, 0], xy[0, 1], color=color, s=25, marker='o',
                       zorder=3, alpha=alpha, edgecolors='none')
            ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=35, marker='^',
                       zorder=3, alpha=alpha, edgecolors='none')

        vxy = agent_xy[v]

        if traj.method == "B0":
            _plot_trajectory_line(ax, vxy, C_VICTIM_NORMAL, lw=3.0, zorder=2)
        elif traj.method == "B1":
            _plot_trajectory_line(ax, vxy, C_VICTIM_ATTACK, lw=3.0, zorder=2)
        else:
            if 0 < det < T - 1:
                _plot_trajectory_line(ax, vxy[:det + 1], C_VICTIM_ATTACK,
                                     lw=3.0, zorder=2)
                _plot_trajectory_line(ax, vxy[det:], C_VICTIM_DEFENSE,
                                     lw=3.0, zorder=2)
                ax.scatter(vxy[det, 0], vxy[det, 1], color=C_DETECTION_STAR,
                           s=280, marker='*', zorder=5, edgecolors='black',
                           linewidths=0.8)
            else:
                _plot_trajectory_line(ax, vxy, C_VICTIM_ATTACK, lw=3.0, zorder=2)

        ax.scatter(vxy[0, 0], vxy[0, 1], color='black', s=70,
                   marker='o', zorder=6, label='Start')
        ax.scatter(vxy[-1, 0], vxy[-1, 1], color='black', s=90,
                   marker='s', zorder=6, label='End')

        _add_direction_arrows(ax, vxy, traj, det, T,
                              C_VICTIM_NORMAL, C_VICTIM_ATTACK, C_VICTIM_DEFENSE)

        r_str = f"R = {traj.total_reward:.1f}"
        caught_str = ", Caught" if traj.caught else ""
        det_str = f", Det@t={det}" if det > 0 else ""
        ax.set_title(f"{traj.method_label}\n({r_str}{caught_str}{det_str})",
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("X", fontsize=11)
        if panel_idx == 0:
            ax.set_ylabel("Y", fontsize=11)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=C_VICTIM_NORMAL, lw=3, label='Victim (normal)'),
        Line2D([0], [0], color=C_VICTIM_ATTACK, lw=3, label='Victim (attacked)'),
        Line2D([0], [0], color=C_VICTIM_DEFENSE, lw=3, label='Victim (defended)'),
        Line2D([0], [0], marker='*', color=C_DETECTION_STAR, markersize=14,
               linestyle='None', markeredgecolor='black', label='Detection point'),
        Line2D([0], [0], color=C_PREY, lw=1.8, label='Prey'),
        Line2D([0], [0], color=AGENT_CMAP[0], lw=1.0, alpha=0.6, label='Other agents'),
        Line2D([0], [0], marker='o', color='black', markersize=6,
               linestyle='None', label='Start'),
        Line2D([0], [0], marker='s', color='black', markersize=6,
               linestyle='None', label='End'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[Trajectory] PDF saved → {output_path}")



def _plot_trajectory_line(ax, xy, color, lw=2.5, zorder=2):
    n = len(xy)
    if n < 2:
        return
    from matplotlib.collections import LineCollection
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    alphas = np.linspace(0.35, 1.0, len(segments))
    colors = []
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    for a in alphas:
        colors.append((r, g, b, a))
    lc = LineCollection(segments, colors=colors, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)


def _add_direction_arrows(ax, vxy, traj, det, T,
                           c_normal, c_attack, c_defense, interval=8):
    n = len(vxy)
    for i in range(interval, n - 1, interval):
        dx = vxy[i + 1, 0] - vxy[i, 0] if i + 1 < n else vxy[i, 0] - vxy[i - 1, 0]
        dy = vxy[i + 1, 1] - vxy[i, 1] if i + 1 < n else vxy[i, 1] - vxy[i - 1, 1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-6:
            continue
        if traj.method == "B0":
            c = c_normal
        elif traj.method == "B1":
            c = c_attack
        elif det > 0 and i >= det:
            c = c_defense
        else:
            c = c_attack
        ax.annotate('', xy=(vxy[i, 0] + dx * 0.3, vxy[i, 1] + dy * 0.3),
                     xytext=(vxy[i, 0], vxy[i, 1]),
                     arrowprops=dict(arrowstyle='->', color=c, lw=1.5),
                     zorder=4)



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-plot saved trajectory data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to trajectory_data.json")
    parser.add_argument("--output", type=str, default="trajectory_comparison.pdf")
    parser.add_argument("--figsize", type=float, nargs=2, default=[20, 7])
    parser.add_argument("--prey_ids", type=int, nargs="+", default=None)
    args = parser.parse_args()

    trajectories = load_trajectories(args.data)
    print(f"[Trajectory] Loaded {len(trajectories)} trajectories from {args.data}")
    for t in trajectories:
        print(f"  {t.method} ({t.method_label}): {len(t.positions)} steps, "
              f"R={t.total_reward:.1f}, det={t.detection_step}")

    plot_trajectory_comparison(
        trajectories, args.output,
        figsize=tuple(args.figsize),
        prey_agent_ids=args.prey_ids,
    )


if __name__ == "__main__":
    main()
