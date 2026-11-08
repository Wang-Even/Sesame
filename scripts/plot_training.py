#!/usr/bin/env python3
"""Plot saved training/evaluation metadata; does not load or execute a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
run = args.run_dir.resolve()
rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines() if line.strip()]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
specs = [
    ("Episode return", "mean_episode_return", None),
    ("Forward displacement per episode (m)", None, "Episode/distance_x_m"),
    ("Episode duration (s)", None, "Episode/duration_s"),
    ("Fall fraction", None, "Episode/fall_fraction"),
]
for ax, (label, top_key, episode_key) in zip(axes.flat, specs):
    points = [(r["environment_transitions"] / 1e6,
               r.get(top_key) if top_key else r.get("episode_metrics", {}).get(episode_key)) for r in rows]
    points = [(x, y) for x, y in points if y is not None]
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points], color="#166b99", linewidth=1.4)
    ax.set(xlabel="Environment transitions (millions)", ylabel=label)
    ax.grid(alpha=0.2)
    if label == "Fall fraction":
        ax.set_ylim(-0.03, 1.03)
fig.suptitle("Sesame PPO baseline — local Isaac task")
fig.savefig(run / "learning_curve.png", dpi=180)
fig.savefig(run / "learning_curve.pdf")
plt.close(fig)

trajectories = {}
for name in ("ppo", "zero_residual"):
    path = run / f"eval_trajectory_{name}.json"
    if path.exists():
        trajectories[name] = json.loads(path.read_text())
if trajectories:
    # Trajectory schema is deliberately checked instead of guessing field semantics.
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plotted = False
    for label, data in trajectories.items():
        frames = data.get("samples", [])
        if not isinstance(frames, list) or not frames:
            continue
        state_key = "root_state_w"
        if state_key not in frames[0]:
            continue
        start_x = frames[0][state_key][0]
        times = [frame.get("time_s", frame.get("t_s", frame.get("step", 0) * 0.02)) for frame in frames]
        distance = [frame[state_key][0] - start_x for frame in frames]
        ax.plot(times, distance, label=label)
        plotted = True
    if plotted:
        ax.set(xlabel="Time (s)", ylabel="Forward displacement (m)", title="First held-out episode, paired initial state")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(run / "evaluation_trajectory.png", dpi=180)
    plt.close(fig)
print(str(run / "learning_curve.png"))
