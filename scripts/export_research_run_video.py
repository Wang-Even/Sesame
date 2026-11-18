"""Evaluate one saved research policy on all held-out clips and render CUDA states."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_research_evaluation_videos import (
    setup, read, write, digest, analyze, bounds_for, panel, RecordedStateRenderer,
)


def interval_velocities(torch, trace, episode, values):
    """Use the same interval measurement for policy and reference displays."""
    for field, key in (("root_pos_w", "velocity"), ("reference_pos_w", "ref_velocity")):
        pos = trace[field][:, episode]
        ends = torch.arange(len(pos), device=pos.device)
        starts = (ends - 10).clamp_min(0)
        elapsed = ((ends - starts) * .02).clamp_min(.02)
        values[key] = ((pos[ends] - pos[starts]) / elapsed[:, None]).cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", choices=("best.pt", "final.pt"), default="best.pt")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch = setup()
    import numpy as np
    import imageio.v2 as imageio
    from sesame_research.evaluation import Evaluator
    from sesame_research.reward import REWARD_VERSION
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor, build_world_model

    saved = read(a.run / "protocol.json")
    args = saved["args"]
    test = Path(args["test_file"])
    checkpoint = a.run / a.checkpoint
    steps = args["eval_steps"]
    evaluator = Evaluator(test, 91042, args["physics_dt"],
                          diagnostic_only_metrics=args.get("diagnostic_only_metrics", []))
    renderer = None
    try:
        cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
        torch.manual_seed(args["seed"])
        wm = build_world_model(cfg, evaluator.env, "cuda:0")
        actor = build_actor(cfg, wm, "cuda:0")
        bundle = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        actor.load_state_dict(bundle["actor"], strict=True)
        actor.eval()
        store = {}

        def capture(t, raw, alive, reference):
            d = raw.scene["robot"].data
            values = {
                "qpos": raw.sim.data.qpos, "qvel": raw.sim.data.qvel,
                "root_pos_w": d.root_link_pos_w - raw.scene.env_origins,
                "root_quat_w": d.root_link_quat_w,
                "root_lin_vel_w": d.root_link_lin_vel_w,
                "reference_pos_w": reference[0][:, 0],
                "reference_lin_vel_w": reference[2][:, 0], "alive": alive,
            }
            for key, value in values.items():
                store.setdefault(key, []).append(value.clone())

        metrics = evaluator.run(actor, wm.representation, wm.sim_context,
                                cfg.history_length, steps, trace_callback=capture)
        trace = {key: torch.stack(value) for key, value in store.items()}
        if not all(bool(torch.isfinite(value).all()) for value in trace.values()):
            raise ValueError("Nonfinite recorded states")
        np.savez_compressed(a.output / "policy_trace.npz",
                            **{key: value.cpu().numpy() for key, value in trace.items()}, fps=50)
        write(a.output / "evaluation.json", metrics)
        summaries, values = [], []
        for episode in range(evaluator.env.num_envs):
            value, summary = analyze(trace, episode, reference=trace, state_limit=steps)
            # Interval-average speed avoids control-endpoint velocity aliasing.
            interval_velocities(torch, trace, episode, value)
            summary["clip"] = read(test / "motion_names.json")[episode]
            values.append(value)
            summaries.append(summary)
        protocol = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
            "checkpoint_real_steps": bundle["steps"], "test_source": str(test),
            "seed": 91042, "physics_dt_s": args["physics_dt"],
            "steps": steps, "statistics_interval_s": [0., (steps - 1) * .02],
            "selection": "All test clips in stored order; no best-of-N selection",
            "video": "Actual policy inference in CUDA simulation, recorded and replayed with GPU FK/EGL; 1x speed",
            "velocity_display": "Policy and reference position difference over preceding 0.2 s; raw instantaneous samples retained",
            "heading": "World-frame body yaw and net movement direction reported separately",
            "evaluation": "Fresh independent replay; original training test results are not overwritten",
            "evaluation_reward_version": REWARD_VERSION,
            "checkpoint_training_reward_version": saved.get("reward_version", "legacy_all_body_tracking_with_joint_limit"),
            "trace_sha256": digest(a.output / "policy_trace.npz"),
            "episode_statistics": summaries,
        }
        write(a.output / "protocol.json", protocol)
        (a.output / "export_source.py").write_bytes(Path(__file__).read_bytes())
        renderer = RecordedStateRenderer(evaluator.env.unwrapped)
        video = a.output / "best_policy_all4.mp4"
        with imageio.get_writer(video, fps=25, codec="libx264", quality=8,
                               macro_block_size=16,
                               ffmpeg_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
            for episode, value in enumerate(values):
                bounds = bounds_for(value)
                for t in range(0, steps, 2):
                    valid = [j for j in range(t + 1) if value["alive"][j]]
                    shown = t if value["alive"][t] else valid[-1]
                    frame = renderer.frame(trace, shown, episode, value)
                    canvas = panel(frame, value, t,
                                   f"SuperMimic · 测试片段 {episode + 1}/{len(values)}",
                                   "训练后策略实际推理 · 独立测试集", bounds)
                    writer.append_data(np.asarray(canvas))
                    if episode == 0 and t == steps // 2:
                        canvas.save(a.output / "preview.png")
        write(a.output / "status.json", {"status": "complete", "video": str(video),
                                         "video_sha256": digest(video)})
        print(json.dumps({"evaluation": metrics, "episodes": summaries,
                          "video": str(video)}, ensure_ascii=False), flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        evaluator.close()


if __name__ == "__main__":
    main()
