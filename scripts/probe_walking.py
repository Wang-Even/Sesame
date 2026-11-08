#!/usr/bin/env python3
"""GPU integration probe for contact, resets, and bounded actions before PPO."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
args = parser.parse_args()
runtime = ROOT / ".runtime/walking_probe"
for name, folder in {"XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data", "CUDA_CACHE_PATH": "cuda_cache"}.items():
    path = runtime / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
report_path = ROOT / "reports/walking_probe.json"
report = {"passed": False, "scope": "GPU standing contact and action/reset integration; no learned policy"}
app = env = None
code = 1
try:
    import torch
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    app = AppLauncher(headless=True, device="cuda:0", kit_args=f"--portable-root {runtime / 'kit'} --/log/file={runtime / 'kit.log'} --/app/settings/persistent=false --/renderer/multiGpu/enabled=false").app
    import warp as wp
    wp.config.kernel_cache_dir = str(runtime / "warp_cache")
    from sesame_rl.environment import SesameWalkEnv, make_env_cfg
    env = SesameWalkEnv(make_env_cfg(num_envs=args.num_envs, seed=42, device="cuda:0"))
    report["contact_setup"] = env.contact_setup_metadata
    obs, _ = env.reset(seed=42)
    actions = torch.zeros((args.num_envs, 8), device="cuda:0")
    falls = torch.zeros(args.num_envs, device="cuda:0", dtype=torch.int64)
    timeouts = torch.zeros_like(falls)
    z_min = torch.full((args.num_envs,), float("inf"), device="cuda:0")
    z_max = torch.full((args.num_envs,), -float("inf"), device="cuda:0")
    upright_min = torch.ones(args.num_envs, device="cuda:0")
    reward_sum = torch.zeros(args.num_envs, device="cuda:0")
    initial_root = env.robot.data.root_pos_w.clone()
    snapshots = []
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        state = env.robot.data.root_state_w
        assert state.is_cuda and reward.is_cuda and bool(torch.isfinite(state).all())
        assert bool(torch.isfinite(reward).all())
        for value in obs.values():
            assert value.is_cuda and bool(torch.isfinite(value).all())
        falls += terminated.long()
        timeouts += truncated.long()
        heights = env.robot.data.root_pos_w[:, 2]
        upright = -env.robot.data.projected_gravity_b[:, 2]
        z_min = torch.minimum(z_min, heights)
        z_max = torch.maximum(z_max, heights)
        upright_min = torch.minimum(upright_min, upright)
        reward_sum += reward
        if step in {0, 24, 49, 99, 249, args.steps - 2, args.steps - 1}:
            snapshots.append({"step":step+1,"base_xyz_first_env":env.robot.data.root_pos_w[0].tolist(),"upright_first_env":upright[0].item(),"falls":falls.sum().item(),"timeouts":timeouts.sum().item()})
    report["standing"] = {"steps":args.steps,"num_envs":args.num_envs,"fall_terminations":falls.sum().item(),"timeout_terminations":timeouts.sum().item(),"min_height_m":z_min.min().item(),"max_height_m":z_max.max().item(),"min_upright":upright_min.min().item(),"mean_return":reward_sum.mean().item(),"snapshots":snapshots}
    assert falls.sum().item() == 0, "Zero-action nominal stand falls; inspect contacts/actuators before training"
    env.reset(seed=43)
    random_dones = 0
    for _ in range(32):
        obs, reward, terminated, truncated, info = env.step(torch.rand_like(actions) * 2 - 1)
        assert bool(torch.isfinite(env.robot.data.root_state_w).all())
        assert bool(torch.isfinite(reward).all())
        random_dones += terminated.sum().item()
    report["random_actions"] = {"steps":32,"finite_states":True,"fall_terminations":random_dones}
    report["joint_order_isaac"] = env.robot.joint_names
    report["observation_shape"] = {key:list(value.shape) for key,value in obs.items()}
    report["passed"] = True
    code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    traceback.print_exc()
finally:
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("WALKING_PROBE=" + json.dumps(report), flush=True)
    if app is not None:
        def forced_exit():
            report["shutdown"] = {"completed":False,"forced_process_exit":True,"timeout_seconds":15}
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            os._exit(code)
        timer = threading.Timer(15, forced_exit)
        timer.daemon = True
        timer.start()
        if env is not None:
            env.close()
        app.close(wait_for_replicator=False)
        timer.cancel()
raise SystemExit(code)
