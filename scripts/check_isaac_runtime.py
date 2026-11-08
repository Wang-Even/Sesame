#!/usr/bin/env python3
"""Check local Isaac Lab GPU physics and CUDA optimization without robot downloads.

Run through env_isaaclab's Conda activation hooks; see isaac_environment.json.
This checks the generic simulation stack, not a Sesame robot or training task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num-objects", type=int, default=16)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--report", type=Path, default=ROOT / "reports/isaac_runtime_smoke.json")
args = parser.parse_args()
if not args.device.startswith("cuda"):
    parser.error("Only CUDA devices are permitted; CPU fallback is disabled.")
if args.steps <= 0 or args.num_objects <= 0:
    parser.error("steps and num-objects must be positive")

runtime = ROOT / ".runtime"
for key, relative in {
    "XDG_CACHE_HOME": "cache",
    "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data",
    "CUDA_CACHE_PATH": "cuda_cache",
}.items():
    directory = runtime / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
args.report.parent.mkdir(parents=True, exist_ok=True)
result = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "requested_device": args.device,
    "scope": "generic Isaac Lab GPU rigid-body simulation and synthetic CUDA optimizer smoke",
    "sesame_robot_tested": False,
    "passed": False,
}
app = None
exit_code = 1
try:
    # CUDA availability is checked before any numerical work.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is prohibited")
    torch.cuda.set_device(args.device)
    result["cuda"] = {
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }

    from isaaclab.app import AppLauncher

    # Ignore this script's CLI flags inside Kit. Keep all runtime output local.
    sys.argv = [sys.argv[0]]
    kit_args = " ".join([
        "--portable-root", str(runtime / "kit"),
        f"--/log/file={runtime / 'kit.log'}",
        "--/app/settings/persistent=false",
        "--/app/telemetry/enabled=false",
        "--/renderer/multiGpu/enabled=false",
    ])
    app = AppLauncher(headless=True, device=args.device, kit_args=kit_args).app
    result["app_started"] = True

    import warp as wp
    wp.config.kernel_cache_dir = str(runtime / "warp_cache")
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    import isaacsim.core.utils.prims as prim_utils

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    for index in range(args.num_objects):
        prim_utils.create_prim(f"/World/Instance{index:02d}", "Xform", translation=(index * 0.5, 0.0, 0.0))
    objects = RigidObject(RigidObjectCfg(
        prim_path="/World/Instance.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    ))
    sim.reset()
    initial = objects.data.root_pos_w.clone()
    assert initial.is_cuda, "Physics state tensor is not on CUDA"
    for _ in range(args.steps):
        objects.write_data_to_sim()
        sim.step(render=False)
        objects.update(sim.get_physics_dt())
    final = objects.data.root_pos_w.clone()
    torch.cuda.synchronize()
    assert final.is_cuda and bool(torch.isfinite(final).all()), "Invalid CUDA physics state"
    assert bool((final[:, 2] < initial[:, 2]).all()), "Rigid bodies did not respond to gravity"
    result["physics"] = {
        "simulation_device": str(sim.device),
        "state_device": str(final.device),
        "objects": int(objects.num_instances),
        "steps": args.steps,
        "dt": sim.get_physics_dt(),
        "initial_mean_height_m": initial[:, 2].mean().item(),
        "final_mean_height_m": final[:, 2].mean().item(),
        "finite_states": True,
        "gravity_response": True,
    }

    # One synthetic update, independent of simulator observations/rewards.
    # Parameters, samples, targets, loss, and gradients all reside on CUDA.
    model = torch.nn.Linear(4, 2, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(16, 4, device=args.device)
    targets = torch.zeros(16, 2, device=args.device)
    before = model.weight.detach().clone()
    loss = (model(inputs) - targets).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None and p.grad.is_cuda for p in model.parameters())
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    optimizer.step()
    torch.cuda.synchronize()
    changed = not torch.equal(before, model.weight)
    assert changed, "Optimizer did not update CUDA parameters"
    result["optimizer"] = {
        "device": str(model.weight.device),
        "loss": loss.item(),
        "finite_cuda_gradients": True,
        "parameters_updated": changed,
        "updates": 1,
        "data": "synthetic tensors, independent of simulator",
    }
    result["passed"] = True
    exit_code = 0
except Exception as exc:
    result["error"] = repr(exc)
    result["traceback"] = traceback.format_exc()
    traceback.print_exc()
finally:
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print("ISAAC_SMOKE_RESULT=" + json.dumps(result), flush=True)
    if app is not None:
        # Isaac Sim 4.5 can stall while closing its stage. This disposable
        # check has no files or training state pending; bound ONLY its own
        # cleanup, preserving whether graceful shutdown really completed.
        def exit_after_shutdown_timeout():
            result["shutdown"] = {
                "completed": False,
                "forced_process_exit": True,
                "timeout_seconds": 15,
            }
            args.report.write_text(json.dumps(result, indent=2) + "\n")
            print("ISAAC_SHUTDOWN_TIMEOUT: check results were saved; exiting this process only", flush=True)
            os._exit(exit_code)

        watchdog = threading.Timer(15.0, exit_after_shutdown_timeout)
        watchdog.daemon = True
        watchdog.start()
        app.close(wait_for_replicator=False)
        watchdog.cancel()
        result["shutdown"] = {"completed": True, "forced_process_exit": False}
        args.report.write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(exit_code)
