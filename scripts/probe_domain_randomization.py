#!/usr/bin/env python3
"""Exercise actual GPU domain randomization, reproducibility, and nominal restore."""
import json
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_ppo import disable_git_imports, write_json

runtime = ROOT / ".runtime/dr_probe"
for key, folder in {"XDG_CACHE_HOME":"cache", "XDG_CONFIG_HOME":"config", "XDG_DATA_HOME":"data", "CUDA_CACHE_PATH":"cuda_cache"}.items():
    path = runtime / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
report_path = ROOT / "reports/domain_randomization_probe.json"
report = {"passed":False,"scope":"CUDA simulation with real property readback, seeded resets and nominal restore"}
app = env = None
code = 1
try:
    disable_git_imports()
    import torch
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    app = AppLauncher(headless=True, device="cuda:0", kit_args=f"--portable-root {runtime / 'kit'} --/log/file={runtime / 'kit.log'} --/app/settings/persistent=false --/renderer/multiGpu/enabled=false").app
    import warp as wp
    wp.config.kernel_cache_dir = str(runtime / "warp_cache")
    from sesame_rl.dr_environment import SesameDREnv, make_env_cfg
    env = SesameDREnv(make_env_cfg(num_envs=16, seed=42, domain_randomization=True))
    with torch.inference_mode():
        env.reset(seed=2042)
        first = env.domain_snapshot()
        initial_root = env.robot.data.root_state_w.clone()
        initial_joints = env.robot.data.joint_pos.clone()
        report["randomized_readback"] = env.verify_domain_parameters()
        env.reset(seed=2042)
        second = env.domain_snapshot()
        assert first == second, "Seeded domain samples differ"
        assert torch.equal(initial_root, env.robot.data.root_state_w)
        assert torch.equal(initial_joints, env.robot.data.joint_pos)
        report["same_seed_matches_domains_and_initial_state"] = True
        env.reset(seed=2043)
        third = env.domain_snapshot()
        assert third != first, "Different seeds must change randomized properties"
        report["different_seed_changes_domains"] = True
        report["randomized_samples_seed2043"] = third
        zeros = torch.zeros((env.num_envs,8), device="cuda:0")
        falls = 0
        for _ in range(200):
            obs,reward,terminated,truncated,info = env.step(zeros)
            assert obs["policy"].is_cuda and bool(torch.isfinite(obs["policy"]).all())
            assert reward.is_cuda and bool(torch.isfinite(reward).all())
            falls += int(terminated.sum().item())
        report["randomized_zero_action_4s"] = {"num_envs":16,"fall_terminations":falls,"finite":True}
        env.set_domain_randomization(False)
        env.reset(seed=2042)
        nominal = env.domain_snapshot()
        report["nominal_samples"] = nominal
        report["nominal_readback"] = env.verify_domain_parameters()
        falls = 0
        for _ in range(200):
            obs,reward,terminated,truncated,info = env.step(zeros)
            assert bool(torch.isfinite(obs["policy"]).all())
            assert bool(torch.isfinite(reward).all())
            falls += int(terminated.sum().item())
        assert falls == 0, "Nominal zero-action stance must remain stable after restoration"
        report["nominal_zero_action_4s"] = {"fall_terminations":falls,"finite":True}
        env.set_domain_randomization(True)
        env.reset(seed=2042)
        assert env.domain_snapshot() == first, "DR restoration changed seeded parameters"
        report["roundtrip_mode_restore_seed_match"] = True
        for _ in range(32):
            obs,reward,terminated,truncated,info = env.step(torch.rand_like(zeros)*2-1)
            assert bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(reward).all())
        report["random_actions_32_steps_finite"] = True
        untouched = [index for index in range(env.num_envs) if index not in (1,7)]
        untouched_domains = env.domain_snapshot(untouched)
        untouched_root = env.robot.data.root_state_w[untouched].clone()
        untouched_history = env._target_history[untouched].clone()
        env._reset_idx(torch.tensor([1,7], device="cuda:0"))
        assert env.domain_snapshot(untouched) == untouched_domains
        assert torch.equal(untouched_root, env.robot.data.root_state_w[untouched])
        assert torch.equal(untouched_history, env._target_history[untouched])
        report["subset_reset_preserves_other_domains_states_and_delay_history"] = True
        report["after_subset_reset_readback"] = env.verify_domain_parameters()
    report["passed"] = True
    code = 0
except Exception as exc:
    report.update(error=repr(exc), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    write_json(report_path,report)
    print("DR_PROBE_RESULT="+json.dumps({key:value for key,value in report.items() if isinstance(value,(bool,str))}),flush=True)
    if app is not None:
        def force_exit():
            report["shutdown"]={"completed":False,"forced_process_exit":True}
            write_json(report_path,report)
            os._exit(code)
        timer=threading.Timer(15,force_exit)
        timer.daemon=True
        timer.start()
        if env is not None:
            env.close()
        app.close(wait_for_replicator=False)
        timer.cancel()
raise SystemExit(code)
