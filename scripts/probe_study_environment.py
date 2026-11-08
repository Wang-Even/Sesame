#!/usr/bin/env python3
"""CUDA study probe: paired domains, nominal restore and independent metrics.

Launch only when the coordinator has handed off the simulator GPU slot.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_ppo import disable_git_imports, write_json


def main():
    runtime = ROOT / ".runtime/study_probe"
    for key, folder in {"XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
                        "XDG_DATA_HOME": "data", "CUDA_CACHE_PATH": "cuda_cache"}.items():
        path = runtime / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.dont_write_bytecode = True
    report_path = ROOT / "reports/study_environment_probe.json"
    report = {"passed": False, "scope": "16-environment CUDA physics and independent CUDA diagnostic recomputation"}
    app = env = None
    code = 1
    try:
        disable_git_imports()
        import torch
        assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        app = AppLauncher(headless=True, device="cuda:0", kit_args=(
            f"--portable-root {runtime / 'kit'} --/log/file={runtime / 'kit.log'} "
            "--/app/settings/persistent=false --/renderer/multiGpu/enabled=false"
        )).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / "warp_cache")
        from sesame_rl.study_environment import SesameStudyEnv, make_study_env_cfg
        env = SesameStudyEnv(make_study_env_cfg(num_envs=16, seed=42, variant="full_dr"))

        def independent_metric_audit(steps, varying_actions=False):
            """Recompute from raw terminal-safe snapshots, never env sums."""
            n = env.num_envs
            sums = torch.zeros((n, 8), device=env.device)
            counts = torch.zeros(n, dtype=torch.long, device=env.device)
            accel_counts = torch.zeros_like(counts)
            heading_max = torch.zeros(n, device=env.device)
            previous_rate = torch.zeros((n, 8), device=env.device)
            previous_joint_velocity = env.robot.data.joint_vel[:, env.firmware_joint_ids].clone()
            initial_y = env.robot.data.root_pos_w[:, 1].clone()
            zeros = torch.zeros((n, 8), device=env.device)
            phase = torch.arange(n * 8, device=env.device).reshape(n, 8) * 0.1
            checked = []
            max_error = torch.zeros((), device=env.device)
            metric_names = (
                "velocity_x_rmse_m_s", "velocity_xy_tracking_rmse_m_s", "heading_rms_rad",
                "heading_max_abs_rad", "lateral_mean_abs_m", "lateral_rms_m",
                "applied_target_rate_rms_rad_s", "applied_target_acceleration_rms_rad_s2",
                "actual_joint_acceleration_rms_rad_s2",
            )
            for step in range(steps):
                before = env._targets.clone()
                actions = 0.10 * torch.sin(phase + step * 0.7) if varying_actions else zeros
                observations, reward, terminated, truncated, _ = env.step(actions)
                assert observations["policy"].is_cuda and bool(torch.isfinite(observations["policy"]).all())
                assert reward.is_cuda and bool(torch.isfinite(reward).all())
                sample = env.last_control_sample
                state = sample["root_state_w"]
                q = state[:, 3:7]
                yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                  1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                rate = (sample["applied_targets"] - before) / env.step_dt
                accel = (rate - previous_rate) / env.step_dt
                valid_accel = counts > 0
                x_error = state[:, 7] - env.cfg.command_forward_m_s
                lateral = state[:, 1] - initial_y
                sums[:, 0] += x_error ** 2
                sums[:, 1] += x_error ** 2 + state[:, 8] ** 2
                sums[:, 2] += yaw ** 2
                sums[:, 3] += abs(lateral)
                sums[:, 4] += lateral ** 2
                sums[:, 5] += torch.sum(rate ** 2, dim=1) / 8
                sums[:, 6] += torch.where(valid_accel, torch.sum(accel ** 2, dim=1) / 8, 0.0)
                joint_accel = (sample["joint_velocity_firmware"] - previous_joint_velocity) / env.step_dt
                sums[:, 7] += torch.sum(joint_accel ** 2, dim=1) / 8
                counts += 1
                accel_counts += valid_accel.long()
                heading_max = torch.maximum(heading_max, abs(yaw))
                previous_rate.copy_(rate)
                previous_joint_velocity.copy_(sample["joint_velocity_firmware"])
                done = terminated | truncated
                ids = torch.nonzero(done, as_tuple=False).flatten()
                if len(ids):
                    records = list(env.episode_records)[-len(ids):]
                    assert [r["env_id"] for r in records] == ids.tolist()
                    means = sums[ids] / counts[ids, None]
                    expected = torch.stack((
                        means[:, 0].sqrt(), means[:, 1].sqrt(), means[:, 2].sqrt(),
                        heading_max[ids], means[:, 3], means[:, 4].sqrt(), means[:, 5].sqrt(),
                        (sums[ids, 6] / accel_counts[ids].clamp_min(1)).sqrt(),
                        means[:, 7].sqrt(),
                    ), dim=1)
                    actual = torch.tensor([
                        [r["study_metrics"][name] if r["study_metrics"][name] is not None else 0.0
                         for name in metric_names] for r in records
                    ], device=env.device)
                    assert torch.allclose(actual, expected, rtol=3e-5, atol=1e-6), (actual - expected).abs().max().item()
                    max_error = torch.maximum(max_error, (actual - expected).abs().max())
                    for row, count, acount in zip(records, counts[ids].tolist(), accel_counts[ids].tolist(), strict=True):
                        metrics = row["study_metrics"]
                        assert metrics["samples"] == metrics["target_rate_samples"] == count
                        assert metrics["actual_joint_acceleration_samples"] == count
                        assert metrics["target_acceleration_samples"] == acount == count - 1
                        if count == 1:
                            assert metrics["applied_target_acceleration_rms_rad_s2"] is None
                        checked.append({"env_id": row["env_id"], "samples": count,
                                        "acceleration_samples": acount, "fell": row["fell"]})
                    # Initial targets after automatic reset are different from
                    # terminal targets; before-target capture on the NEXT step
                    # ensures this reset jump cannot contaminate derivatives.
                    sums[ids] = 0.0
                    counts[ids] = 0
                    accel_counts[ids] = 0
                    heading_max[ids] = 0.0
                    previous_rate[ids] = 0.0
                    previous_joint_velocity[ids] = env.robot.data.joint_vel[ids][:, env.firmware_joint_ids]
                    initial_y[ids] = env.robot.data.root_pos_w[ids, 1]
            return {"control_steps": steps, "checked_episodes": checked,
                    "maximum_absolute_metric_error": max_error.item(), "passed": True}

        with torch.inference_mode():
            env.reset(seed=3042)
            full = env.domain_snapshot()
            full_rng = env._domain_rng.get_state().to(env.device)
            full_root = env.robot.data.root_state_w.clone()
            full_joints = env.robot.data.joint_pos.clone()
            report["full_dr_readback"] = env.verify_domain_parameters()
            env.reset(seed=3042)
            assert full == env.domain_snapshot()
            assert torch.equal(full_root, env.robot.data.root_state_w)
            assert torch.equal(full_joints, env.robot.data.joint_pos)
            report["repeated_seed_reproduces_domain_and_initial_state"] = True

            # This public configuration change is only a probe convenience;
            # production runs construct a separate configuration for each arm.
            env.cfg.study_variant = "fixed_delay"
            env.reset(seed=3042)
            fixed = env.domain_snapshot()
            assert torch.equal(full_rng, env._domain_rng.get_state().to(env.device))
            assert torch.equal(full_root, env.robot.data.root_state_w)
            assert torch.equal(full_joints, env.robot.data.joint_pos)
            for key in full:
                if key != "parameters":
                    assert full[key] == fixed[key], key
            for key in full["parameters"]:
                if key not in ("target_tau_s", "delay_control_steps"):
                    assert full["parameters"][key] == fixed["parameters"][key], key
            report["fixed_delay_readback"] = env.verify_domain_parameters()
            report["fixed_delay_same_rng_and_identical_other_domains_and_initial_state"] = True
            report["full_dr_sample"] = full
            report["fixed_delay_sample"] = fixed

            env.reset(seed=4042)
            assert env.domain_snapshot() != fixed
            report["different_seed_changes_domains"] = True
            env.set_domain_randomization(False)
            env.reset(seed=3042)
            fixed_nominal = env.domain_snapshot()
            nominal_root = env.robot.data.root_state_w.clone()
            nominal_joints = env.robot.data.joint_pos.clone()
            report["nominal_readback"] = env.verify_domain_parameters()
            env.cfg.study_variant = "full_dr"
            env.reset(seed=3042)
            assert fixed_nominal == env.domain_snapshot()
            assert torch.equal(nominal_root, env.robot.data.root_state_w)
            assert torch.equal(nominal_joints, env.robot.data.joint_pos)
            report["fixed_and_full_nominal_restore_identical"] = True

            env.episode_records.clear()
            report["nominal_full_10s_metrics"] = independent_metric_audit(500)
            records = report["nominal_full_10s_metrics"]["checked_episodes"]
            assert len(records) == 16 and all(r["samples"] == 500 and not r["fell"] for r in records)
            report["metric_definitions"] = env.metric_definitions

            # Several short episodes exercise terminal samples, history clears,
            # and target derivative boundaries after actual automatic resets.
            env.cfg.episode_length_s = 0.2
            env.cfg.study_variant = "fixed_delay"
            env.set_domain_randomization(True)
            env.reset(seed=3042)
            env.episode_records.clear()
            report["fixed_delay_three_episode_boundary_metrics"] = independent_metric_audit(30, varying_actions=True)
            assert len(report["fixed_delay_three_episode_boundary_metrics"]["checked_episodes"]) == 48
            report["fixed_delay_after_rollout_readback"] = env.verify_domain_parameters()

            # Reset only two environments while the other diagnostic histories
            # are nonempty; neither their sums nor raw state may change.
            env.cfg.episode_length_s = 10.0
            env.reset(seed=3042)
            independent_metric_audit(3, varying_actions=True)
            subset = torch.tensor([1, 7], device=env.device)
            untouched = torch.tensor([i for i in range(16) if i not in (1, 7)], device=env.device)
            saved_sums = env._study_sums[untouched].clone()
            saved_rate = env._study_previous_target_rate[untouched].clone()
            saved_velocity = env._study_previous_joint_velocity[untouched].clone()
            saved_root = env.robot.data.root_state_w[untouched].clone()
            saved_domains = env.domain_snapshot(untouched)
            env._reset_idx(subset)
            assert torch.equal(saved_sums, env._study_sums[untouched])
            assert torch.equal(saved_rate, env._study_previous_target_rate[untouched])
            assert torch.equal(saved_velocity, env._study_previous_joint_velocity[untouched])
            assert torch.equal(saved_root, env.robot.data.root_state_w[untouched])
            assert saved_domains == env.domain_snapshot(untouched)
            assert bool((env._study_count[subset] == 0).all())
            assert bool((env._study_previous_target_rate[subset] == 0).all())
            report["subset_reset_preserves_other_metrics_states_and_domains"] = True

            env.cfg.episode_length_s = 0.02
            env.set_domain_randomization(False)
            env.reset(seed=3042)
            env.episode_records.clear()
            report["one_step_episode_undefined_acceleration"] = independent_metric_audit(1)
            assert len(report["one_step_episode_undefined_acceleration"]["checked_episodes"]) == 16
        report["passed"] = True
        code = 0
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(report_path, report)
        print("STUDY_PROBE_RESULT=" + json.dumps({k: v for k, v in report.items() if isinstance(v, (str, bool))}), flush=True)
        if app is not None:
            def force_exit():
                report["shutdown"] = {"completed": False, "forced_process_exit": True}
                write_json(report_path, report)
                os._exit(code)
            timer = threading.Timer(15, force_exit)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
