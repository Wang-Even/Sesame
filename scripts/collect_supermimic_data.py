#!/usr/bin/env python3
"""Collect Sesame reference candidates and world-model transitions on CUDA.

This prepares an adaptation dataset, not an official SuperMimic reproduction.
Existing environments, trained weights and historical evaluation are untouched.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

from train_ppo import ROOT, disable_git_imports, write_json
from train_ppo_study import require_runtime, sha256
from evaluate_offpolicy import OffpolicyAdapter, resolve_checkpoints

PROFILES = ("straight_010", "straight_015", "straight_020", "start_stop", "action_noise_015", "action_noise_100")
SPLITS = {"train": 41042, "validation": 51042, "test": 61042}
SOURCES = ("scripts/collect_supermimic_data.py", "sesame_rl/motion_data_environment.py",
           "sesame_rl/environment.py", "sesame_rl/dr_environment.py", "sesame_rl/study_environment.py",
           "sesame_rl/heading_environment.py", "sesame_rl/offpolicy.py", "scripts/evaluate_offpolicy.py",
           "scripts/train_ppo.py", "scripts/train_ppo_study.py")
# Frozen before collecting any new data. These are engineering selection gates,
# not claims of a task's official success definition or zero jitter.
GATES = {"warmup_steps": 50, "min_steps": 500, "velocity_rmse_max_m_s": 0.05,
         "mean_velocity_error_max_m_s": 0.025, "heading_rms_max_deg": 10.0,
         "heading_max_abs_max_deg": 20.0, "lateral_displacement_max_m": 0.12,
         "target_acceleration_rms_max_rad_s2": 250.0,
         "joint_acceleration_rms_max_rad_s2": 400.0,
         "joint_limit_tolerance_rad": 0.02}


def metrics(arrays, env, torch):
    """CUDA-only quality metrics on the saved steady segment, excluding reset."""
    begin = min(GATES["warmup_steps"], len(arrays["command_world"]) - 1)
    root = arrays["state_root_state"][begin + 1:]
    command = arrays["command_world"][begin:]
    quat = root[:, 3:7]
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
    target = arrays["state_applied_target"][begin:]
    dq = arrays["state_joint_vel"][begin:]
    q = arrays["state_joint_pos"]
    all_finite = all(bool(torch.isfinite(v).all()) for v in arrays.values() if v.is_floating_point())
    acceleration = torch.diff(target, n=2, dim=0) / env.step_dt**2
    actual = torch.diff(dq, dim=0) / env.step_dt
    def scalar(t):
        value = float(t.item())
        return value if __import__('math').isfinite(value) else None
    return {
        "duration_s": len(command) * env.step_dt,
        "all_finite": all_finite,
        "fell": bool(arrays["fell"].any()), "out_of_bounds": bool(arrays["out_of_bounds"].any()),
        "invalid": bool(arrays["invalid"].any()),
        "velocity_rmse_m_s": scalar(((root[:, 7] - command[:, 0]).square() + (root[:, 8] - command[:, 1]).square()).mean().sqrt()),
        "mean_velocity_m_s": scalar(root[:, 7].mean()),
        "mean_velocity_error_m_s": scalar((root[:, 7].mean() - command[:, 0].mean()).abs()),
        "heading_rms_deg": scalar(yaw.square().mean().sqrt() * 180 / torch.pi),
        "heading_max_abs_deg": scalar(yaw.abs().max() * 180 / torch.pi),
        "lateral_displacement_m": scalar((root[-1, 1] - arrays["state_root_state"][begin, 1]).abs()),
        "target_acceleration_rms_rad_s2": scalar(acceleration.square().mean().sqrt()) if len(acceleration) else None,
        "joint_acceleration_rms_rad_s2": scalar(actual.square().mean().sqrt()),
        "joint_limit_violation_max_rad": scalar(torch.maximum(env._joint_low - q, q - env._joint_high).clamp_min(0).max()),
        "contact_force_max_n": scalar(arrays["body_contact_force"].norm(dim=-1).max()),
    }


def rejection_reasons(profile, length, m):
    reasons = []
    if not profile.startswith("straight_"): reasons.append("not_a_steady_reference_profile")
    if length < GATES["min_steps"]: reasons.append("incomplete_10s_episode")
    if not m["all_finite"]: reasons.append("nonfinite")
    for flag in ("fell", "out_of_bounds", "invalid"):
        if m[flag]: reasons.append(flag)
    for metric, gate in (
        ("velocity_rmse_m_s", "velocity_rmse_max_m_s"),
        ("mean_velocity_error_m_s", "mean_velocity_error_max_m_s"),
        ("heading_rms_deg", "heading_rms_max_deg"),
        ("heading_max_abs_deg", "heading_max_abs_max_deg"),
        ("lateral_displacement_m", "lateral_displacement_max_m"),
        ("target_acceleration_rms_rad_s2", "target_acceleration_rms_max_rad_s2"),
        ("joint_acceleration_rms_rad_s2", "joint_acceleration_rms_max_rad_s2"),
        ("joint_limit_violation_max_rad", "joint_limit_tolerance_rad"),
    ):
        if m[metric] is None or m[metric] > GATES[gate]: reasons.append(metric)
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--envs-per-profile", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-only", action="store_true", help="Supplement with recorded tilted/angular-velocity initial states")
    parser.add_argument("--reference-only", action="store_true", help="Additional independent .10 m/s candidates without changing quality gates")
    args = parser.parse_args()
    if args.recovery_only and args.reference_only: parser.error('Select one supplemental mode')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or args.envs_per_profile < 1:
        parser.error("Output must be inside Sesame and env count positive")
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists() and not args.resume:
        parser.error("Existing run; use a fresh directory or --resume")
    protocol_source = ROOT / 'runs/offpolicy_heading_20261107/evaluation_protocol.json'
    original = json.loads(protocol_source.read_text())
    selected = [e for e in original['checkpoints'] if e['id'] in ('ppo_seed42', 'sac_seed42')]
    entries = resolve_checkpoints(dict(original, checkpoints=selected), smoke=False)
    splits = {"smoke": 71042} if args.smoke else dict(SPLITS)
    if args.recovery_only: splits = {k:v+200000 for k,v in splits.items()}
    if args.reference_only: splits = {k:v+100000 for k,v in splits.items()}
    profiles = ("recovery_initial_tilt",) if args.recovery_only else (("straight_010",) if args.reference_only else PROFILES)
    steps = 100 if args.smoke else 500
    protocol = {"schema_version": 1, "dataset_kind": "Sesame SuperMimic adaptation preparation",
                "smoke": args.smoke, "steps": steps, "control_dt_s": 0.02,
                "envs_per_profile": args.envs_per_profile, "profiles": profiles, "split_seeds": splits,
                "recovery_only": args.recovery_only,
                "reference_only": args.reference_only,
                "recovery_initialization": "If enabled: alternating signed roll .25.. .90 rad, root angular velocity x signed 1.5..4.5 rad/s, root height .065m; other initial values retained. Applied BEFORE first state snapshot; no unrecorded mid-episode pushes.",
                "conditions": ["nominal", "randomized"], "reference_gates": GATES,
                "reference_caps_per_speed": {"train": 24, "validation": 8, "test": 8},
                "checkpoints": [{k: e[k] for k in ('id', 'variant', 'checkpoint', 'sha256')} for e in entries],
                "sources": {p: sha256(ROOT / p) for p in SOURCES},
                "asset_sha256": sha256(ROOT / 'assets/isaac/sesame/sesame.usd'),
                "urdf_sha256": sha256(ROOT / 'assets/isaac/sesame/import_source/sesame.urdf'),
                "split_rule": "Whole reset-seed groups; paired initial conditions across policies remain in same split.",
                "action_rule": "raw policy plus independent Gaussian noise on noise profiles, then frozen [-1,1] clip and actuator pipeline",
                "command_rule": "World-frame XYZ linear velocity; X only. The policy was trained ONLY at 0.10 m/s.",
                "start_stop_rule": "0..1 s:0; 1..3:linear 0 to .10; 3..5:.10; 5..7:linear .10 to .20; 7..9:linear .20 to 0; 9..10:0",
                "reward_rule": "legacy_reward retains frozen +X .10 m/s tracking/forward terms. It is NOT an imitation reward or a valid variable-speed task score. Recompute future reward from reference/state.",
                "failure_rule": "Retain first episode through terminal pre-reset state. No second episode appended. Nonfinite episodes quarantined.",
                "selection_rule": "Gates fixed before collection; only training clips ranked by acceleration. Held-out eligible clips ordered by id, never used for tuning."}
    protocol = json.loads(json.dumps(protocol))
    if (output / 'protocol.json').exists():
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise RuntimeError('Resume protocol or code mismatch')
    else:
        write_json(output / 'protocol.json', protocol)
        for path in SOURCES:
            dest = output / 'sources' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, dest)
    runtime = ROOT / '.runtime/supermimic_data'
    for key, relative in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data', 'CUDA_CACHE_PATH': 'cuda_cache'}.items():
        p = runtime / relative
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    result = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "protocol_sha256": sha256(output / 'protocol.json')}
    write_json(output / 'result.json', result)
    app = env = None
    started = time.monotonic()
    exit_code = 1
    try:
        require_runtime()
        disable_git_imports()
        import numpy as np
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        app = AppLauncher(headless=True, device='cuda:0', kit_args=' '.join([
            '--portable-root', str(runtime / 'kit'), f'--/log/file={output / "isaac.log"}',
            '--/app/settings/persistent=false', '--/app/telemetry/enabled=false', '--/renderer/multiGpu/enabled=false'])).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / 'warp_cache')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.heading_environment import make_heading_env_cfg
        from sesame_rl.motion_data_environment import SesameMotionDataEnv
        from sesame_rl.offpolicy import load_policy
        num_envs = len(profiles) * args.envs_per_profile
        cfg = make_heading_env_cfg(num_envs=num_envs, seed=list(splits.values())[0], episode_length_s=10)
        cfg.robot.spawn.activate_contact_sensors = True
        write_json(output / 'environment_config.json', cfg.to_dict())
        env = SesameMotionDataEnv(cfg)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1)
        write_json(output / 'schema.json', {
            "schema_version": 1, "npz_allow_pickle": False, "sample_rate_hz": 50,
            "state_arrays": "state_* have T+1 frames: state[0] reset, state[t+1] after action[t], BEFORE auto-reset",
            "transition_arrays": "Others have T frames. next_observation[t] uses command_world[t]; observations are not normalized by PPO running moments.",
            "position_frame": "World axes with per-environment origin subtracted; z=0 is ground. Velocities remain world axes.",
            "quaternion_order": "wxyz", "joint_order": list(env.firmware_joint_names),
            "body_order": list(env.robot.body_names), "foot_order": env.foot_names,
            "root_state_layout": "position of root link (3), root link quaternion (4), root COM world linear velocity (3), world angular velocity (3)",
            "body_link_state_layout": "link-origin position (3), quaternion (4), link-origin linear velocity (3), world angular velocity (3)",
            "units": "m, s, rad, N, Nm", "controller": "state_previous_action is most recently executed normalized action, state_target_history newest first, length=3",
            "contacts": "Post-step net normal contact force by rigid body, sampled at final 500Hz physics substep of each 50Hz control interval; not a full-step impulse or exact toe contact",
            "torque": "Isaac implicit-actuator applied_torque estimate; not a direct measured solver torque",
            "observation_layout": "body linear vel*5 (3), body angular vel*.25 (3), projected gravity (3), body-frame world command*5 (3), q-q_stand (8), dq/10.471975512 (8), previous executed action (8); clipped +-20",
            "domain_parameters": "Per-episode JSON context; numeric parameter values and actual property readback included. They are world-model privileged context, not deployment observations.",
            "references": "Selected 9-second steady segment after 1-second warmup; no retiming, smoothing, mirroring or artificial cyclic wrap. Tracking fields derived directly from measured states.",
            "reward_warning": protocol['reward_rule'],
        })
        profile_ids = torch.arange(num_envs, device=env.device) // args.envs_per_profile
        sigma = torch.zeros(num_envs, device=env.device)
        sigma[profile_ids == 4] = .15
        sigma[profile_ids == 5] = 1.0
        all_rows = []
        with torch.inference_mode():
            for entry in entries:
                if entry['variant'] == 'ppo':
                    config = copy.deepcopy(entry['config']['policy']); config.pop('class_name')
                    with torch.device(env.device):
                        actor = ActorCritic(wrapper.get_observations(), entry['config']['obs_groups'], wrapper.num_actions, **config).to(env.device)
                    ckpt = torch.load(entry['checkpoint'], map_location=env.device, weights_only=False)
                    actor.load_state_dict(ckpt['model_state_dict']); actor.eval(); del ckpt
                else:
                    actor = OffpolicyAdapter(load_policy(entry['checkpoint'], device=env.device))
                for split, seed in splits.items():
                    for condition in ('nominal', 'randomized'):
                        case_id = f'{split}_{entry["id"]}_{condition}'
                        case_dir = output / 'episodes' / split / case_id
                        done_path = case_dir / 'index.json'
                        if done_path.exists() and args.resume:
                            case = json.loads(done_path.read_text())
                            for row in case:
                                if sha256(output / row['path']) != row['sha256']: raise RuntimeError('Resume data checksum mismatch')
                            all_rows.extend(case)
                            continue
                        if case_dir.exists(): raise RuntimeError(f'Partial case exists; preserve and use a fresh output: {case_dir}')
                        case_dir.mkdir(parents=True)
                        env.set_domain_randomization(condition == 'randomized')
                        env._commands_w.zero_(); env._commands_w[:, 0] = .1
                        env.reset(seed=seed + (1000 if condition == 'randomized' else 0))
                        env.episode_records.clear()
                        if args.recovery_only:
                            from isaaclab.utils.math import quat_from_euler_xyz
                            initial_root = env.robot.data.root_state_w.clone()
                            signs = torch.where(torch.arange(num_envs, device=env.device) % 2 == 0, 1.0, -1.0)
                            rolls = torch.linspace(.25, .90, num_envs, device=env.device) * signs
                            initial_root[:, 2] = env.scene.env_origins[:, 2] + .065
                            initial_root[:, 3:7] = quat_from_euler_xyz(rolls, torch.zeros_like(rolls), env._heading_yaw())
                            initial_root[:, 10] = torch.linspace(1.5, 4.5, num_envs, device=env.device) * signs
                            env.robot.write_root_pose_to_sim(initial_root[:, :7])
                            env.robot.write_root_velocity_to_sim(initial_root[:, 7:])
                            env.scene.write_data_to_sim()
                            env.sim.forward()
                        domain = {k: v.clone() for k, v in env.domain_parameters.items()}
                        properties = {k: v.clone() for k, v in env._applied.items()}
                        readback = env.verify_domain_parameters()
                        write_json(case_dir / 'domain_readback.json', readback)
                        noise_rng = torch.Generator(device=env.device).manual_seed(seed + 9000 + (1000 if condition == 'randomized' else 0))
                        initial = env.snapshot()
                        state_buffers = {k: torch.empty((steps+1, *v.shape), dtype=v.dtype, device=env.device) for k, v in initial.items()}
                        for k, v in initial.items(): state_buffers[k][0].copy_(v)
                        transition_buffers = None
                        lengths = torch.full((num_envs,), steps, dtype=torch.long, device=env.device)
                        alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
                        continuity_max = 0.0
                        for step in range(steps):
                            command = torch.full((num_envs,), .1, device=env.device)
                            command[profile_ids == 1] = .15
                            command[profile_ids == 2] = .20
                            t = step * env.step_dt
                            ramp = 0.0 if t < 1 else (.05*(t-1) if t < 3 else (.1 if t < 5 else (.1+.05*(t-5) if t < 7 else (.2-.1*(t-7) if t < 9 else 0.0))))
                            command[profile_ids == 3] = ramp
                            env._commands_w[:, 0] = command
                            obs = env._get_observations()['policy']
                            if step:
                                pre = env.snapshot()
                                for key, value in pre.items():
                                    if bool(alive.any()):
                                        err = float((value[alive] - state_buffers[key][step, alive]).abs().max().item())
                                        continuity_max = max(continuity_max, err)
                                if continuity_max > 1e-5: raise RuntimeError(f'Unrecorded state jump: {continuity_max}')
                            policy_action = actor.act_inference({'policy': obs})
                            action = policy_action + torch.randn(policy_action.shape, device=env.device, generator=noise_rng) * sigma[:, None]
                            if not action.is_cuda or not bool(torch.isfinite(action).all()): raise RuntimeError('Invalid action')
                            # Direct env.step preserves the executed noisy input; clipping is in frozen preprocessing.
                            env.step(action)
                            values = {**env.capture_transition, 'observation': obs, 'policy_action': policy_action,
                                      'input_action': action, 'command_world': env._commands_w.clone()}
                            if transition_buffers is None:
                                transition_buffers = {k: torch.empty((steps, *v.shape), dtype=v.dtype, device=env.device) for k, v in values.items()}
                            for key, value in values.items(): transition_buffers[key][step].copy_(value)
                            for key, value in env.capture_next.items(): state_buffers[key][step+1].copy_(value)
                            done = values['terminated'] | values['truncated']
                            lengths[alive & done] = step+1
                            alive &= ~done
                        rows = []
                        for index in range(num_envs):
                            length = int(lengths[index].item())
                            arrays = {f'state_{k}': v[:length+1, index] for k,v in state_buffers.items()}
                            arrays.update({k: v[:length, index] for k,v in transition_buffers.items()})
                            metric = metrics(arrays, env, torch)
                            profile = profiles[index // args.envs_per_profile]
                            reasons = rejection_reasons(profile, length, metric)
                            name = f'{profile}_env{index:03d}'
                            path = case_dir / (name + '.npz')
                            # File transport/compression only; no CPU model or kinematic computation.
                            np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k,v in arrays.items()})
                            metadata = {'id': case_id + '_' + name, 'split': split, 'profile': profile,
                                        'policy_id': entry['id'], 'checkpoint_sha256': entry['sha256'],
                                        'condition': condition, 'reset_seed': seed + (1000 if condition == 'randomized' else 0),
                                        'env_id': index, 'frames': length+1, 'transitions': length,
                                        'domain_parameters': {k:v[index].cpu().tolist() for k,v in domain.items()},
                                        'applied_properties': {k:v[index].cpu().tolist() for k,v in properties.items()},
                                        'env_origin_world': env.scene.env_origins[index].cpu().tolist(),
                                        'continuity_max_abs': continuity_max,
                                        'metrics': metric, 'reference_rejection_reasons': reasons,
                                        'world_model_usable': metric['all_finite'] and not metric['invalid'],
                                        'collector_cutoff': bool(alive[index].item()),
                                        'path': str(path.relative_to(output)), 'sha256': sha256(path)}
                            write_json(path.with_suffix('.json'), metadata)
                            metadata['metadata_path'] = str(path.with_suffix('.json').relative_to(output))
                            metadata['metadata_sha256'] = sha256(path.with_suffix('.json'))
                            rows.append(metadata)
                        write_json(done_path, rows)
                        all_rows.extend(rows)
                        write_json(output / 'progress.json', {'status':'running','cases':len(all_rows)//num_envs,'episodes':len(all_rows)})
                        print('DATA_CASE_COMPLETED=' + json.dumps({'case':case_id,'episodes':len(rows),
                            'transitions':sum(x['transitions'] for x in rows), 'falls':sum(x['metrics']['fell'] for x in rows),
                            'eligible_references':sum(not x['reference_rejection_reasons'] for x in rows)}), flush=True)
                        del state_buffers, transition_buffers
                del actor
        write_json(output / 'episode_index.json', all_rows)
        result.update(status='completed', episodes=len(all_rows), transitions=sum(r['transitions'] for r in all_rows),
                      gpu=torch.cuda.get_device_name(0), collector_wall_seconds=time.monotonic()-started,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'result.json', result)
        print('COLLECTION_RESULT=' + json.dumps(result), flush=True)
        if app is not None:
            timer = threading.Timer(15, lambda: os._exit(exit_code)); timer.daemon=True; timer.start()
            if env is not None: env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
