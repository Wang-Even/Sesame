#!/usr/bin/env python3
"""Record the first fixed-seed PPO/SAC/branching-DQN episode using live Isaac RTX.

Run with env_isaaclab Python through its Conda hooks. The selected training seed
is 42, test seed 25042, full randomized physics, heading_v1 reward, and at most
10 seconds. Early failure is retained without selecting another episode.
Camera, grid, arrows and HUD follow record_heading_comparison.py; no training
or physical settings are modified. CPU RGB annotation/encoding is presentation
work only. All policy inference and simulation must remain on CUDA.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from train_ppo import ROOT, disable_git_imports
from train_ppo_study import require_runtime
from evaluate_offpolicy import (ENVIRONMENT_SOURCES, OffpolicyAdapter,
                                canonical_hash, resolve_checkpoints, validate_protocol)
from record_heading_comparison import ffprobe, sha256, write_json

POLICIES = ('ppo_seed42', 'sac_seed42', 'dqn_seed42')
TEST_SEED = 25042


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--policy-id', choices=POLICIES, required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=25, choices=(25, 50))
    parser.add_argument('--encoder', choices=('h264_nvenc', 'libx264'), default='h264_nvenc')
    args = parser.parse_args()
    args.protocol, args.output_dir = args.protocol.resolve(), args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / 'runs'):
        parser.error('output-dir must be inside Sesame runs')
    if min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
        parser.error('Video dimensions must be positive even numbers')
    return args


def resolve_video_policy(protocol_path, policy_id):
    """Validate only metadata; this function does not import/execute models."""
    protocol = json.loads(protocol_path.read_text())
    if sha256(protocol_path) != protocol_path.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('Evaluation protocol hash mismatch')
    validation_args = argparse.Namespace(smoke=False, num_envs=None,
                                         episodes_per_seed=None, eval_seeds=None)
    validate_protocol(protocol, validation_args)
    selected = [entry for entry in protocol['checkpoints'] if entry['id'] == policy_id]
    if len(selected) != 1 or selected[0]['train_seed'] != 42:
        raise ValueError('Video requires one declared seed-42 final checkpoint')
    if selected[0]['variant'] != policy_id.removesuffix('_seed42'):
        raise ValueError('Policy id and algorithm differ')
    entry = resolve_checkpoints(dict(protocol, checkpoints=selected), smoke=False)[0]
    return protocol, entry


def add_visual_grid(sim_utils, stage, Gf, UsdGeom):
    """Original comparison layout, using purely visual non-colliding prims."""
    for index in range(-6, 24):
        major = index % 5 == 0
        line = sim_utils.CuboidCfg(
            size=(0.0010 if major else 0.00045, 1.6, 0.00015),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.45, 0.49) if major else (0.32, 0.35, 0.38)))
        line.func(f'/World/VideoGrid/X_{index+6}', line, translation=(index*0.1, 0.0, 0.00015))
    for index in range(-8, 9):
        line = sim_utils.CuboidCfg(size=(3.0, 0.00045, 0.00015),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.35, 0.38)))
        line.func(f'/World/VideoGrid/Y_{index+8}', line, translation=(0.85, index*0.1, 0.00015))
    for index in range(5):
        x = index*0.5
        shaft = sim_utils.CuboidCfg(size=(0.075, 0.003, 0.0002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.72, 0.78)))
        shaft.func(f'/World/WorldXArrows/shaft{index}', shaft, translation=(x+0.04, -0.15, 0.0003))
        arrow = UsdGeom.Mesh.Define(stage, f'/World/WorldXArrows/head{index}')
        arrow.CreatePointsAttr([Gf.Vec3f(x+0.10, -0.15, 0.0003),
                               Gf.Vec3f(x+0.07, -0.16, 0.0003),
                               Gf.Vec3f(x+0.07, -0.14, 0.0003)])
        arrow.CreateFaceVertexCountsAttr([3])
        arrow.CreateFaceVertexIndicesAttr([0, 1, 2])
        arrow.CreateDisplayColorAttr([(0.18, 0.72, 0.78)])
        arrow.CreateDoubleSidedAttr(True)


def initial_snapshot(env):
    return {
        'evaluation_seed': TEST_SEED, 'num_envs': 1, 'condition': 'randomized',
        'domain_snapshot': env.domain_snapshot(),
        'root_state_w': env.robot.data.root_state_w[0].tolist(),
        'joint_pos_rad': env.robot.data.joint_pos[0].tolist(),
        'joint_vel_rad_s': env.robot.data.joint_vel[0].tolist(),
        'applied_targets_firmware_rad': env._targets[0].tolist(),
    }


def validate_prior_pairing(output, report_path, report):
    paired = []
    for path in output.parent.glob('*/offpolicy_*_seed25042.json'):
        if path == report_path:
            continue
        prior = json.loads(path.read_text())
        if prior.get('status') != 'completed':
            continue
        if prior.get('protocol_sha256') != report['protocol_sha256']:
            raise RuntimeError(f'Prior video has a different protocol: {path}')
        if (prior.get('initial_conditions') != report['initial_conditions'] or
                prior.get('initial_condition_sha256') != report['initial_condition_sha256']):
            raise RuntimeError(f'Video initial state/physical parameters do not pair: {path}')
        paired.append(prior['policy_id'])
    return paired


def main():
    args = parse_args()
    protocol, entry = resolve_video_policy(args.protocol, args.policy_id)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'recording.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stem = f'offpolicy_{args.policy_id}_seed{TEST_SEED}'
    video, report_path = output / (stem+'.mp4'), output / (stem+'.json')
    if any((output / name).exists() for name in (stem+'.mp4', stem+'.json', 'thumbnail.png')):
        raise FileExistsError('Preserve prior video/report; use a new output directory')
    runtime = ROOT / '.runtime/offpolicy_video'
    for key, relative in {'XDG_CACHE_HOME':'cache', 'XDG_CONFIG_HOME':'config',
                          'XDG_DATA_HOME':'data', 'CUDA_CACHE_PATH':'cuda_cache'}.items():
        path = runtime / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    source_paths = (*ENVIRONMENT_SOURCES, 'sesame_rl/offpolicy.py',
                    'scripts/evaluate_offpolicy.py', 'scripts/record_heading_comparison.py',
                    'scripts/record_offpolicy_video.py')
    report = dict(
        status='running', pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat(),
        policy_id=args.policy_id, training_variant=entry['variant'], train_seed=42,
        checkpoint=entry['checkpoint'], checkpoint_sha256=entry['sha256'],
        protocol=str(args.protocol), protocol_sha256=sha256(args.protocol),
        config_sha256=entry['config_sha256'], source_checks=entry['source_checks'],
        recording_sources_sha256={path: sha256(ROOT / path) for path in source_paths},
        domain_mode='randomized', evaluation_environment_variant='heading_full_dr_for_all_policies',
        reward_version='absolute_heading_v1', heading_penalty_scale=0.5,
        seed=TEST_SEED, num_envs=1, device='cuda:0', fps=args.fps,
        resolution=[args.width, args.height], configured_episode_length_s=10.0,
        policy='deterministic: PPO trained normalizers; SAC tanh(mean); DQN greedy discrete branches',
        scope='Live Isaac RTX rendering of the first episode, fixed before training results; at most 10s, early failure retained.',
        comparison_note='Single-environment illustration, separate from batched statistics. CUDA draw shapes change physical samples with num_envs. Prior videos are paired only after exact initial-state/domain hash checks.',
        camera='world-aligned oblique translation-follow camera, preserving visible body yaw',
        visual_only_changes='Non-colliding 0.1m grid, world +X arrows, camera, HUD',
        training_environment_source_unchanged=True, git_operations='disabled')
    write_json(report_path, report)
    app = env = encoder = encoder_log = None
    code, frames = 1, 0
    started = time.monotonic()
    try:
        require_runtime()
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU model/simulation fallback prohibited')
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        app = AppLauncher(headless=True, enable_cameras=True, device='cuda:0',
            kit_args=f'--portable-root {runtime / "kit"} --/log/file={output / "recording_isaac.log"} --/app/settings/persistent=false --/app/telemetry/enabled=false --/renderer/multiGpu/enabled=false').app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / 'warp_cache')
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.heading_environment import SesameHeadingEnv, make_heading_env_cfg
        from sesame_rl.offpolicy import load_policy
        from pxr import Gf, UsdGeom
        from PIL import Image, ImageDraw, ImageFont

        cfg = make_heading_env_cfg(num_envs=1, seed=TEST_SEED, device='cuda:0',
                                   episode_length_s=10.0, heading_penalty_scale=0.5)
        cfg.viewer.resolution = (args.width, args.height)
        cfg.viewer.eye, cfg.viewer.lookat = (-0.17, -0.32, 0.20), (0.02, 0.0, 0.035)
        cfg.viewer.origin_type = 'world'
        cfg.sim.render_interval = cfg.decimation * (50 // args.fps)
        env = SesameHeadingEnv(cfg, render_mode='rgb_array')
        env.set_domain_randomization(True)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        stage = get_current_stage()
        add_visual_grid(sim_utils, stage, Gf, UsdGeom)
        camera = UsdGeom.Camera(stage.GetPrimAtPath(cfg.viewer.cam_prim_path))
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.005, 100.0))
        camera.GetFocalLengthAttr().Set(24.0)
        if sha256(entry['checkpoint']) != entry['sha256']:
            raise RuntimeError('Checkpoint changed after provenance verification')
        if entry['variant'] == 'ppo':
            config = entry['config']
            policy_cfg = copy.deepcopy(config['policy'])
            policy_cfg.pop('class_name')
            with torch.device('cuda:0'):
                policy = ActorCritic(wrapper.get_observations(), config['obs_groups'],
                                    wrapper.num_actions, **policy_cfg).to('cuda:0')
            state = torch.load(entry['checkpoint'], map_location='cuda:0', weights_only=False)
            policy.load_state_dict(state['model_state_dict'])
            policy.eval()
            if not all(value.is_cuda for value in (*policy.parameters(), *policy.buffers())):
                raise RuntimeError('PPO parameters and normalization buffers must remain on CUDA')
            del state
        else:
            policy = OffpolicyAdapter(load_policy(entry['checkpoint'], device='cuda:0'))
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        font, title_font, small_font = (ImageFont.truetype(font_path, size) for size in (23, 30, 18))
        label = {'ppo':'PPO + HEADING', 'sac':'SAC', 'dqn':'DQN (BRANCHING)'}[entry['variant']]
        encoder_log = (output / 'ffmpeg.log').open('wb')
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo',
                   '-pixel_format', 'rgb24', '-video_size', f'{args.width}x{args.height}',
                   '-framerate', str(args.fps), '-i', '-', '-an', '-c:v', args.encoder]
        command += (['-preset', 'slow', '-rc', 'vbr', '-cq', '19', '-b:v', '5M']
                    if args.encoder == 'h264_nvenc' else ['-preset', 'medium', '-crf', '19'])
        command += ['-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
        trajectory, terminal = [], None
        stride = 50 // args.fps
        with torch.inference_mode():
            # Reseed AFTER model initialization; every method gets the same
            # single-env starting state and physical sample.
            env.reset(seed=TEST_SEED)
            env.episode_records.clear()
            obs = wrapper.get_observations()
            initial = initial_snapshot(env)
            report.update(initial_conditions=initial, initial_condition_sha256=canonical_hash(initial),
                          domain_parameters=initial['domain_snapshot'], domain_readback=env.verify_domain_parameters())
            report['exactly_paired_with_prior_videos'] = validate_prior_pairing(output, report_path, report)
            write_json(report_path, report)
            initial_position = env.robot.data.root_pos_w[0].clone()
            env.render()
            for _ in range(12):
                env.sim.render()
            for step in range(env.max_episode_length):
                root = env.robot.data.root_state_w[0]
                x, y, z = root[:3].tolist()
                delta_x = float((root[0]-initial_position[0]).item())
                qw, qx, qy, qz = root[3:7].tolist()
                yaw_deg = math.degrees(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
                if step % stride == 0:
                    env.sim.set_camera_view(eye=(x-0.17, y-0.32, 0.20), target=(x+0.02, y, 0.035))
                    env.sim.render()
                    rgb = env.render(recompute=True)
                    if rgb is None or not rgb.size:
                        raise RuntimeError('RTX render returned no RGB data')
                    frame = Image.fromarray(rgb).convert('RGB')
                    if frame.size != (args.width, args.height):
                        raise RuntimeError('Rendered frame dimensions differ from encoder configuration')
                    draw = ImageDraw.Draw(frame, 'RGBA')
                    draw.rectangle((0, 0, args.width, 88), fill=(10, 17, 25, 225))
                    draw.text((28, 14), f'SESAME  /  {label}  /  TRAIN SEED 42', font=title_font, fill=(235, 241, 247, 255))
                    draw.text((29, 54), f'LIVE POLICY  ·  TEST SEED {TEST_SEED}  ·  FULL DR  ·  TARGET +X 0.10 m/s', font=small_font, fill=(161, 185, 202, 255))
                    draw.rectangle((0, args.height-68, args.width, args.height), fill=(10, 17, 25, 225))
                    draw.text((28, args.height-53), f't  {step*env.step_dt:5.2f} s      dx  {delta_x:5.3f} m      heading  {yaw_deg:+6.1f} deg', font=font, fill=(230, 241, 247, 255))
                    draw.text((args.width-282, args.height-50), 'GRID 0.10 m   /   +X →', font=small_font, fill=(91, 205, 215, 255))
                    if frames in (0, args.fps*5):
                        frame.save(output / f'frame_{frames:04d}.png')
                        frame.save(output / 'thumbnail.png')
                        report['thumbnail_frame_index'] = frames
                    encoder.stdin.write(frame.tobytes())
                    frames += 1
                    if frames % 50 == 0:
                        print(f'VIDEO_PROGRESS policy={args.policy_id} frames={frames} simulated_s={step*env.step_dt:.2f} dx_m={delta_x:.3f} yaw_deg={yaw_deg:.1f}', flush=True)
                actions = policy.act_inference(obs)
                if not actions.is_cuda or not bool(torch.isfinite(actions).all()):
                    raise RuntimeError('Policy actions must be finite CUDA tensors')
                trajectory.append(dict(step=step, time_s=step*env.step_dt,
                                       root_state_w=root.tolist(), raw_action=actions[0].tolist()))
                obs, reward, done, extras = wrapper.step(actions)
                if not bool(torch.isfinite(obs['policy']).all()):
                    raise RuntimeError('Nonfinite returned observation')
                if bool(done[0]):
                    if not env.episode_records:
                        raise RuntimeError('Missing first-episode terminal record')
                    terminal = dict(env.episode_records[-1])
                    break
        if terminal is None or not frames:
            raise RuntimeError('No completed first episode or video frames')
        frame.save(output / 'last_frame.png')
        if frames <= args.fps*5:
            frame.save(output / 'thumbnail.png')
            report['thumbnail_frame_index'] = frames-1
        encoder.stdin.close()
        encoder.wait(timeout=60)
        if encoder.returncode != 0:
            raise RuntimeError(f'ffmpeg exited {encoder.returncode}; see ffmpeg.log')
        encoder = None
        probe = ffprobe(video)
        stream = probe['streams'][0]
        if (stream['codec_name'], stream['width'], stream['height'], stream['r_frame_rate']) != ('h264', args.width, args.height, f'{args.fps}/1'):
            raise RuntimeError('Encoded video settings differ from requested settings')
        duration = float(probe['format']['duration'])
        if int(stream['nb_frames']) != frames or abs(duration-frames/args.fps) > 1e-4:
            raise RuntimeError('Encoded frame count/duration mismatch')
        if not 0.0 < terminal['duration_s'] <= 10.00001 or abs(duration-terminal['duration_s']) > 1/args.fps+1e-5:
            raise RuntimeError('Video does not represent the complete first episode')
        trajectory_path = output / (stem+'_trajectory.json')
        write_json(trajectory_path, dict(samples=trajectory, terminal_record=terminal))
        report.update(status='completed', video=str(video), video_sha256=sha256(video),
                      thumbnail=str(output/'thumbnail.png'), thumbnail_sha256=sha256(output/'thumbnail.png'),
                      trajectory=str(trajectory_path), trajectory_sha256=sha256(trajectory_path),
                      frames=frames, video_duration_s=duration, episode=terminal,
                      early_termination=not bool(terminal['timeout']), encoder=args.encoder, ffprobe=probe,
                      wall_seconds_before_shutdown=time.monotonic()-started)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report.update(status='interrupted' if code == 130 else 'failed', error=repr(exc),
                      traceback=traceback.format_exc(), frames=frames)
        traceback.print_exc()
    finally:
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait(timeout=10)
        if encoder_log is not None:
            encoder_log.close()
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        print('SESAME_OFFPOLICY_VIDEO_RESULT='+json.dumps(report, allow_nan=False), flush=True)
        if app is not None:
            def timeout_exit():
                report['shutdown'] = dict(completed=False, forced_process_exit=True, timeout_seconds=15)
                write_json(report_path, report)
                os._exit(code)
            timer = threading.Timer(15, timeout_exit)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            report['shutdown'] = dict(completed=True, forced_process_exit=False)
            write_json(report_path, report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
