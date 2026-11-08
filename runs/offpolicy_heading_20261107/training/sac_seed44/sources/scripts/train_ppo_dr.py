#!/usr/bin/env python3
"""Train Sesame PPO from scratch with domain randomization and paired held-out tests.

The fixed baseline's policy, optimizer settings, observations and reward remain
unchanged. Only training/evaluation domains vary. This is local simulation
research, not a calibrated hardware model or official course score.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import threading
import time
import traceback

from train_ppo import (
    ROOT, disable_git_imports, jsonable, snapshot_sources, training_config, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--steps-per-env', type=int, default=32)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-seed', type=int, default=2042)
    parser.add_argument('--eval-episodes', type=int, default=64)
    parser.add_argument('--episode-length-s', type=float, default=10.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--baseline-checkpoint', type=Path,
                        default=ROOT / 'runs/ppo_baseline_20261102_seed42/final.pt')
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='DR checkpoint; only used with --eval-only')
    args = parser.parse_args()
    # These exact values are shared with the prior fixed-domain experiment.
    args.learning_rate = 3e-4
    args.hidden_dims = [128, 128, 64]
    if args.device != 'cuda:0':
        parser.error('CUDA cuda:0 is required; there is no CPU fallback')
    for key in ('num_envs', 'steps_per_env', 'eval_episodes', 'save_interval'):
        if getattr(args, key) <= 0:
            parser.error(f'{key} must be positive')
    if args.iterations <= 0 and not args.eval_only:
        parser.error('iterations must be positive')
    if args.episode_length_s <= 0:
        parser.error('episode-length-s must be positive')
    if args.eval_only != (args.checkpoint is not None):
        parser.error('--eval-only and --checkpoint must be specified together')
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    if not args.skip_baseline and not args.baseline_checkpoint.is_file():
        parser.error('baseline checkpoint does not exist')
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            parser.error('DR evaluation checkpoint does not exist')
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.run_dir = ROOT / 'runs' / f'ppo_dr_{stamp}_seed{args.seed}'
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(ROOT):
        parser.error('run-dir must remain inside the Sesame workspace')
    if (args.run_dir / 'result.json').exists():
        parser.error('run-dir already contains results; choose a new directory')
    return args


def cpu_metadata(value, torch):
    """Only serialize values after simulation/model calculations on the GPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): cpu_metadata(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_metadata(item, torch) for item in value]
    return value


def get_domains(env, count, torch):
    indices = list(range(count))
    return cpu_metadata(env.get_domain_snapshot(env_ids=indices), torch)


def snapshot_dr_sources(run_dir):
    records = snapshot_sources(run_dir)
    for relative in ('scripts/train_ppo_dr.py', 'sesame_rl/dr_environment.py'):
        source = ROOT / relative
        destination = run_dir / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({'path': relative, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'bytes': source.stat().st_size})
    return records


def distribution(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError('Nonfinite or empty evaluation metric')
    return {'mean': statistics.mean(values), 'std': statistics.pstdev(values),
            'min': min(values), 'max': max(values)}


def evaluate(env, wrapper, policy, args, label, condition, torch):
    """Exactly one complete episode per selected environment under matched domains."""
    records, initial_conditions, trajectory, domain_checks = [], [], [], []
    trajectory_terminal = None
    batch = 0
    raw_action_abs_max = 0.0
    clipped_components = total_components = 0
    env.set_domain_randomization(condition == 'randomized')
    while len(records) < args.eval_episodes:
        count = min(env.num_envs, args.eval_episodes - len(records))
        batch_seed = args.eval_seed + batch
        env.reset(seed=batch_seed)
        env.episode_records.clear()
        domain_checks.append(cpu_metadata(env.verify_domain_parameters(env_ids=list(range(count))), torch))
        obs = wrapper.get_observations()
        initial_conditions.append({
            'batch_seed': batch_seed,
            'root_state_w': env.robot.data.root_state_w[:count].detach().cpu().tolist(),
            'joint_pos_rad': env.robot.data.joint_pos[:count].detach().cpu().tolist(),
            'domains': get_domains(env, count, torch),
        })
        seen = set()
        active = torch.ones(count, dtype=torch.bool, device=args.device)
        maximum_yaw = torch.zeros(count, device=args.device)
        yaw_gt45_steps = torch.zeros(count, device=args.device)
        tracked_steps = torch.zeros(count, device=args.device)
        previous_yaw = torch.zeros(count, device=args.device)
        for step in range(int(env.max_episode_length) + 5):
            # Diagnostics do not enter observations or rewards.
            q = env.robot.data.root_quat_w[:count]
            w, x, y, z = q.unbind(dim=-1)
            yaw = torch.atan2(2*(w*z + x*y), 1-2*(y.square()+z.square()))
            previous_yaw = torch.where(active, yaw, previous_yaw)
            maximum_yaw = torch.where(active, torch.maximum(maximum_yaw, yaw.abs()), maximum_yaw)
            yaw_gt45_steps += (active & (yaw.abs() > math.pi/4)).float()
            tracked_steps += active.float()
            actions = (torch.zeros((env.num_envs, wrapper.num_actions), device=args.device)
                       if policy is None else policy(obs))
            if not actions.is_cuda or not bool(torch.isfinite(actions).all()):
                raise RuntimeError(f'{condition}/{label}: invalid or non-CUDA action')
            selected_actions = actions[:count][active]
            if selected_actions.numel():
                raw_action_abs_max = max(raw_action_abs_max, float(selected_actions.abs().max().item()))
                clipped_components += int((selected_actions.abs() > 1).sum().item())
                total_components += selected_actions.numel()
            if batch == 0 and 0 not in seen:
                trajectory.append({
                    'step': step, 'time_s': step * float(env.step_dt),
                    'root_state_w': env.robot.data.root_state_w[0].detach().cpu().tolist(),
                    'joint_pos_rad': env.robot.data.joint_pos[0].detach().cpu().tolist(),
                    'raw_action': actions[0].detach().cpu().tolist(),
                    'clipped_normalized_action': actions[0].clamp(-1, 1).detach().cpu().tolist(),
                })
            obs, reward, done, extras = wrapper.step(actions)
            if not bool(torch.isfinite(reward).all()) or not bool(torch.isfinite(obs['policy']).all()):
                raise RuntimeError(f'{condition}/{label}: nonfinite state or reward')
            while env.episode_records:
                record = dict(env.episode_records.popleft())
                index = int(record['env_id'])
                if index < count and index not in seen:
                    seen.add(index)
                    active[index] = False
                    final_yaw = float(record.get('final_heading_yaw_rad', previous_yaw[index].item()))
                    record.update({
                        'batch_seed': batch_seed, 'condition': condition, 'policy': label,
                        'final_heading_yaw_deg': math.degrees(final_yaw),
                        'max_abs_heading_yaw_deg': math.degrees(max(float(maximum_yaw[index].item()), abs(final_yaw),
                                                                float(record.get('max_abs_heading_yaw_rad', 0.0)))),
                        'heading_gt45deg_time_fraction': float((yaw_gt45_steps[index]/tracked_steps[index].clamp_min(1)).item()),
                    })
                    records.append(record)
                    if batch == 0 and index == 0:
                        trajectory_terminal = record
            if len(seen) == count:
                break
        if len(seen) != count:
            raise RuntimeError(f'{condition}/{label}: incomplete evaluation episodes')
        batch += 1
    records.sort(key=lambda item: (item['batch_seed'], item['env_id']))
    metrics = {
        'policy': label, 'condition': condition, 'episodes': len(records), 'seed': args.eval_seed,
        'all_actions_observations_rewards_finite': True,
        'max_abs_raw_action': raw_action_abs_max,
        'raw_action_clipped_fraction': clipped_components / max(total_components, 1),
        'fall_rate': statistics.mean(float(row['fell']) for row in records),
        'timeout_rate': statistics.mean(float(row['timeout']) for row in records),
        'invalid_state_rate': statistics.mean(float(row.get('invalid_state', False)) for row in records),
        'out_of_bounds_rate': statistics.mean(float(row.get('out_of_bounds', False)) for row in records),
        'final_heading_gt45deg_fraction': statistics.mean(abs(row['final_heading_yaw_deg']) > 45 for row in records),
    }
    for key in ('distance_x_m', 'avg_speed_m_s', 'duration_s', 'return', 'final_heading_yaw_deg',
                'max_abs_heading_yaw_deg', 'heading_gt45deg_time_fraction'):
        metrics[key] = distribution([record[key] for record in records])
    metrics['abs_final_heading_yaw_deg'] = distribution([abs(row['final_heading_yaw_deg']) for row in records])
    if all('lateral_displacement_m' in record for record in records):
        metrics['lateral_displacement_m'] = distribution([record['lateral_displacement_m'] for record in records])
    output = args.run_dir / 'evaluation' / condition
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f'{label}.json', {'scope': 'local paired simulation evaluation, not course score',
        'metrics': metrics, 'episodes': records, 'initial_conditions': initial_conditions,
        'domain_readback_validation': domain_checks})
    write_json(output / f'trajectory_{label}.json', {
        'policy': label, 'condition': condition, 'batch_seed': args.eval_seed, 'env_id': 0,
        'robot_joint_order': list(env.robot.joint_names),
        'action_order': ['R1', 'R2', 'L1', 'L2', 'R4', 'R3', 'L3', 'L4'],
        'state_timing': 'pre-step, world coordinates; root=[xyz,quat_wxyz,linear_velocity,angular_velocity]',
        'samples': trajectory, 'terminal_record': trajectory_terminal,
    })
    return metrics, initial_conditions


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'PPO domain randomization ablation with unchanged fixed-baseline reward and algorithm',
              'args': vars(args), 'run_dir': str(args.run_dir), 'git_operations': 'disabled',
              'fresh_from_scratch': not args.eval_only, 'heading_reward_changed': False}
    write_json(args.run_dir / 'result.json', result)
    runtime = ROOT / '.runtime/ppo_dr'
    for key, relative in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
                          'XDG_DATA_HOME': 'data', 'CUDA_CACHE_PATH': 'cuda_cache'}.items():
        directory = runtime / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    app = env = runner = None
    exit_code = 1
    started = time.monotonic()
    try:
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU simulation/model execution prohibited')
        if int(os.getenv('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('Only a single CUDA GPU is supported')
        torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        kit_args = ' '.join(['--portable-root', str(runtime / 'kit'),
            f'--/log/file={args.run_dir / "isaac.log"}', '--/app/settings/persistent=false',
            '--/app/telemetry/enabled=false', '--/renderer/multiGpu/enabled=false'])
        app = AppLauncher(headless=True, device=args.device, kit_args=kit_args).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / 'warp_cache')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import rsl_rl.runners.on_policy_runner as runner_module
        from rsl_rl.modules import ActorCritic
        from sesame_rl.dr_environment import SesameDREnv, make_dr_env_cfg
        runner_module.store_code_state = lambda *unused_args, **unused_kwargs: []

        class JsonRunner(runner_module.OnPolicyRunner):
            def __init__(self, *runner_args, **runner_kwargs):
                super().__init__(*runner_args, **runner_kwargs)
                self.git_status_repos = []
                self.logger_type = 'tensorboard'

            def add_git_repo_to_log(self, unused):
                raise RuntimeError('Git operations are disabled')

            def log(self, locs, *extra_args, **extra_kwargs):
                filtered = dict(locs)
                filtered['ep_infos'] = [info for info in locs['ep_infos'] if info]
                super().log(filtered, *extra_args, **extra_kwargs)
                row = {'iteration': int(locs['it'])+1, 'environment_transitions': self.tot_timesteps,
                       'collection_seconds': float(locs['collection_time']), 'update_seconds': float(locs['learn_time']),
                       'learning_rate': float(self.alg.learning_rate),
                       'losses': {key: float(value) for key, value in locs['loss_dict'].items()},
                       'mean_episode_return': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
                       'mean_episode_length_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None}
                episode_metrics = {}
                for info in locs['ep_infos']:
                    for key, value in info.items():
                        scalar = float(value.detach().float().mean().item()) if torch.is_tensor(value) else float(value)
                        episode_metrics.setdefault(key, []).append(scalar)
                row['episode_metrics'] = {key: statistics.mean(values) for key, values in episode_metrics.items()}
                if not all(math.isfinite(value) for value in row['losses'].values()):
                    raise RuntimeError('Nonfinite PPO loss')
                with (args.run_dir / 'training.jsonl').open('a') as output:
                    output.write(json.dumps(row, allow_nan=False)+'\n')

            def save(self, path, infos=None):
                metadata = dict(infos or {})
                metadata.update({'environment_transitions': self.tot_timesteps, 'seed': args.seed,
                                 'domain_randomization': True, 'source_run_dir': str(args.run_dir)})
                super().save(path, metadata)

        config = training_config(args)
        baseline_cfg = None
        if not args.skip_baseline:
            baseline_cfg = json.loads((args.baseline_checkpoint.parent / 'ppo_config.json').read_text())
            for section in ('policy', 'algorithm', 'obs_groups'):
                if config[section] != baseline_cfg[section]:
                    raise RuntimeError(f'Baseline {section} differs; domain-only comparison requires unchanged PPO')
            if config['num_steps_per_env'] != baseline_cfg['num_steps_per_env']:
                raise RuntimeError('Rollout length must match the fixed-domain baseline')
            baseline_env_cfg = json.loads((args.baseline_checkpoint.parent / 'environment_config.json').read_text())
            if args.episode_length_s != baseline_env_cfg['episode_length_s']:
                raise RuntimeError('Episode duration must match the fixed-domain baseline')
            previous_source = args.baseline_checkpoint.parent / 'sources/sesame_rl/environment.py'
            if previous_source.read_bytes() != (ROOT / 'sesame_rl/environment.py').read_bytes():
                raise RuntimeError('Fixed environment source differs from the saved baseline')
            result['baseline_checkpoint'] = str(args.baseline_checkpoint)
            result['baseline_checkpoint_sha256'] = hashlib.sha256(args.baseline_checkpoint.read_bytes()).hexdigest()
        cfg = make_dr_env_cfg(num_envs=args.num_envs, seed=args.seed, device=args.device,
                              episode_length_s=args.episode_length_s)
        write_json(args.run_dir / 'ppo_config.json', config)
        write_json(args.run_dir / 'environment_config.json', cfg.to_dict())
        env = SesameDREnv(cfg)
        env.set_domain_randomization(True)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapper.get_observations()
        if not obs['policy'].is_cuda or not bool(torch.isfinite(obs['policy']).all()):
            raise RuntimeError('Invalid CUDA observations')
        initial_domains = get_domains(env, env.num_envs, torch)
        write_json(args.run_dir / 'domain_samples_initial.json', initial_domains)
        write_json(args.run_dir / 'domain_validation_initial.json', cpu_metadata(env.verify_domain_parameters(), torch))
        with torch.device(args.device):
            runner = JsonRunner(wrapper, copy.deepcopy(config), log_dir=str(args.run_dir), device=args.device)
        if not all(parameter.is_cuda for parameter in runner.alg.policy.parameters()):
            raise RuntimeError('Policy parameters must remain on CUDA')
        result['sources'] = snapshot_dr_sources(args.run_dir)
        result['gpu'] = torch.cuda.get_device_name(0)
        result['versions'] = {}
        for package in ('torch', 'rsl-rl-lib', 'isaaclab', 'isaaclab-rl', 'isaacsim', 'gymnasium'):
            try:
                result['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result['versions'][package] = 'not in distribution metadata'
        result['observation_shape'] = list(obs['policy'].shape)
        result['action_count'] = wrapper.num_actions
        write_json(args.run_dir / 'result.json', result)
        if args.eval_only:
            for relative in ('sesame_rl/environment.py', 'sesame_rl/dr_environment.py'):
                if (args.checkpoint.parent / 'sources' / relative).read_bytes() != (ROOT / relative).read_bytes():
                    raise RuntimeError('Evaluation requires the environment sources saved with the DR checkpoint')
            checkpoint_cfg = json.loads((args.checkpoint.parent / 'ppo_config.json').read_text())
            if checkpoint_cfg['policy'] != config['policy']:
                raise RuntimeError('Evaluation checkpoint architecture differs')
            checkpoint_info = runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
            runner.tot_timesteps = int((checkpoint_info or {}).get('environment_transitions', 0))
        else:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final_path = args.run_dir / 'final.pt'
        runner.save(str(final_path))
        if runner.writer is not None:
            runner.writer.flush()
        write_json(args.run_dir / 'domain_samples_after_training.json', get_domains(env, env.num_envs, torch))
        write_json(args.run_dir / 'domain_validation_after_training.json', cpu_metadata(env.verify_domain_parameters(), torch))

        def load_policy(path, policy_config):
            actor_config = copy.deepcopy(policy_config['policy'])
            actor_config.pop('class_name')
            with torch.device(args.device):
                actor = ActorCritic(wrapper.get_observations(), policy_config['obs_groups'],
                                    wrapper.num_actions, **actor_config).to(args.device)
            saved = torch.load(path, map_location=args.device, weights_only=False)
            actor.load_state_dict(saved['model_state_dict'])
            actor.eval()
            if not all(parameter.is_cuda for parameter in actor.parameters()):
                raise RuntimeError('Reloaded policy left CUDA')
            return actor

        dr_actor = load_policy(final_path, config)
        baseline_actor = None if args.skip_baseline else load_policy(args.baseline_checkpoint, baseline_cfg)
        runner.eval_mode()
        with torch.inference_mode():
            probe = wrapper.get_observations()
            restored = dr_actor.act_inference(probe)
            original = runner.alg.policy.act_inference(probe)
            reload_error = float((restored-original).abs().max().item())
            if reload_error != 0.0 or not bool(torch.isfinite(restored).all()):
                raise RuntimeError('Fresh checkpoint reload mismatch')
            policy_list = [('zero_residual', None)]
            if baseline_actor is not None:
                policy_list.append(('baseline_ppo', baseline_actor.act_inference))
            policy_list.append(('dr_ppo', dr_actor.act_inference))
            evaluations = {}
            for condition in ('nominal', 'randomized'):
                condition_metrics = {}
                reference_initial = None
                pairing = {}
                for label, policy in policy_list:
                    metrics, initial = evaluate(env, wrapper, policy, args, label, condition, torch)
                    condition_metrics[label] = metrics
                    if reference_initial is None:
                        reference_initial = initial
                    pairing[label] = initial == reference_initial
                    write_json(args.run_dir / f'evaluation_progress_{condition}.json', condition_metrics)
                    if not pairing[label]:
                        raise RuntimeError(f'Initial states or domain parameters differ in paired {condition}/{label} evaluation')
                condition_metrics['paired_initial_states_and_domains_equal'] = pairing
                evaluations[condition] = condition_metrics
        result.update({'status': 'completed', 'checkpoint': str(final_path),
            'checkpoint_sha256': hashlib.sha256(final_path.read_bytes()).hexdigest(),
            'environment_transitions_total': runner.tot_timesteps,
            'training_iterations_this_run': 0 if args.eval_only else args.iterations,
            'reload_max_abs_action_error': reload_error, 'evaluations': evaluations,
            'wall_seconds_before_shutdown': time.monotonic()-started})
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if interrupted:
            exit_code = 130
            if runner is not None:
                try:
                    runner.save(str(args.run_dir / 'interrupted.pt'))
                    result['interrupted_checkpoint'] = str(args.run_dir / 'interrupted.pt')
                except Exception as save_error:
                    result['interrupted_checkpoint_error'] = repr(save_error)
        result.update({'status': 'interrupted' if interrupted else 'failed',
                       'error': repr(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(args.run_dir / 'result.json', result)
        print('SESAME_PPO_DR_RESULT='+json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def shutdown_timeout():
                result['shutdown'] = {'completed': False, 'forced_process_exit': True, 'timeout_seconds': 15}
                write_json(args.run_dir / 'result.json', result)
                print('ISAAC_SHUTDOWN_TIMEOUT: results saved; exiting only this process', flush=True)
                os._exit(exit_code)
            watchdog = threading.Timer(15.0, shutdown_timeout)
            watchdog.daemon = True
            watchdog.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            watchdog.cancel()
            result['shutdown'] = {'completed': True, 'forced_process_exit': False}
            write_json(args.run_dir / 'result.json', result)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
