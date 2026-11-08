#!/usr/bin/env python3
"""CUDA-only Sesame PPO baseline and paired deterministic evaluation.

Use env_isaaclab through its activation hooks. This is a local research baseline,
not the course's official evaluation environment or scoring implementation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.machinery
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
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--steps-per-env', type=int, default=32)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-seed', type=int, default=1042)
    parser.add_argument('--eval-episodes', type=int, default=32)
    parser.add_argument('--episode-length-s', type=float, default=10.0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 64])
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('This launcher requires cuda:0; select a physical GPU outside the process if needed.')
    for name in ('num_envs', 'steps_per_env', 'eval_episodes', 'save_interval'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.iterations <= 0 and not args.eval_only:
        parser.error('iterations must be positive for training')
    if args.eval_only and args.resume is None:
        parser.error('--eval-only requires --resume')
    if args.episode_length_s <= 0 or args.learning_rate <= 0:
        parser.error('episode length and learning rate must be positive')
    if args.run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.run_dir = ROOT / 'runs' / f'ppo_{timestamp}_seed{args.seed}'
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(ROOT):
        parser.error('run-dir must be inside /home/ywwang/sesame')
    if (args.run_dir / 'result.json').exists():
        parser.error('run-dir already contains a result; choose a new directory (also for resume)')
    if args.resume is not None:
        args.resume = args.resume.resolve()
        if not args.resume.is_file():
            parser.error('resume checkpoint does not exist')
    return args


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def disable_git_imports():
    """GitPython calls `git version` on import; prevent that before importing RSL-RL."""
    if 'git' in sys.modules:
        raise RuntimeError('GitPython was imported before the no-Git guard')
    disabled = types.ModuleType('git')
    disabled.__spec__ = importlib.machinery.ModuleSpec('git', loader=None)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Git operations are disabled in this Sesame workspace')
    disabled.Repo = forbidden
    disabled.Git = forbidden
    disabled.__version__ = 'disabled-local-policy'
    sys.modules['git'] = disabled


def training_config(args):
    return {
        'seed': args.seed,
        'num_steps_per_env': args.steps_per_env,
        'save_interval': args.save_interval,
        'logger': 'tensorboard',
        'obs_groups': {'policy': ['policy'], 'critic': ['policy']},
        'policy': {
            'class_name': 'ActorCritic',
            'actor_hidden_dims': args.hidden_dims,
            'critic_hidden_dims': args.hidden_dims,
            'activation': 'elu',
            'actor_obs_normalization': True,
            'critic_obs_normalization': True,
            'init_noise_std': 0.5,
            'noise_std_type': 'log',
        },
        'algorithm': {
            'class_name': 'PPO',
            'clip_param': 0.2,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': args.learning_rate,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'value_loss_coef': 1.0,
            'entropy_coef': 0.005,
            'max_grad_norm': 1.0,
            'use_clipped_value_loss': True,
            'schedule': 'adaptive',
            'desired_kl': 0.01,
            'normalize_advantage_per_mini_batch': False,
        },
    }


def snapshot_sources(run_dir):
    paths = [ROOT / 'scripts/train_ppo.py', ROOT / 'sesame_rl/environment.py',
             ROOT / 'scripts/import_sesame_isaac.py']
    model = ROOT / 'assets/upstream/sesame-ml/src/sesame_ml/assets'
    paths += [model / 'urdf/sesame.urdf', model / 'mjcf/sesame.xml', model / 'calibration/default.yaml']
    paths += sorted((ROOT / 'assets/isaac').glob('**/*')) if (ROOT / 'assets/isaac').exists() else []
    records = []
    for source in paths:
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        target = run_dir / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({'path': str(relative), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'bytes': source.stat().st_size})
    return records


def evaluate(env, wrapper, policy, args, label, torch):
    """Take one complete episode per selected environment, avoiding early-failure selection bias."""
    records = []
    initial_conditions = []
    max_action = 0.0
    trajectory = []
    trajectory_terminal = None
    batch_index = 0
    while len(records) < args.eval_episodes:
        selected_count = min(env.num_envs, args.eval_episodes - len(records))
        batch_seed = args.eval_seed + batch_index
        obs, _ = env.reset(seed=batch_seed)
        env.episode_records.clear()
        # These CPU copies only serialize already-computed CUDA states for reproducibility.
        initial_conditions.append({
            'seed': batch_seed,
            'root_state': env.robot.data.root_state_w[:selected_count].detach().cpu().tolist(),
            'joint_pos': env.robot.data.joint_pos[:selected_count].detach().cpu().tolist(),
        })
        obs = wrapper.get_observations()
        seen = set()
        for control_step in range(int(env.max_episode_length) + 5):
            with torch.inference_mode():
                actions = (torch.zeros((env.num_envs, wrapper.num_actions), device=args.device)
                           if policy is None else policy(obs))
                if not actions.is_cuda or not bool(torch.isfinite(actions).all()):
                    raise RuntimeError(f'{label} produced invalid or non-CUDA actions')
                max_action = max(max_action, float(actions.abs().max().item()))
                if batch_index == 0 and 0 not in seen:
                    trajectory.append({
                        'step': control_step,
                        'time_s': control_step * float(env.step_dt),
                        'root_state_w': env.robot.data.root_state_w[0].detach().cpu().tolist(),
                        'joint_pos_rad': env.robot.data.joint_pos[0].detach().cpu().tolist(),
                        'raw_action': actions[0].detach().cpu().tolist(),
                        'clipped_normalized_action': actions[0].clamp(-1, 1).detach().cpu().tolist(),
                    })
                obs, reward, done, extras = wrapper.step(actions)
                if not bool(torch.isfinite(reward).all()) or not bool(torch.isfinite(obs['policy']).all()):
                    raise RuntimeError(f'{label} produced nonfinite observations/rewards')
            while env.episode_records:
                record = dict(env.episode_records.popleft())
                env_id = int(record['env_id'])
                if env_id < selected_count and env_id not in seen:
                    seen.add(env_id)
                    record.update({'batch_seed': batch_seed, 'policy': label})
                    records.append(record)
                    if batch_index == 0 and env_id == 0:
                        trajectory_terminal = record
            if len(seen) == selected_count:
                break
        if len(seen) != selected_count:
            raise RuntimeError(f'{label}: missing terminal records for {selected_count-len(seen)} episodes')
        batch_index += 1
    records.sort(key=lambda item: (item['batch_seed'], item['env_id']))
    metrics = {'policy': label, 'episodes': len(records), 'seed': args.eval_seed,
               'all_actions_observations_rewards_finite': True, 'max_abs_raw_action': max_action,
               'fall_rate': statistics.mean(float(r['fell']) for r in records),
               'timeout_rate': statistics.mean(float(r['timeout']) for r in records)}
    for key in ('distance_x_m', 'avg_speed_m_s', 'duration_s', 'return'):
        values = [float(r[key]) for r in records]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f'Nonfinite episode metadata: {key}')
        metrics[key] = {'mean': statistics.mean(values), 'std': statistics.pstdev(values),
                        'min': min(values), 'max': max(values)}
    write_json(args.run_dir / f'evaluation_{label}.json', {
        'scope': 'local task evaluation; not official course scoring',
        'metrics': metrics, 'episodes': records, 'initial_conditions': initial_conditions,
    })
    write_json(args.run_dir / f'eval_trajectory_{label}.json', {
        'policy': label, 'batch_seed': args.eval_seed, 'env_id': 0,
        'robot_joint_order': list(env.robot.joint_names),
        'action_order': ['R1', 'R2', 'L1', 'L2', 'R4', 'R3', 'L3', 'L4'],
        'state_timing': 'pre-step, world coordinates; root_state=[xyz,quat_wxyz,linear_velocity,angular_velocity]',
        'action_semantics': 'wrapper-clipped normalized residual; further target-angle and slew limits apply inside env',
        'samples': trajectory, 'terminal_record': trajectory_terminal,
    })
    return metrics, initial_conditions


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'scope': 'local Sesame PPO baseline; not official course evaluation',
              'run_dir': str(args.run_dir), 'args': vars(args), 'git_operations': 'disabled'}
    write_json(args.run_dir / 'result.json', result)
    runtime = ROOT / '.runtime/ppo'
    for key, subdir in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
                        'XDG_DATA_HOME': 'data', 'CUDA_CACHE_PATH': 'cuda_cache'}.items():
        path = runtime / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    app = env = runner = None
    exit_code = 1
    started = time.monotonic()
    try:
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; CPU fallback is prohibited')
        if int(os.getenv('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('This baseline is a single-GPU run; unset distributed launch variables')
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
        from sesame_rl.environment import SesameEnv, make_env_cfg
        # Defense in depth: never call the GitPython-based snapshot helper.
        runner_module.store_code_state = lambda *unused_args, **unused_kwargs: []

        class JsonRunner(runner_module.OnPolicyRunner):
            def __init__(self, *runner_args, **runner_kwargs):
                super().__init__(*runner_args, **runner_kwargs)
                self.git_status_repos = []
                self.logger_type = 'tensorboard'

            def add_git_repo_to_log(self, repo_file_path):
                raise RuntimeError('Git logging is disabled; source hashes are saved instead')

            def log(self, locs, *log_args, **log_kwargs):
                log_locs = dict(locs)
                log_locs['ep_infos'] = [info for info in locs['ep_infos'] if info]
                super().log(log_locs, *log_args, **log_kwargs)
                row = {'iteration': int(locs['it']) + 1, 'environment_transitions': self.tot_timesteps,
                       'collection_seconds': float(locs['collection_time']),
                       'update_seconds': float(locs['learn_time']),
                       'learning_rate': float(self.alg.learning_rate),
                       'losses': {k: float(v) for k, v in locs['loss_dict'].items()},
                       'mean_episode_return': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
                       'mean_episode_length_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None}
                episode_metrics = {}
                for info in locs['ep_infos']:
                    for key, value in info.items():
                        scalar = float(value.detach().float().mean().item()) if torch.is_tensor(value) else float(value)
                        episode_metrics.setdefault(key, []).append(scalar)
                row['episode_metrics'] = {key: statistics.mean(values) for key, values in episode_metrics.items()}
                for value in row['losses'].values():
                    if not math.isfinite(value):
                        raise RuntimeError('PPO loss became nonfinite')
                with (args.run_dir / 'training.jsonl').open('a') as output:
                    output.write(json.dumps(row, allow_nan=False) + '\n')

            def save(self, path, infos=None):
                info = dict(infos or {})
                info.update({'environment_transitions': self.tot_timesteps,
                             'seed': args.seed, 'policy_config_path': str(args.run_dir / 'ppo_config.json')})
                super().save(path, info)

        config = training_config(args)
        # Checkpoint architecture/config must remain the same when resuming or evaluating.
        if args.resume is not None:
            saved_config_path = args.resume.parent / 'ppo_config.json'
            if not saved_config_path.exists():
                raise RuntimeError(f'Resume needs saved PPO config: {saved_config_path}')
            saved_config = json.loads(saved_config_path.read_text())
            config['policy'] = saved_config['policy']
            config['obs_groups'] = saved_config['obs_groups']
            if not args.eval_only:
                config['algorithm'] = saved_config['algorithm']
        write_json(args.run_dir / 'ppo_config.json', config)
        cfg = make_env_cfg(num_envs=args.num_envs, seed=args.seed, device=args.device,
                           episode_length_s=args.episode_length_s)
        if args.resume is not None:
            previous_source = args.resume.parent / 'sources/sesame_rl/environment.py'
            current_source = ROOT / 'sesame_rl/environment.py'
            if not previous_source.is_file() or previous_source.read_bytes() != current_source.read_bytes():
                raise RuntimeError('Resume requires the unchanged environment source saved with the checkpoint')
            previous_env_path = args.resume.parent / 'environment_config.json'
            previous_env = json.loads(previous_env_path.read_text())
            if previous_env.get('episode_length_s') != args.episode_length_s:
                raise RuntimeError('Resume episode-length-s must match the saved environment configuration')
        write_json(args.run_dir / 'environment_config.json', cfg.to_dict())
        env = SesameEnv(cfg)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapper.get_observations()
        if not obs['policy'].is_cuda or not bool(torch.isfinite(obs['policy']).all()):
            raise RuntimeError('Environment observations must be finite CUDA tensors')
        with torch.device(args.device):
            runner = JsonRunner(wrapper, copy.deepcopy(config), log_dir=str(args.run_dir), device=args.device)
        if not all(p.is_cuda for p in runner.alg.policy.parameters()):
            raise RuntimeError('Policy parameters are not entirely on CUDA')
        result['sources'] = snapshot_sources(args.run_dir)
        result['versions'] = {}
        for package in ('torch', 'rsl-rl-lib', 'isaaclab', 'isaaclab-rl', 'isaacsim', 'gymnasium'):
            try:
                result['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result['versions'][package] = 'not in distribution metadata'
        result['gpu'] = torch.cuda.get_device_name(0)
        result['observation_shape'] = list(obs['policy'].shape)
        result['action_count'] = wrapper.num_actions
        write_json(args.run_dir / 'result.json', result)
        if args.resume is not None:
            checkpoint_info = runner.load(str(args.resume), load_optimizer=not args.eval_only, map_location=args.device)
            runner.tot_timesteps = int((checkpoint_info or {}).get('environment_transitions', 0))
            runner.current_learning_iteration += 1
        if not args.eval_only:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final_path = args.run_dir / 'final.pt'
        runner.save(str(final_path))
        if runner.writer is not None:
            runner.writer.flush()
        # Reload into a fresh CUDA policy, including the trained observation normalizers.
        policy_config = copy.deepcopy(config['policy'])
        policy_config.pop('class_name')
        with torch.device(args.device):
            reloaded = ActorCritic(wrapper.get_observations(), config['obs_groups'], wrapper.num_actions,
                                   **policy_config).to(args.device)
        saved = torch.load(final_path, map_location=args.device, weights_only=False)
        reloaded.load_state_dict(saved['model_state_dict'])
        reloaded.eval()
        runner.eval_mode()
        with torch.inference_mode():
            probe = wrapper.get_observations()
            original = runner.alg.policy.act_inference(probe)
            restored = reloaded.act_inference(probe)
            reload_error = float((original-restored).abs().max().item())
            if reload_error != 0.0 or not bool(torch.isfinite(restored).all()):
                raise RuntimeError('Saved policy did not reload identically')
        # RSL-RL collects rollouts in inference_mode; Isaac's lazily-created
        # state buffers inherit that mode, so resets must use it as well.
        with torch.inference_mode():
            zero_metrics, zero_initial = evaluate(env, wrapper, None, args, 'zero_residual', torch)
            ppo_metrics, ppo_initial = evaluate(env, wrapper, reloaded.act_inference, args, 'ppo', torch)
        initial_conditions_equal = zero_initial == ppo_initial
        if not initial_conditions_equal:
            raise RuntimeError('Paired evaluation initial conditions differ despite matching seeds')
        result.update({'status': 'completed', 'checkpoint': str(final_path),
            'checkpoint_sha256': hashlib.sha256(final_path.read_bytes()).hexdigest(),
            'environment_transitions_total': runner.tot_timesteps,
            'training_iterations_this_run': 0 if args.eval_only else args.iterations,
            'reload_max_abs_action_error': reload_error,
            'evaluation': {'identical_initial_conditions': initial_conditions_equal,
                           'zero_residual': zero_metrics, 'ppo': ppo_metrics},
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
        result.update({'status': 'interrupted' if interrupted else 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        if runner is not None and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(args.run_dir / 'result.json', result)
        print('SESAME_PPO_RESULT=' + json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def shutdown_timeout():
                result['shutdown'] = {'completed': False, 'forced_process_exit': True, 'timeout_seconds': 15}
                write_json(args.run_dir / 'result.json', result)
                print('ISAAC_SHUTDOWN_TIMEOUT: saved results; terminating only this process', flush=True)
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
