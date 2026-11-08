#!/usr/bin/env python3
"""Train Dyna-style PPO with full domain randomization on CUDA; final-only evaluation."""
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

from train_ppo import ROOT, disable_git_imports, jsonable, snapshot_sources, training_config, write_json

ISAAC_ENV = Path('/home/ubuntu/miniforge3/envs/env_isaaclab')
BASELINE_REFERENCE = ROOT / 'runs/ppo_baseline_20261102_seed42'
DR_REFERENCE = ROOT / 'runs/ppo_dr_20261102_seed42'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_sources():
    checks = {}
    for reference, relative in ((BASELINE_REFERENCE, 'sesame_rl/environment.py'),
                                (DR_REFERENCE, 'sesame_rl/dr_environment.py'),
                                (BASELINE_REFERENCE, 'scripts/train_ppo.py'),
                                (DR_REFERENCE, 'scripts/train_ppo_dr.py')):
        current = ROOT / relative
        frozen = reference / 'sources' / relative
        if current.read_bytes() != frozen.read_bytes():
            raise RuntimeError(f'Frozen source changed: {relative}')
        checks[relative] = {'sha256': sha256(current), 'reference': str(frozen), 'matches': True}
    return checks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('dyna_ppo',), default='dyna_ppo')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dyna-config', required=True, type=Path)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--steps-per-env', type=int, default=32)
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('Only cuda:0 is allowed; CPU fallback is prohibited')
    if min(args.num_envs, args.iterations, args.steps_per_env) <= 0:
        parser.error('Training sizes must be positive')
    if not args.smoke and (args.num_envs, args.iterations, args.steps_per_env) != (256, 250, 32):
        parser.error('Study training is fixed at 256 environments, 250 updates and 32 rollout steps; use --smoke for probes')
    if not args.smoke and args.seed not in (42, 43, 44):
        parser.error('Study training seeds are fixed at 42, 43 and 44')
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(ROOT):
        parser.error('run-dir must remain inside the Sesame workspace')
    if (args.run_dir / 'result.json').exists():
        parser.error('run-dir already contains a result; do not overwrite a study run')
    args.dyna_config = args.dyna_config.resolve()
    if not args.dyna_config.is_file():
        parser.error('dyna-config must be a saved JSON configuration')
    args.save_interval = 50
    args.hidden_dims = [128, 128, 64]
    args.learning_rate = 3e-4
    return args


def require_runtime():
    if Path(sys.executable).resolve() != (ISAAC_ENV / 'bin/python').resolve():
        raise RuntimeError('Run this script with env_isaaclab Python')
    if Path(os.environ.get('CONDA_PREFIX', '/nonexistent')).resolve() != ISAAC_ENV.resolve():
        raise RuntimeError('Activate env_isaaclab via its Conda hooks before launching')


def snapshot_study_sources(run_dir):
    records = snapshot_sources(run_dir)
    for relative in ('scripts/train_ppo_dr.py', 'scripts/train_ppo_study.py',
                     'sesame_rl/dr_environment.py', 'sesame_rl/study_environment.py',
                     'scripts/train_dyna_ppo.py', 'sesame_rl/dyna_world_model.py', 'sesame_rl/dyna_ppo.py'):
        source = ROOT / relative
        destination = run_dir / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({'path': relative, 'sha256': sha256(source), 'bytes': source.stat().st_size})
    return records


def domain_readback(env, torch):
    if hasattr(env, 'get_domain_snapshot'):
        return env.get_domain_snapshot(), env.verify_domain_parameters()
    view = env.robot.root_physx_view
    properties = {
        'masses': view.get_masses().to(env.device), 'inertias': view.get_inertias().to(env.device),
        'materials': view.get_material_properties().to(env.device),
        'stiffness': view.get_dof_stiffnesses().to(env.device),
        'damping': view.get_dof_dampings().to(env.device),
        'effort_limits': view.get_dof_max_forces().to(env.device),
        'velocity_limits': view.get_dof_max_velocities().to(env.device),
    }
    if not all(bool(torch.isfinite(value).all()) for value in properties.values()):
        raise RuntimeError('Nonfinite nominal physical coefficients')
    if not bool((properties['masses'] > 0).all()):
        raise RuntimeError('Nominal dynamic links must have positive mass')
    for key, expected in (('stiffness', 5.0), ('damping', 0.0025),
                          ('effort_limits', 0.18), ('velocity_limits', 10.471975512)):
        if not torch.allclose(properties[key], torch.full_like(properties[key], expected), rtol=1e-5, atol=1e-8):
            raise RuntimeError(f'Unexpected nominal {key}')
    snapshot = {'enabled': [False]*env.num_envs,
                'physx_readback': {key: value.detach().cpu().tolist() for key, value in properties.items()}}
    return snapshot, {'passed': True, 'scope': 'finite positive-mass nominal PhysX coefficients and servo values'}


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'variant': args.variant, 'train_seed': args.seed,
              'smoke': args.smoke, 'fresh_from_scratch': True, 'checkpoint_selection': 'final only',
              'run_dir': str(args.run_dir), 'args': vars(args), 'git_operations': 'disabled',
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.run_dir / 'result.json', result)
    runtime = ROOT / '.runtime/dyna_ppo'
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
        require_runtime()
        result['frozen_sources'] = validate_frozen_sources()
        config = training_config(args)
        baseline_config = json.loads((BASELINE_REFERENCE / 'ppo_config.json').read_text())
        for section in ('policy', 'algorithm', 'obs_groups'):
            if config[section] != baseline_config[section]:
                raise RuntimeError(f'Study must preserve fixed-baseline PPO {section}')
        write_json(args.run_dir / 'ppo_config.json', config)
        dyna_config = json.loads(args.dyna_config.read_text())
        write_json(args.run_dir / 'dyna_config.json', dyna_config)
        result['dyna_config_sha256'] = sha256(args.run_dir / 'dyna_config.json')
        result['algorithm'] = 'DynaPPO'
        result['training_environment_variant'] = 'full_dr'
        result['scope'] = 'Local Dyna-style PPO implementation, not an exact replication of a named paper; continuing one-step imagined transitions, no learned termination model'
        effective_config = copy.deepcopy(config)
        effective_config['algorithm'].update({'class_name': 'DynaPPO', 'dyna_config': dyna_config, 'dyna_seed': args.seed})
        write_json(args.run_dir / 'effective_training_config.json', effective_config)
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; no CPU simulation/model fallback')
        if int(os.getenv('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('This study uses one CUDA GPU per process')
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
        runner_module.store_code_state = lambda *unused, **unused_kwargs: []
        from sesame_rl.dyna_ppo import DynaPPO
        runner_module.DynaPPO = DynaPPO
        # Keep the full-DR task, observation, controller and reward unchanged.
        from sesame_rl.dr_environment import SesameDREnv as Env, make_env_cfg
        cfg = make_env_cfg(num_envs=args.num_envs, seed=args.seed, device=args.device, episode_length_s=10.0)
        write_json(args.run_dir / 'environment_config.json', cfg.to_dict())
        env = Env(cfg)
        if args.variant != 'baseline':
            env.set_domain_randomization(True)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapper.get_observations()
        if not obs['policy'].is_cuda or not bool(torch.isfinite(obs['policy']).all()):
            raise RuntimeError('Invalid non-CUDA/nonfinite observations')
        initial_domains, initial_validation = domain_readback(env, torch)
        write_json(args.run_dir / 'domain_samples_initial.json', initial_domains)
        write_json(args.run_dir / 'domain_validation_initial.json', initial_validation)

        class JsonRunner(runner_module.OnPolicyRunner):
            def __init__(self, *runner_args, **runner_kwargs):
                super().__init__(*runner_args, **runner_kwargs)
                self.git_status_repos = []
                self.logger_type = 'tensorboard'

            def add_git_repo_to_log(self, unused):
                raise RuntimeError('Git logging is disabled')

            def log(self, locs, *extra_args, **extra_kwargs):
                filtered = dict(locs)
                filtered['ep_infos'] = [item for item in locs['ep_infos'] if item]
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
                                 'variant': args.variant, 'source_run_dir': str(args.run_dir)})
                saved = {'model_state_dict': self.alg.policy.state_dict(),
                         'optimizer_state_dict': self.alg.optimizer.state_dict(),
                         'iter': self.current_learning_iteration, 'infos': metadata,
                         'dyna_state_dict': self.alg.dyna_state_dict()}
                torch.save(saved, path)

        with torch.device(args.device):
            runner = JsonRunner(wrapper, copy.deepcopy(effective_config), log_dir=str(args.run_dir), device=args.device)
        if not all(parameter.is_cuda for parameter in runner.alg.policy.parameters()):
            raise RuntimeError('Policy parameters left CUDA')
        result['sources'] = snapshot_study_sources(args.run_dir)
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
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final = args.run_dir / 'final.pt'
        runner.save(str(final))
        if runner.writer is not None:
            runner.writer.flush()
        final_domains, final_validation = domain_readback(env, torch)
        write_json(args.run_dir / 'domain_samples_after_training.json', final_domains)
        write_json(args.run_dir / 'domain_validation_after_training.json', final_validation)
        if args.variant == 'baseline' and initial_domains != final_domains:
            raise RuntimeError('Baseline physical coefficients changed during training')
        actor_config = copy.deepcopy(config['policy'])
        actor_config.pop('class_name')
        with torch.device(args.device):
            restored = ActorCritic(wrapper.get_observations(), config['obs_groups'], wrapper.num_actions,
                                   **actor_config).to(args.device)
        checkpoint = torch.load(final, map_location=args.device, weights_only=False)
        restored.load_state_dict(checkpoint['model_state_dict'])
        restored.eval()
        runner.eval_mode()
        with torch.inference_mode():
            probe = wrapper.get_observations()
            reloaded_actions = restored.act_inference(probe)
            expected_actions = runner.alg.policy.act_inference(probe)
            reload_error = float((reloaded_actions-expected_actions).abs().max().item())
            if reload_error != 0.0 or not bool(torch.isfinite(reloaded_actions).all()):
                raise RuntimeError('Final checkpoint reload does not reproduce finite CUDA inference exactly')
        dyna_state = checkpoint['dyna_state_dict']
        from sesame_rl.dyna_world_model import DynaWorldModel
        restored_model = DynaWorldModel(args.device, args.seed, dyna_state['world_model']['config'])
        restored_model.load_state_dict(dyna_state['world_model'])
        with torch.inference_mode():
            model_history = runner.alg.world_model.sample_starts(64)
            model_actions = torch.zeros((64, 8), device=args.device)
            model_x = torch.cat((model_history.flatten(1), model_actions), dim=1)
            original_scales = runner.alg.world_model._scales()
            restored_scales = restored_model._scales()
            expected_prediction = runner.alg.world_model._predict(model_x, original_scales[0])
            actual_prediction = restored_model._predict(model_x, restored_scales[0])
            model_reload_error = float((actual_prediction - expected_prediction).abs().max().item())
            if model_reload_error != 0 or not bool(torch.isfinite(actual_prediction).all()):
                raise RuntimeError('World-model reload does not reproduce CUDA predictions exactly')
        result['world_model_reload_max_abs_prediction_error'] = model_reload_error
        result['dyna_training_summary'] = runner.alg.training_summary()
        if not args.smoke and result['dyna_training_summary']['imagined_optimizer_steps'] <= 0:
            raise RuntimeError('No imagined PPO update occurred: this cannot be reported as completed Dyna training')
        expected_transitions = args.num_envs * args.steps_per_env * args.iterations
        if runner.tot_timesteps != expected_transitions:
            raise RuntimeError('Training transition count does not match the fixed budget')
        result.update({'status': 'completed', 'checkpoint': str(final), 'checkpoint_sha256': sha256(final),
                       'training_iterations_this_run': args.iterations,
                       'environment_transitions_total': runner.tot_timesteps,
                       'dyna_checkpoint_keys': sorted(dyna_state),
                       'reload_max_abs_action_error': reload_error,
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
        print('SESAME_PPO_STUDY_RESULT='+json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def shutdown_timeout():
                result['shutdown'] = {'completed': False, 'forced_process_exit': True, 'timeout_seconds': 15}
                write_json(args.run_dir / 'result.json', result)
                print('ISAAC_SHUTDOWN_TIMEOUT: results saved; exiting only this process', flush=True)
                os._exit(exit_code)
            timer = threading.Timer(15.0, shutdown_timeout)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            result['shutdown'] = {'completed': True, 'forced_process_exit': False}
            write_json(args.run_dir / 'result.json', result)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
