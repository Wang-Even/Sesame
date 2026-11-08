#!/usr/bin/env python3
"""Paired CUDA evaluation of reward-matched PPO, SAC and branching DQN finals.

No checkpoint selection or hyperparameter tuning occurs here. The original
heading evaluator supplies first-episode sampling and diagnostic accounting;
every policy now runs in the same heading-reward environment (coefficient 0.5).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback

from train_ppo import ROOT, disable_git_imports, jsonable, write_json
from train_ppo_study import require_runtime, sha256
from evaluate_heading_ppo import canonical_hash, evaluate_case, validate_saved_case

VARIANTS = ('ppo', 'sac', 'dqn')
ENVIRONMENT_SOURCES = ('sesame_rl/environment.py', 'sesame_rl/dr_environment.py',
                       'sesame_rl/study_environment.py', 'sesame_rl/heading_environment.py')
CODE_PATHS = ('scripts/evaluate_offpolicy.py', 'scripts/evaluate_heading_ppo.py',
              'scripts/train_ppo_study.py', 'scripts/train_ppo.py',
              'scripts/train_offpolicy.py', 'sesame_rl/offpolicy.py',
              'sesame_rl/offpolicy_environment.py', *ENVIRONMENT_SOURCES)


def build_evaluation_protocol(study_dir):
    """Freeze nine completed finals; metadata-only helper for the run queue.

    Run only after all six new runs have succeeded. Existing protocol files
    are preserved and accepted only if their complete payload matches.
    """
    study = Path(study_dir).resolve()
    if not study.is_relative_to(ROOT):
        raise ValueError('Study must remain inside Sesame')
    entries = []
    for variant in VARIANTS:
        for seed in (42, 43, 44):
            run_dir = (ROOT / 'runs/ppo_heading_20261105/training' / f'heading_seed{seed}'
                       if variant == 'ppo' else study / 'training' / f'{variant}_seed{seed}')
            checkpoint = run_dir / 'final.pt'
            result = json.loads((run_dir / 'result.json').read_text())
            if result.get('status') != 'completed' or sha256(checkpoint) != result.get('checkpoint_sha256'):
                raise RuntimeError(f'Cannot freeze incomplete or changed checkpoint: {run_dir}')
            entries.append({'id': f'{variant}_seed{seed}', 'variant': variant, 'train_seed': seed,
                            'run_dir': str(run_dir), 'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint)})
    protocol = {
        'schema_version': 1, 'study_id': study.name,
        'research_question': 'Compare PPO, SAC and branching DQN on identical heading_v1 reward and full DR.',
        'training': {'seeds': [42, 43, 44], 'num_envs': 256, 'transitions_per_run': 2048000,
                     'episode_length_s': 10.0, 'control_dt_s': 0.02, 'heading_penalty_scale': 0.5,
                     'reward_version': 'absolute_heading_v1', 'domain_randomization': 'full_dr unchanged',
                     'observation_dimension': 36, 'action_dimension': 8,
                     'checkpoint_selection': 'final only',
                     'ppo_provenance': 'Existing heading_v1 final models reused; fresh common test scenarios.'},
        'evaluation': {'num_envs': 256, 'episodes_per_seed': 64, 'seeds': [25042, 26042],
                       'conditions': ['nominal', 'randomized'], 'episode_length_s': 10.0,
                       'environment_variant': 'heading_full_dr_for_all_policies',
                       'pairing': 'Exact initial root/joint states and actual domain parameters across all nine policies.',
                       'checkpoint_selection': 'final only',
                       'policy_action': 'PPO saved-normalizer mean; SAC tanh mean; branching DQN greedy per-branch bins.',
                       'primary_metric': 'Forward velocity RMSE; report heading, falls, displacement and smoothness jointly.',
                       'return_semantics': 'Common heading_v1 reward, coefficient 0.5, for all training/evaluation methods.'},
        'checkpoints': entries,
        'representative_sample': {'train_seed': 42, 'evaluation_seed': 25042, 'condition': 'randomized', 'env_id': 0,
                                  'selection': 'First episode; fixed before evaluating, no performance selection.'},
        'limitations': ['DQN means branching dueling Double-DQN with 5 bins per joint; action discretization differs from continuous PPO/SAC.',
                       'Equal real transitions do not imply equal optimizer work or wall-clock compute.',
                       'No hardware or out-of-distribution evaluation.'],
        'frozen_sources': {relative: sha256(ROOT / relative) for relative in CODE_PATHS},
    }
    training_protocol = study / 'protocol.json'
    if training_protocol.exists():
        protocol['training_protocol'] = str(training_protocol)
        protocol['training_protocol_sha256'] = sha256(training_protocol)
    # Validate all source snapshots and matched budgets before writing.
    resolve_checkpoints(protocol, smoke=False)
    output = study / 'evaluation_protocol.json'
    if output.exists():
        if json.loads(output.read_text()) != protocol:
            raise RuntimeError('Existing evaluation protocol differs; refusing to overwrite')
    else:
        write_json(output, protocol)
    (study / 'evaluation_protocol.sha256').write_text(sha256(output) + '  evaluation_protocol.json\n')
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--episodes-per-seed', type=int)
    parser.add_argument('--eval-seeds', type=int, nargs='+')
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('Only cuda:0 is permitted; CPU fallback is disabled')
    if not args.smoke and any(x is not None for x in (args.num_envs, args.episodes_per_seed, args.eval_seeds)):
        parser.error('Overrides require --smoke; final evaluation uses the frozen protocol')
    args.protocol = args.protocol.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error('output-dir must remain inside Sesame')
    if not args.protocol.is_file():
        parser.error('protocol does not exist')
    if (args.output_dir / 'evaluation_manifest.json').exists() and not args.resume:
        parser.error('Existing evaluation requires --resume; never overwrite a manifest')
    return args


def validate_protocol(protocol, args):
    if protocol.get('schema_version') != 1:
        raise ValueError('Unsupported protocol schema version')
    settings = copy.deepcopy(protocol['evaluation'])
    for arg_key, config_key in (('num_envs', 'num_envs'), ('episodes_per_seed', 'episodes_per_seed'), ('eval_seeds', 'seeds')):
        if getattr(args, arg_key) is not None:
            settings[config_key] = getattr(args, arg_key)
    if settings['num_envs'] < 1 or not 1 <= settings['episodes_per_seed'] <= settings['num_envs']:
        raise ValueError('Invalid first-episode sample count')
    if not settings['seeds'] or len(set(settings['seeds'])) != len(settings['seeds']):
        raise ValueError('Unique nonempty test seeds are required')
    if settings['conditions'] != ['nominal', 'randomized'] or settings['episode_length_s'] != 10.0:
        raise ValueError('Both nominal and randomized 10-second conditions are required')
    if settings.get('environment_variant') != 'heading_full_dr_for_all_policies':
        raise ValueError('All methods must use the common heading reward and full-DR environment')
    if protocol['training'].get('heading_penalty_scale') != 0.5:
        raise ValueError('The common heading coefficient must be 0.5')
    entries = protocol['checkpoints']
    ids = [entry['id'] for entry in entries]
    if not ids or len(set(ids)) != len(ids) or any(re.fullmatch(r'[A-Za-z0-9_-]+', key) is None for key in ids):
        raise ValueError('Checkpoint ids must be unique safe path components')
    if any(entry['variant'] not in VARIANTS for entry in entries):
        raise ValueError('Supported methods: ppo, sac, dqn (branching Double DQN)')
    if len({(entry['variant'], entry['train_seed']) for entry in entries}) != len(entries):
        raise ValueError('Duplicate method/training seed')
    if not args.smoke:
        actual = (settings['num_envs'], settings['episodes_per_seed'], settings['seeds'])
        if actual != (256, 64, [25042, 26042]):
            raise ValueError('Final sample: 256 envs, first 64 per test seed, seeds 25042/26042')
        if protocol['training']['seeds'] != [42, 43, 44] or protocol['training']['transitions_per_run'] != 2048000:
            raise ValueError('Three training seeds and matched 2,048,000 true transitions are required')
        expected = {(variant, seed) for variant in VARIANTS for seed in (42, 43, 44)}
        if len(entries) != 9 or {(entry['variant'], entry['train_seed']) for entry in entries} != expected:
            raise ValueError('Final evaluation requires all nine final checkpoints')
    for relative, expected_hash in protocol.get('frozen_sources', {}).items():
        if sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f'Frozen source changed: {relative}')
    return settings


def resolve_checkpoints(protocol, smoke):
    resolved = []
    for entry in protocol['checkpoints']:
        run_dir = Path(entry['run_dir']).resolve()
        checkpoint = Path(entry['checkpoint']).resolve()
        if checkpoint.name != 'final.pt' or checkpoint.parent != run_dir:
            raise ValueError(f'{entry["id"]}: only run final.pt is permitted')
        result_path = run_dir / 'result.json'
        result = json.loads(result_path.read_text())
        if result.get('status') != 'completed':
            raise RuntimeError(f'{entry["id"]}: training incomplete')
        digest = sha256(checkpoint)
        if digest != result.get('checkpoint_sha256') or digest != entry.get('sha256', digest):
            raise RuntimeError(f'{entry["id"]}: checkpoint digest mismatch')
        if result.get('train_seed', result.get('args', {}).get('seed')) != entry['train_seed']:
            raise RuntimeError(f'{entry["id"]}: training seed mismatch')
        if result.get('reward_version') != 'absolute_heading_v1' or result.get('heading_penalty_scale') != 0.5:
            raise RuntimeError(f'{entry["id"]}: rewards are not matched to heading_v1, scale 0.5')
        if not result.get('domain_randomization'):
            raise RuntimeError(f'{entry["id"]}: full domain randomization required')
        if result.get('world_model_enabled') or result.get('imagined_updates'):
            raise RuntimeError(f'{entry["id"]}: no model-based updates in this experiment')
        if not smoke and result.get('environment_transitions_total') != protocol['training']['transitions_per_run']:
            raise RuntimeError(f'{entry["id"]}: training transition budget mismatch')
        config_path = run_dir / ('ppo_config.json' if entry['variant'] == 'ppo' else 'offpolicy_config.json')
        config = json.loads(config_path.read_text())
        if entry['variant'] == 'ppo':
            if result.get('algorithm') != 'PPO' or result.get('variant') != 'heading_v1':
                raise RuntimeError('PPO must be the existing heading_v1 model')
            if config['policy']['class_name'] != 'ActorCritic' or config['seed'] != entry['train_seed']:
                raise RuntimeError('Unexpected PPO architecture or saved seed')
        elif result.get('args', {}).get('algorithm', result.get('variant')) != entry['variant']:
            raise RuntimeError(f'{entry["id"]}: training algorithm mismatch')
        environment_path = run_dir / 'environment_config.json'
        environment = json.loads(environment_path.read_text())
        if environment.get('heading_penalty_scale') != 0.5:
            raise RuntimeError(f'{entry["id"]}: environment heading coefficient mismatch')
        if not smoke and (environment['scene']['num_envs'] != 256 or environment['episode_length_s'] != 10.0):
            raise RuntimeError('Training environment budget/duration mismatch')
        sources = {record['path']: record for record in result['sources']}
        required = set(ENVIRONMENT_SOURCES)
        if entry['variant'] != 'ppo':
            required.update(('sesame_rl/offpolicy.py', 'sesame_rl/offpolicy_environment.py', 'scripts/train_offpolicy.py'))
        if not required <= sources.keys():
            raise RuntimeError(f'{entry["id"]}: missing source provenance')
        checks = {}
        for relative, record in sources.items():
            if sha256(run_dir / 'sources' / relative) != record['sha256']:
                raise RuntimeError(f'{entry["id"]}: changed source snapshot: {relative}')
            if relative in required or relative.startswith('assets/'):
                if sha256(ROOT / relative) != record['sha256']:
                    raise RuntimeError(f'{entry["id"]}: evaluation source/asset differs: {relative}')
                checks[relative] = record['sha256']
        resolved.append({**entry, 'run_dir': str(run_dir), 'checkpoint': str(checkpoint), 'sha256': digest,
                         'config': config, 'config_sha256': sha256(config_path),
                         'training_result_sha256': sha256(result_path),
                         'environment_config_sha256': sha256(environment_path), 'source_checks': checks,
                         'environment_transitions_total': result.get('environment_transitions_total'),
                         'reward_version': result['reward_version'], 'heading_penalty_scale': 0.5})
    return resolved


class OffpolicyAdapter:
    """Expose the existing evaluator's tensor-dict inference interface."""

    def __init__(self, agent):
        self.agent = agent

    def act_inference(self, obs):
        return self.agent.act(obs['policy'], deterministic=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    settings = validate_protocol(protocol, args)
    entries = resolve_checkpoints(protocol, args.smoke)
    core = {'schema_version': 1, 'study_id': protocol['study_id'], 'protocol_sha256': protocol_hash,
            'effective_evaluation': settings, 'smoke': args.smoke,
            'reward_version': 'absolute_heading_v1', 'heading_penalty_scale': 0.5,
            'evaluation_sources': {relative: sha256(ROOT / relative) for relative in CODE_PATHS},
            'resolved_checkpoints': entries}
    signature = canonical_hash(core)
    manifest_path = args.output_dir / 'evaluation_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('manifest_signature') != signature or old.get('core') != core:
            raise RuntimeError('Resume rejected: changed protocol, source, checkpoint or configuration')
    else:
        write_json(manifest_path, {'manifest_signature': signature, 'core': core,
                                  'created_at_utc': datetime.now(timezone.utc).isoformat()})
        shutil.copy2(args.protocol, args.output_dir / 'protocol_snapshot.json')
        for relative in CODE_PATHS:
            target = args.output_dir / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    progress_path = args.output_dir / 'progress.json'
    progress = {'status': 'running', 'protocol_sha256': protocol_hash, 'manifest_signature': signature,
                'total_cases': len(entries) * len(settings['conditions']), 'completed_cases': {}}
    result = {'status': 'running', 'pid': os.getpid(), 'study_id': protocol['study_id'],
              'protocol_sha256': protocol_hash, 'manifest_signature': signature, 'smoke': args.smoke,
              'evaluation': settings, 'reward_version': 'absolute_heading_v1', 'heading_penalty_scale': 0.5,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'git_operations': 'disabled'}
    write_json(args.output_dir / 'result.json', result)
    runtime = ROOT / '.runtime/offpolicy_evaluation'
    for key, relative in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
                          'XDG_DATA_HOME': 'data', 'CUDA_CACHE_PATH': 'cuda_cache'}.items():
        directory = runtime / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    app = env = None
    exit_code = 1
    started = time.monotonic()
    try:
        require_runtime()
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU evaluation prohibited')
        torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        kit_args = ' '.join(['--portable-root', str(runtime / 'kit'),
            f'--/log/file={args.output_dir / "isaac.log"}', '--/app/settings/persistent=false',
            '--/app/telemetry/enabled=false', '--/renderer/multiGpu/enabled=false'])
        app = AppLauncher(headless=True, device=args.device, kit_args=kit_args).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / 'warp_cache')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.heading_environment import SesameHeadingEnv, make_heading_env_cfg
        from sesame_rl.offpolicy import load_policy
        cfg = make_heading_env_cfg(num_envs=settings['num_envs'], seed=settings['seeds'][0], device=args.device,
                                   episode_length_s=settings['episode_length_s'], heading_penalty_scale=0.5)
        write_json(args.output_dir / 'evaluation_environment_config.json', cfg.to_dict())
        env = SesameHeadingEnv(cfg)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        result['gpu'] = torch.cuda.get_device_name(0)
        result['versions'] = {}
        for package in ('torch', 'rsl-rl-lib', 'isaaclab', 'isaaclab-rl', 'isaacsim'):
            try:
                result['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result['versions'][package] = 'not in distribution metadata'
        with torch.inference_mode():
            for entry in entries:
                if sha256(entry['checkpoint']) != entry['sha256']:
                    raise RuntimeError(f'Checkpoint changed: {entry["id"]}')
                if entry['variant'] == 'ppo':
                    actor_config = copy.deepcopy(entry['config']['policy'])
                    actor_config.pop('class_name')
                    with torch.device(args.device):
                        actor = ActorCritic(wrapper.get_observations(), entry['config']['obs_groups'],
                                            wrapper.num_actions, **actor_config).to(args.device)
                    checkpoint = torch.load(entry['checkpoint'], map_location=args.device, weights_only=False)
                    actor.load_state_dict(checkpoint['model_state_dict'])
                    actor.eval()
                    if not all(p.is_cuda for p in actor.parameters()):
                        raise RuntimeError('PPO parameters must be on CUDA')
                    del checkpoint
                else:
                    agent = load_policy(entry['checkpoint'], device=args.device)
                    if agent.algorithm != entry['variant'] or agent.config != entry['config']:
                        raise RuntimeError(f'{entry["id"]}: checkpoint algorithm/config differs from saved configuration')
                    actor = OffpolicyAdapter(agent)
                    del agent
                for condition in settings['conditions']:
                    case_path = args.output_dir / 'cases' / entry['id'] / f'{condition}.json'
                    case_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.resume and case_path.exists():
                        case = json.loads(case_path.read_text())
                        validate_saved_case(case, entry, condition, settings, args, protocol_hash, signature)
                        if case.get('reward_version') != 'absolute_heading_v1' or case.get('heading_penalty_scale') != 0.5:
                            raise RuntimeError('Saved case reward mismatch')
                        resumed = True
                    else:
                        case = evaluate_case(env, wrapper, actor, entry, condition, settings, args,
                                             protocol_hash, signature, torch)
                        case.update(reward_version='absolute_heading_v1', heading_penalty_scale=0.5,
                                    reward_semantics='Common training and evaluation heading_v1 reward for all methods')
                        write_json(case_path, case)
                        resumed = False
                    key = f'{entry["id"]}/{condition}'
                    progress['completed_cases'][key] = {'path': str(case_path), 'sha256': sha256(case_path),
                        'policy_id': entry['id'], 'condition': condition, 'resumed': resumed}
                    write_json(progress_path, progress)
                    print('OFFPOLICY_CASE_COMPLETED=' + json.dumps({'case': key, 'resumed': resumed,
                        'episodes': case['metrics']['episodes'],
                        'velocity_x_rmse_m_s': case['metrics']['velocity_x_rmse_m_s']['mean']}), flush=True)
                del actor
        progress['status'] = 'completed'
        write_json(progress_path, progress)
        result.update(status='completed', case_paths=[value['path'] for value in progress['completed_cases'].values()],
                      cases_completed=len(progress['completed_cases']), pairing_verified_all_cases=True,
                      wall_seconds_before_shutdown=time.monotonic() - started)
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if interrupted:
            exit_code = 130
        result.update(status='interrupted' if interrupted else 'failed', error=repr(exc), traceback=traceback.format_exc())
        progress['status'] = result['status']
        write_json(progress_path, progress)
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir / 'result.json', result)
        print('SESAME_OFFPOLICY_EVALUATION_RESULT=' + json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def shutdown_timeout():
                result['shutdown'] = {'completed': False, 'forced_process_exit': True, 'timeout_seconds': 15}
                write_json(args.output_dir / 'result.json', result)
                os._exit(exit_code)
            timer = threading.Timer(15.0, shutdown_timeout)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            result['shutdown'] = {'completed': True, 'forced_process_exit': False}
            write_json(args.output_dir / 'result.json', result)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
