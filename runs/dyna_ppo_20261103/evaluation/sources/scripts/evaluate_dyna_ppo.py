#!/usr/bin/env python3
"""Evaluate baseline, full-DR, and Dyna-PPO final checkpoints on common CUDA physics.

Case files contain both fixed comparison seeds, 3042/4042. These were excluded
from training, but observed in the prior DR study; this is not a new blind test.
Resuming requires unchanged protocol,
checkpoint hashes, effective evaluation settings and evaluator/environment sources.
All policies use the full-DR evaluation environment, irrespective of training arm.
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
import re
import shutil
import statistics
import sys
import threading
import time
import traceback

from train_ppo import ROOT, disable_git_imports, jsonable, write_json
from train_ppo_study import require_runtime, sha256

VARIANTS = ('baseline', 'full_dr', 'dyna_ppo')
DYNA_SOURCES = ('scripts/train_dyna_ppo.py', 'sesame_rl/dyna_ppo.py',
                'sesame_rl/dyna_world_model.py')
STUDY_METRICS = (
    'velocity_x_rmse_m_s', 'velocity_xy_tracking_rmse_m_s',
    'heading_rms_rad', 'heading_max_abs_rad', 'lateral_mean_abs_m', 'lateral_rms_m',
    'applied_target_rate_rms_rad_s', 'applied_target_acceleration_rms_rad_s2',
    'actual_joint_acceleration_rms_rad_s2',
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--num-envs', type=int, help='Smoke-only override')
    parser.add_argument('--episodes-per-seed', type=int, help='Smoke-only override')
    parser.add_argument('--eval-seeds', type=int, nargs='+', help='Smoke-only override')
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('Only cuda:0 is permitted; CPU fallback is disabled')
    if not args.smoke and any(value is not None for value in (args.num_envs, args.episodes_per_seed, args.eval_seeds)):
        parser.error('Evaluation overrides require --smoke; final settings come only from frozen protocol')
    args.protocol = args.protocol.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error('output-dir must remain inside the Sesame workspace')
    if not args.protocol.is_file():
        parser.error('protocol does not exist')
    if (args.output_dir / 'evaluation_manifest.json').exists() and not args.resume:
        parser.error('Existing evaluation output requires --resume; never overwrite its manifest')
    return args


def validate_protocol(protocol, args):
    if protocol.get('schema_version') != 1:
        raise ValueError('Unsupported protocol schema version')
    evaluation = copy.deepcopy(protocol['evaluation'])
    for arg_key, config_key in (('num_envs', 'num_envs'), ('episodes_per_seed', 'episodes_per_seed'),
                                ('eval_seeds', 'seeds')):
        if getattr(args, arg_key) is not None:
            evaluation[config_key] = getattr(args, arg_key)
    if evaluation['num_envs'] < 1 or not 1 <= evaluation['episodes_per_seed'] <= evaluation['num_envs']:
        raise ValueError('Evaluate the first positive number of environments without exceeding num_envs')
    if not evaluation['seeds'] or len(set(evaluation['seeds'])) != len(evaluation['seeds']):
        raise ValueError('Evaluation seeds must be unique and nonempty')
    if evaluation['conditions'] != ['nominal', 'randomized']:
        raise ValueError('Both nominal and randomized domains, in the frozen order, are required')
    if evaluation['episode_length_s'] <= 0:
        raise ValueError('Invalid episode duration')
    entries = protocol['checkpoints']
    ids = [entry['id'] for entry in entries]
    if not ids or len(set(ids)) != len(ids) or any(re.fullmatch(r'[A-Za-z0-9_-]+', key) is None for key in ids):
        raise ValueError('Checkpoint ids must be unique safe path components')
    if any(entry['variant'] not in VARIANTS for entry in entries):
        raise ValueError('Unknown training variant')
    if not args.smoke:
        actual = (evaluation['num_envs'], evaluation['episodes_per_seed'], evaluation['seeds'], evaluation['episode_length_s'])
        if actual != (256, 64, [3042, 4042], 10.0):
            raise ValueError('Final evaluation is frozen at 256 envs, first 64 episodes, seeds 3042/4042, 10 seconds')
        expected = {(variant, seed) for variant in VARIANTS for seed in (42, 43, 44)}
        if len(entries) != 9 or {(entry['variant'], entry['train_seed']) for entry in entries} != expected:
            raise ValueError('Final evaluation requires all three seeds of all three arms, exactly nine checkpoints')
        if evaluation.get('environment_variant') != 'full_dr_for_all_policies':
            raise ValueError('All training arms must share the full-DR evaluation distribution')
    for relative, expected_hash in protocol.get('frozen_sources', {}).items():
        if sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f'Preregistered frozen source changed: {relative}')
    return evaluation


def resolve_checkpoints(protocol, smoke):
    reference = json.loads(Path(protocol['training']['ppo_config_reference']).read_text())
    resolved = []
    for entry in protocol['checkpoints']:
        run_dir = Path(entry['run_dir']).resolve()
        checkpoint = Path(entry['checkpoint']).resolve()
        if checkpoint.name != 'final.pt' or checkpoint.parent != run_dir:
            raise ValueError(f'{entry["id"]}: only the declared run final.pt may be evaluated')
        result_path = run_dir / 'result.json'
        result = json.loads(result_path.read_text())
        if result.get('status') != 'completed':
            raise RuntimeError(f'{entry["id"]}: training has not completed successfully')
        actual_hash = sha256(checkpoint)
        if actual_hash != result.get('checkpoint_sha256'):
            raise RuntimeError(f'{entry["id"]}: final checkpoint hash differs from training result')
        if entry.get('sha256') is not None and actual_hash != entry['sha256']:
            raise RuntimeError(f'{entry["id"]}: final checkpoint hash differs from protocol')
        result_seed = result.get('train_seed', result.get('args', {}).get('seed'))
        if result_seed != entry['train_seed']:
            raise RuntimeError(f'{entry["id"]}: training seed mismatch')
        result_variant = result.get('variant', result.get('args', {}).get('variant'))
        if result_variant is not None and result_variant != entry['variant']:
            raise RuntimeError(f'{entry["id"]}: training variant mismatch')
        config_path = run_dir / 'ppo_config.json'
        config = json.loads(config_path.read_text())
        for section in ('policy', 'algorithm', 'obs_groups'):
            if config[section] != reference[section]:
                raise RuntimeError(f'{entry["id"]}: PPO {section} differs from protocol reference')
        if config['policy']['class_name'] != 'ActorCritic':
            raise RuntimeError('Only the declared feed-forward ActorCritic architecture is supported')
        if config['seed'] != entry['train_seed']:
            raise RuntimeError(f'{entry["id"]}: saved PPO seed mismatch')
        env_config_path = run_dir / 'environment_config.json'
        env_config = json.loads(env_config_path.read_text())
        if not smoke:
            training = protocol['training']
            if result.get('environment_transitions_total') != training['transitions_per_run']:
                raise RuntimeError(f'{entry["id"]}: transition budget mismatch')
            if result.get('training_iterations_this_run') != training['iterations']:
                raise RuntimeError(f'{entry["id"]}: update budget mismatch')
            if config['num_steps_per_env'] != training['steps_per_env']:
                raise RuntimeError(f'{entry["id"]}: rollout length mismatch')
            if env_config['scene']['num_envs'] != training['num_envs']:
                raise RuntimeError(f'{entry["id"]}: training environment count mismatch')
            if env_config['episode_length_s'] != training['episode_length_s']:
                raise RuntimeError(f'{entry["id"]}: training episode duration mismatch')
        source_records = {record['path']: record for record in result['sources']}
        required = ['sesame_rl/environment.py']
        if entry['variant'] != 'baseline':
            required.append('sesame_rl/dr_environment.py')
        if entry['variant'] == 'dyna_ppo':
            required.extend(DYNA_SOURCES)
        for relative in required:
            if relative not in source_records:
                raise RuntimeError(f'{entry["id"]}: missing source provenance for {relative}')
        checked = {}
        for relative, record in source_records.items():
            snapshot = run_dir / 'sources' / relative
            if sha256(snapshot) != record['sha256']:
                raise RuntimeError(f'{entry["id"]}: source snapshot hash mismatch: {relative}')
            if relative in required or relative.startswith('assets/'):
                if sha256(ROOT / relative) != record['sha256']:
                    raise RuntimeError(f'{entry["id"]}: common evaluation asset/environment differs: {relative}')
                checked[relative] = record['sha256']
        dyna_audit = {}
        if entry['variant'] == 'dyna_ppo':
            dyna_path = run_dir / 'dyna_config.json'
            dyna_config = json.loads(dyna_path.read_text())
            dyna_hash = sha256(dyna_path)
            if dyna_hash != result.get('dyna_config_sha256'):
                raise RuntimeError(f'{entry["id"]}: Dyna configuration digest differs from training result')
            reference_path = protocol['training'].get('dyna_config_reference')
            if reference_path and dyna_config != json.loads(Path(reference_path).read_text()):
                raise RuntimeError(f'{entry["id"]}: Dyna configuration differs from preregistered reference')
            if result.get('algorithm') != 'DynaPPO' or result.get('training_environment_variant') != 'full_dr':
                raise RuntimeError(f'{entry["id"]}: training must identify DynaPPO on full-DR physics')
            effective_path = run_dir / 'effective_training_config.json'
            effective = json.loads(effective_path.read_text())
            expected_effective = copy.deepcopy(config)
            expected_effective['algorithm'].update({'class_name': 'DynaPPO', 'dyna_config': dyna_config,
                                                    'dyna_seed': entry['train_seed']})
            if effective != expected_effective:
                raise RuntimeError(f'{entry["id"]}: actual algorithm config is not the declared Dyna augmentation')
            counters = result.get('dyna_training_summary', {})
            if not smoke and counters.get('imagined_optimizer_steps', 0) <= 0:
                raise RuntimeError(f'{entry["id"]}: no actual imagined policy update is recorded')
            dyna_audit = {'dyna_config': dyna_config, 'dyna_config_sha256': dyna_hash,
                          'effective_training_config_sha256': sha256(effective_path),
                          'actual_training_algorithm': result['algorithm'],
                          'training_summary': counters,
                          'declared_checkpoint_keys': result.get('dyna_checkpoint_keys'),
                          'source_sha256': {relative: source_records[relative]['sha256'] for relative in DYNA_SOURCES}}
        resolved.append({
            **entry, 'checkpoint': str(checkpoint), 'run_dir': str(run_dir),
            'sha256': actual_hash, 'ppo_config': config, 'ppo_config_sha256': sha256(config_path),
            'environment_config_sha256': sha256(env_config_path), 'source_checks': checked,
            'environment_transitions_total': result.get('environment_transitions_total'),
            'training_iterations': result.get('training_iterations_this_run'),
            **({'dyna_audit': dyna_audit} if dyna_audit else {}),
        })
    return resolved


def audit_dyna_checkpoint(checkpoint, entry, smoke, torch):
    """Audit auxiliary CUDA weights/counters without executing the world model."""
    state = checkpoint.get('dyna_state_dict')
    if not isinstance(state, dict) or not state:
        raise RuntimeError('Dyna checkpoint lacks its auxiliary training state')
    audit = entry['dyna_audit']
    if sorted(state) != audit['declared_checkpoint_keys']:
        raise RuntimeError('Dyna checkpoint schema differs from the saved training result')
    if state.get('version') != 1 or state.get('seed') != entry['train_seed']:
        raise RuntimeError('Dyna checkpoint version/seed mismatch')
    if not isinstance(state.get('config'), dict) or any(
        state['config'].get(key) != value for key, value in audit['dyna_config'].items()
    ):
        raise RuntimeError('Dyna checkpoint effective configuration differs from declared inputs')
    counters = audit['training_summary']
    counter_keys = ('real_transitions_seen', 'real_updates', 'real_optimizer_steps',
                    'model_optimizer_steps', 'model_member_optimizer_steps', 'model_fit_calls',
                    'imagined_generated', 'imagined_accepted', 'imagined_optimizer_steps', 'imagined_update_calls')
    for key in counter_keys:
        if not isinstance(state.get(key), int) or state[key] < 0 or state[key] != counters.get(key):
            raise RuntimeError(f'Dyna checkpoint counter differs from training result: {key}')
    if state['real_transitions_seen'] != entry['environment_transitions_total'] or state['real_updates'] != entry['training_iterations']:
        raise RuntimeError('Dyna internal real-data accounting differs from runner budget')
    ppo = entry['ppo_config']['algorithm']
    if state['real_optimizer_steps'] != state['real_updates'] * ppo['num_learning_epochs'] * ppo['num_mini_batches']:
        raise RuntimeError('Dyna real PPO optimizer-step count differs from unchanged PPO schedule')
    if state['model_member_optimizer_steps'] != 3 * state['model_optimizer_steps']:
        raise RuntimeError('World-model per-member optimizer count mismatch')
    if state['imagined_accepted'] > state['imagined_generated'] or state['imagined_update_calls'] > state['real_updates']:
        raise RuntimeError('Invalid imagined data/update accounting')
    dictionaries = []
    def visit(value, path='dyna_state_dict'):
        if isinstance(value, dict):
            dictionaries.append((path, value))
            for key, child in value.items():
                if isinstance(child, dict):
                    visit(child, path+'.'+str(key))
    visit(state)
    step_values = [value['imagined_optimizer_steps'] for _, value in dictionaries
                   if 'imagined_optimizer_steps' in value]
    expected_steps = audit['training_summary']['imagined_optimizer_steps']
    if not step_values or any(not isinstance(value, int) or value != expected_steps for value in step_values):
        raise RuntimeError('Actual Dyna checkpoint update counter differs from its training result')
    if not smoke and expected_steps <= 0:
        raise RuntimeError('Formal Dyna checkpoint must contain positive imagined optimizer updates')
    world_models = [(path, value) for path, value in dictionaries
                    if {'models', 'input_moments', 'target_moments', 'fit_calls'} <= value.keys()]
    if len(world_models) != 1:
        raise RuntimeError('Expected exactly one complete ensemble world-model state')
    model_path, model = world_models[0]
    if not isinstance(model['fit_calls'], int) or (not smoke and model['fit_calls'] <= 0):
        raise RuntimeError('Formal Dyna checkpoint contains no fitted world model')
    if model['fit_calls'] != state['model_fit_calls'] or state['model_optimizer_steps'] != model['fit_calls'] * model['config']['fit_steps']:
        raise RuntimeError('World-model fit/update counters disagree')
    if state.get('imagined_horizon') != 1 or state.get('history_frames') != 4:
        raise RuntimeError('Checkpoint does not use the declared one-step/four-frame Dyna method')
    weights = model['models']
    if not isinstance(weights, dict) or not weights:
        raise RuntimeError('Empty world-model parameters')
    parameters = 0
    for name, tensor in weights.items():
        if not torch.is_tensor(tensor) or not tensor.is_cuda or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f'World-model checkpoint parameter must be finite CUDA tensor: {name}')
        parameters += tensor.numel()
    if {str(name).split('.')[0] for name in weights} != {'0', '1', '2'}:
        raise RuntimeError('World-model checkpoint must contain all three ensemble members')
    return {'verified': True, 'imagined_optimizer_steps': expected_steps,
            'verified_training_counters': {key: state[key] for key in counter_keys},
            'world_model_fit_calls': model['fit_calls'], 'world_model_state_path': model_path,
            'world_model_parameter_count': parameters, 'auxiliary_parameters_finite_cuda': True,
            'world_model_forward_executed': False}


def statistics_or_none(values, total):
    valid = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in valid):
        raise RuntimeError('Nonfinite evaluation statistic')
    if not valid:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'valid_count': 0, 'total_count': total}
    return {'mean': statistics.mean(valid), 'std': statistics.pstdev(valid), 'min': min(valid), 'max': max(valid),
            'valid_count': len(valid), 'total_count': total}


def summarize(records):
    count = len(records)
    if not count:
        raise RuntimeError('No evaluation episodes')
    metrics = {'episodes': count, 'fall_rate': statistics.mean(float(row['fell']) for row in records),
               'timeout_rate': statistics.mean(float(row['timeout']) for row in records),
               'invalid_state_rate': statistics.mean(float(row['invalid_state']) for row in records),
               'out_of_bounds_rate': statistics.mean(float(row['out_of_bounds']) for row in records)}
    for key in ('distance_x_m', 'avg_speed_m_s', 'duration_s', 'return', 'average_speed_error_m_s',
                'average_speed_abs_error_m_s', 'abs_final_heading_yaw_deg', 'abs_lateral_displacement_m',
                'raw_action_clipped_fraction', 'raw_action_abs_max'):
        metrics[key] = statistics_or_none([row[key] for row in records], count)
    for key in STUDY_METRICS:
        metrics[key] = statistics_or_none([row['study_metrics'][key] for row in records], count)
    metrics['raw_action_clipped_fraction_pooled'] = (
        sum(row['raw_action_clipped_count'] for row in records) / sum(row['raw_action_scalar_count'] for row in records)
    )
    metrics['final_heading_gt45deg_fraction'] = statistics.mean(row['abs_final_heading_yaw_deg'] > 45 for row in records)
    return metrics


def pair_initial_condition(output_dir, condition, seed, initial, protocol_hash, signature):
    directory = output_dir / 'paired_initial_conditions'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{condition}_seed{seed}.json'
    document = {'protocol_sha256': protocol_hash, 'manifest_signature': signature,
                'condition': condition, 'evaluation_seed': seed, 'initial_condition': initial}
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise RuntimeError(f'Paired initial states/actual domain parameters differ: {condition}, seed {seed}')
    else:
        write_json(path, document)
    return {'path': str(path), 'sha256': sha256(path), 'verified': True}


def evaluate_case(env, wrapper, actor, entry, condition, settings, args, protocol_hash, signature, torch):
    records, initials, validations, pairings, trajectory = [], [], [], [], []
    terminal_env0 = None
    count = settings['episodes_per_seed']
    env.set_domain_randomization(condition == 'randomized')
    for seed_index, seed in enumerate(settings['seeds']):
        env.reset(seed=seed)
        env.episode_records.clear()
        ids = list(range(count))
        validation = env.verify_domain_parameters(env_ids=ids)
        validations.append({'evaluation_seed': seed, 'validation': validation})
        initial = {
            'evaluation_seed': seed,
            'root_state_w': env.robot.data.root_state_w[:count].detach().cpu().tolist(),
            'joint_pos_rad': env.robot.data.joint_pos[:count].detach().cpu().tolist(),
            'joint_vel_rad_s': env.robot.data.joint_vel[:count].detach().cpu().tolist(),
            'domains': env.get_domain_snapshot(env_ids=ids),
        }
        pairings.append(pair_initial_condition(args.output_dir, condition, seed, initial, protocol_hash, signature))
        initials.append(initial)
        obs = wrapper.get_observations()
        seen = set()
        active = torch.ones(count, dtype=torch.bool, device=args.device)
        clipped_count = torch.zeros(count, dtype=torch.long, device=args.device)
        scalar_count = torch.zeros_like(clipped_count)
        raw_max = torch.zeros(count, device=args.device)
        for step in range(int(env.max_episode_length)+5):
            actions = actor.act_inference(obs)
            if not actions.is_cuda or not bool(torch.isfinite(actions).all()):
                raise RuntimeError(f'{entry["id"]}: non-CUDA or nonfinite actor action')
            selected = actions[:count]
            clipped_count += torch.where(active, (selected.abs()>1.0).sum(dim=1), 0)
            scalar_count += active.long()*wrapper.num_actions
            raw_max = torch.where(active, torch.maximum(raw_max, selected.abs().max(dim=1).values), raw_max)
            frame = None
            if seed_index == 0 and 0 not in seen:
                frame = {'step': step, 'time_s': step*float(env.step_dt),
                         'pre_root_state_w': env.robot.data.root_state_w[0].detach().cpu().tolist(),
                         'joint_pos_rad': env.robot.data.joint_pos[0].detach().cpu().tolist(),
                         'raw_action': actions[0].detach().cpu().tolist(),
                         'clipped_normalized_action': actions[0].clamp(-1, 1).detach().cpu().tolist()}
            obs, reward, dones, extras = wrapper.step(actions)
            if not bool(torch.isfinite(reward).all()) or not bool(torch.isfinite(obs['policy']).all()):
                raise RuntimeError(f'{entry["id"]}: nonfinite observation/reward')
            if frame is not None:
                frame['post_root_state_w'] = env.last_control_sample['root_state_w'][0].detach().cpu().tolist()
                frame['applied_targets_rad'] = env.last_control_sample['applied_targets'][0].detach().cpu().tolist()
                frame['joint_velocity_firmware_rad_s'] = env.last_control_sample['joint_velocity_firmware'][0].detach().cpu().tolist()
                trajectory.append(frame)
            while env.episode_records:
                record = dict(env.episode_records.popleft())
                index = int(record['env_id'])
                if index >= count or index in seen:
                    continue
                seen.add(index)
                active[index] = False
                if 'study_metrics' not in record or any(key not in record['study_metrics'] for key in STUDY_METRICS):
                    raise RuntimeError('Terminal record lacks preregistered instantaneous metrics')
                actions_seen = int(scalar_count[index].item())
                expected_samples = actions_seen // wrapper.num_actions
                if record['study_metrics']['samples'] != expected_samples:
                    raise RuntimeError('Diagnostic sample count differs from first-episode action count')
                if abs(expected_samples*float(env.step_dt)-float(record['duration_s'])) > 1e-4:
                    raise RuntimeError('Diagnostic sampling duration does not match episode duration')
                clipped = int(clipped_count[index].item())
                record.update({
                    'policy_id': entry['id'], 'variant': entry['variant'], 'train_seed': entry['train_seed'],
                    'condition': condition, 'evaluation_seed': seed,
                    'raw_action_clipped_count': clipped, 'raw_action_scalar_count': actions_seen,
                    'raw_action_clipped_fraction': clipped/actions_seen,
                    'raw_action_abs_max': float(raw_max[index].item()),
                    'average_speed_error_m_s': float(record['avg_speed_m_s'])-float(env.cfg.command_forward_m_s),
                    'average_speed_abs_error_m_s': abs(float(record['avg_speed_m_s'])-float(env.cfg.command_forward_m_s)),
                    'abs_final_heading_yaw_deg': abs(math.degrees(float(record['final_heading_yaw_rad']))),
                    'abs_lateral_displacement_m': abs(float(record['lateral_displacement_m'])),
                })
                records.append(record)
                if seed_index == 0 and index == 0:
                    terminal_env0 = record
            if len(seen) == count:
                break
        if len(seen) != count:
            raise RuntimeError(f'{entry["id"]}/{condition}/{seed}: missing complete first episodes')
    records.sort(key=lambda row: (row['evaluation_seed'], row['env_id']))
    trajectory_dir = args.output_dir / 'trajectories'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f'{entry["id"]}_{condition}_seed{settings["seeds"][0]}.json'
    write_json(trajectory_path, {
        'protocol_sha256': protocol_hash, 'manifest_signature': signature, 'policy_id': entry['id'],
        'variant': entry['variant'], 'train_seed': entry['train_seed'], 'condition': condition,
        'evaluation_seed': settings['seeds'][0], 'env_id': 0,
        'state_timing': 'pre_root_state before action; post_root_state after physics and before automatic reset',
        'root_state_layout': 'xyz, quaternion wxyz, world linear velocity, world angular velocity',
        'joint_position_order': list(env.robot.joint_names),
        'action_target_velocity_order': list(env.firmware_joint_names),
        'samples': trajectory, 'terminal_record': terminal_env0,
    })
    return {
        'status': 'completed', 'protocol_sha256': protocol_hash, 'manifest_signature': signature,
        'policy_id': entry['id'], 'variant': entry['variant'], 'train_seed': entry['train_seed'],
        'condition': condition, 'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['sha256'],
        'evaluation_seeds': settings['seeds'], 'evaluation_num_envs': settings['num_envs'],
        'episodes_per_seed': count, 'evaluation_environment_variant': 'full_dr',
        'metrics': summarize(records), 'episodes': records,
        'per_evaluation_seed_metrics': {
            str(seed): summarize([row for row in records if row['evaluation_seed']==seed]) for seed in settings['seeds']
        },
        'initial_conditions': initials, 'pairing_verified': True, 'pairing_references': pairings,
        'domain_readback_validation': validations, 'trajectory_file': str(trajectory_path),
        'trajectory_sha256': sha256(trajectory_path), 'metric_definitions': env.metric_definitions,
        'all_actions_observations_rewards_finite': True,
        'statistics_note': 'std describes episode variation within this trained policy; training-seed variability must be computed across three policy means',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def validate_saved_case(case, entry, condition, settings, args, protocol_hash, signature):
    for key, expected in (('status', 'completed'), ('protocol_sha256', protocol_hash), ('manifest_signature', signature),
                          ('policy_id', entry['id']), ('condition', condition), ('checkpoint_sha256', entry['sha256']),
                          ('pairing_verified', True), ('evaluation_environment_variant', 'full_dr')):
        if case.get(key) != expected:
            raise RuntimeError(f'Cannot resume inconsistent case {entry["id"]}/{condition}: {key}')
    expected_ids = {(seed, env_id) for seed in settings['seeds'] for env_id in range(settings['episodes_per_seed'])}
    actual_ids = [(row['evaluation_seed'], row['env_id']) for row in case['episodes']]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise RuntimeError('Saved case does not contain all first episodes exactly once')
    for initial in case['initial_conditions']:
        pair_initial_condition(args.output_dir, condition, initial['evaluation_seed'], initial, protocol_hash, signature)
    if sha256(case['trajectory_file']) != case['trajectory_sha256']:
        raise RuntimeError('Saved trajectory digest mismatch')
    if summarize(case['episodes']) != case['metrics']:
        raise RuntimeError('Saved episode metrics do not reproduce saved case summary')


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    settings = validate_protocol(protocol, args)
    entries = resolve_checkpoints(protocol, args.smoke)
    code_paths = ('scripts/evaluate_dyna_ppo.py', 'scripts/train_ppo_study.py', 'scripts/train_ppo.py',
                  'sesame_rl/environment.py', 'sesame_rl/dr_environment.py', 'sesame_rl/study_environment.py',
                  *DYNA_SOURCES)
    core = {'schema_version': 1, 'study_id': protocol['study_id'], 'protocol_sha256': protocol_hash,
            'effective_evaluation': settings, 'smoke': args.smoke,
            'evaluation_sources': {relative: sha256(ROOT/relative) for relative in code_paths},
            'resolved_checkpoints': entries}
    signature = canonical_hash(core)
    manifest_path = args.output_dir / 'evaluation_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('manifest_signature') != signature or old.get('core') != core:
            raise RuntimeError('Resume rejected: protocol, final checkpoint, configuration or evaluator sources changed')
    else:
        write_json(manifest_path, {'manifest_signature': signature, 'core': core,
                                  'created_at_utc': datetime.now(timezone.utc).isoformat()})
        shutil.copy2(args.protocol, args.output_dir / 'protocol_snapshot.json')
        for relative in code_paths:
            target = args.output_dir / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
    progress_path = args.output_dir / 'progress.json'
    progress = {'status': 'running', 'protocol_sha256': protocol_hash, 'manifest_signature': signature,
                'total_cases': len(entries)*len(settings['conditions']), 'completed_cases': {}}
    result = {'status': 'running', 'study_id': protocol['study_id'], 'protocol_sha256': protocol_hash,
              'manifest_signature': signature, 'smoke': args.smoke, 'evaluation': settings,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'git_operations': 'disabled'}
    write_json(args.output_dir/'result.json', result)
    runtime = ROOT / '.runtime/dyna_ppo_evaluation'
    for key, relative in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
                          'XDG_DATA_HOME': 'data', 'CUDA_CACHE_PATH': 'cuda_cache'}.items():
        directory = runtime/relative
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
        kit_args = ' '.join(['--portable-root', str(runtime/'kit'),
            f'--/log/file={args.output_dir/"isaac.log"}', '--/app/settings/persistent=false',
            '--/app/telemetry/enabled=false', '--/renderer/multiGpu/enabled=false'])
        app = AppLauncher(headless=True, device=args.device, kit_args=kit_args).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime/'warp_cache')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.study_environment import SesameStudyEnv, make_study_env_cfg
        cfg = make_study_env_cfg(num_envs=settings['num_envs'], seed=settings['seeds'][0], device=args.device,
                                 episode_length_s=settings['episode_length_s'], variant='full_dr')
        write_json(args.output_dir/'evaluation_environment_config.json', cfg.to_dict())
        env = SesameStudyEnv(cfg)
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
                # Re-hash immediately before deserialization to detect changed checkpoint files.
                if sha256(entry['checkpoint']) != entry['sha256']:
                    raise RuntimeError(f'Checkpoint changed after manifest resolution: {entry["id"]}')
                actor_config = copy.deepcopy(entry['ppo_config']['policy'])
                actor_config.pop('class_name')
                with torch.device(args.device):
                    actor = ActorCritic(wrapper.get_observations(), entry['ppo_config']['obs_groups'],
                                        wrapper.num_actions, **actor_config).to(args.device)
                checkpoint = torch.load(entry['checkpoint'], map_location=args.device, weights_only=False)
                if entry['variant'] == 'dyna_ppo':
                    audit = audit_dyna_checkpoint(checkpoint, entry, args.smoke, torch)
                    result.setdefault('dyna_checkpoint_audits', {})[entry['id']] = audit
                    write_json(args.output_dir/'dyna_checkpoint_audits.json', result['dyna_checkpoint_audits'])
                actor.load_state_dict(checkpoint['model_state_dict'])
                actor.eval()
                if not all(parameter.is_cuda for parameter in actor.parameters()):
                    raise RuntimeError('Loaded policy parameters must be on CUDA')
                for condition in settings['conditions']:
                    case_path = args.output_dir/'cases'/entry['id']/f'{condition}.json'
                    case_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.resume and case_path.exists():
                        case = json.loads(case_path.read_text())
                        validate_saved_case(case, entry, condition, settings, args, protocol_hash, signature)
                        resumed = True
                    else:
                        case = evaluate_case(env, wrapper, actor, entry, condition, settings, args,
                                             protocol_hash, signature, torch)
                        write_json(case_path, case)
                        resumed = False
                    key = f'{entry["id"]}/{condition}'
                    progress['completed_cases'][key] = {'path': str(case_path), 'sha256': sha256(case_path),
                        'policy_id': entry['id'], 'condition': condition, 'resumed': resumed}
                    write_json(progress_path, progress)
                    print('STUDY_CASE_COMPLETED='+json.dumps({'case': key, 'resumed': resumed,
                        'episodes': case['metrics']['episodes'],
                        'velocity_x_rmse_m_s': case['metrics']['velocity_x_rmse_m_s']['mean']}), flush=True)
                del actor, checkpoint
        progress['status'] = 'completed'
        write_json(progress_path, progress)
        result.update({'status': 'completed', 'case_paths': [value['path'] for value in progress['completed_cases'].values()],
                       'cases_completed': len(progress['completed_cases']), 'pairing_verified_all_cases': True,
                       'wall_seconds_before_shutdown': time.monotonic()-started})
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if interrupted:
            exit_code = 130
        result.update({'status': 'interrupted' if interrupted else 'failed',
                       'error': repr(exc), 'traceback': traceback.format_exc()})
        progress['status'] = result['status']
        write_json(progress_path, progress)
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir/'result.json', result)
        print('SESAME_STUDY_EVALUATION_RESULT='+json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def shutdown_timeout():
                result['shutdown'] = {'completed': False, 'forced_process_exit': True, 'timeout_seconds': 15}
                write_json(args.output_dir/'result.json', result)
                print('ISAAC_SHUTDOWN_TIMEOUT: evaluation results saved; exiting only this process', flush=True)
                os._exit(exit_code)
            timer = threading.Timer(15.0, shutdown_timeout)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            result['shutdown'] = {'completed': True, 'forced_process_exit': False}
            write_json(args.output_dir/'result.json', result)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
