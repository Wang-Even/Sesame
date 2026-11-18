#!/usr/bin/env python3
"""Audit saved experiment metadata without loading or computing any model."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    protocol = read(study / 'protocol.json')
    reuse = read(study / 'baseline_reuse.json') if (study / 'baseline_reuse.json').exists() else None
    boundary = read(study / 'boundary_validation.json') if reuse else None
    if reuse:
        assert boundary['passed'] and boundary['ppo_implementation_unchanged']
    assert not protocol['smoke'], 'Use completed formal runs'
    assert read(study / 'queue.json')['status'] == 'completed'
    preflight = read(study / 'preflight_numerical_validation.json')
    assert preflight['passed']
    thresholds = [.05, math.radians(10), .25, .02, .10]
    names = ['velocity', 'heading', 'joint_position', 'foot_position', 'root_position']
    method_checks = {}
    failure_rows = []
    for algorithm in ('supermimic', 'ppo'):
        run = study / f'{algorithm}_seed42'
        result = read(run / 'result.json')
        assert result['status'] == 'completed'
        if algorithm == 'ppo' and reuse:
            arm_protocol_path = study / reuse['local_protocol_copy'] if reuse.get('local_protocol_copy') else Path(reuse['original_study']) / 'protocol.json'
        else:
            arm_protocol_path = study / 'protocol.json'
        arm_protocol = read(arm_protocol_path)
        assert result['protocol_sha256'] == digest(arm_protocol_path)
        if algorithm == 'ppo' and reuse:
            assert digest(arm_protocol_path) == reuse['original_protocol_sha256']
            for key in ('seed', 'num_envs', 'real_transition_budget', 'steps_per_collection', 'evaluation_every_updates',
                        'evaluation_episodes', 'success', 'reward_weights', 'reward_scales', 'networks', 'ppo', 'common',
                        'dataset_checksums_sha256', 'reference_index_sha256', 'asset_sha256'):
                assert arm_protocol[key] == protocol[key], key
        assert result['initial_actor_sha256'] == preflight['checks'][algorithm]['initial_actor_sha256']
        assert result['online_steps'] == result['updates'] * protocol['num_envs'] * protocol['steps_per_collection']
        assert result['cumulative_real_steps'] == result['online_steps'] + result['offline_supplied_steps']
        assert result['cumulative_real_steps'] <= protocol['real_transition_budget']
        if algorithm == 'supermimic':
            assert result['offline_supplied_steps'] == protocol['supermimic']['offline_supplied_transitions']
            assert result['imagined_steps'] == result['updates'] * protocol['supermimic']['learner_rounds_per_collection'] * protocol['supermimic']['imagination_batch'] * protocol['supermimic']['imagination_horizon']
        else:
            assert result['offline_supplied_steps'] == result['imagined_steps'] == 0
        numerical = read(run / 'math_validation.json')
        assert numerical['cuda'] and numerical['fk_max_abs_m'] < 1e-4
        assert numerical['observation_max_abs'] < 3e-5 and numerical['goal_max_abs'] < 3e-5
        evaluations = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
        evaluation_cost = sum(row['evaluation_seconds'] for row in evaluations
                              if row.get('stage') != 'world_pretrained_actor_unchanged')
        assert abs(evaluation_cost - result['evaluation_seconds']) < 1e-5
        training = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
        training_cost = result.get('offline_pretraining_seconds', 0) + sum(
            row['collection_seconds'] + row['update_seconds'] for row in training)
        assert abs(training_cost - result['training_seconds']) < 1e-5
        validation = [r for r in evaluations if r['label'] == 'validation']
        assert [r['cumulative_real_steps'] for r in validation] == sorted(r['cumulative_real_steps'] for r in validation)
        assert [r['training_seconds'] for r in validation] == sorted(r['training_seconds'] for r in validation)
        for row in evaluations:
            assert row['episodes'] == 64 and row['initial_pairing_max_abs'] <= 1e-6
            assert len(row['episode_metrics']) == len(row['episode_lengths']) == len(row['episode_success']) == 64
            assert sum(row['episode_success']) / 64 == row['success_rate']
            for success, length, metrics in zip(row['episode_success'], row['episode_lengths'], row['episode_metrics']):
                assert all(math.isfinite(v) for v in metrics)
                if success:
                    assert length == 450 and all(v <= t for v, t in zip(metrics, thresholds))
        stable = result['stable_success']
        if stable:
            assert stable['first_crossing']['success_rate'] >= .8 and stable['confirmation']['success_rate'] >= .8
            assert (run / 'first_success.pt').exists()
        else:
            assert result['stop_reason'] == 'budget_exhausted'
        test = read(run / 'test_result.json')
        assert test == result['test']
        selected = run / test['selected_checkpoint']
        assert selected.exists() and (run / 'final.pt').exists()
        assert test['selected_checkpoint_cumulative_steps'] <= result['cumulative_real_steps']
        for relative, expected in arm_protocol['sources'].items():
            assert digest(run / 'sources' / relative) == expected
            current = digest(ROOT / relative)
            if algorithm == 'ppo' and reuse and relative == 'sesame_rl/supermimic_small.py':
                assert expected == boundary['old_source_sha256'] and current == boundary['corrected_source_sha256']
                def definitions(path):
                    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
                old, new = definitions(run / 'sources' / relative), definitions(ROOT / relative)
                assert sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)) == ['supermimic_update']
            else:
                assert current == expected, f'Live source changed: {relative}'
        for i, (name, threshold) in enumerate(zip(names, thresholds)):
            count = sum(row[i] <= threshold for row in test['episode_metrics'])
            failure_rows.append({'algorithm': algorithm, 'condition': name, 'passing_episodes': count, 'episodes': 64,
                                 'note': 'Each condition separately, including trajectories that failed another condition'})
        method_checks[algorithm] = {'completed': True, 'validation_points': len(validation),
                                    'total_real_steps': result['cumulative_real_steps'],
                                    'initial_actor_sha256': result['initial_actor_sha256'],
                                    'test_success_rate': test['success_rate'],
                                    'selected_checkpoint': test['selected_checkpoint'],
                                    'selected_checkpoint_sha256': digest(selected)}
    assert method_checks['supermimic']['initial_actor_sha256'] == method_checks['ppo']['initial_actor_sha256']
    dataset = Path(protocol['dataset'])
    assert digest(dataset / 'checksums.json') == protocol['dataset_checksums_sha256']
    assert digest(dataset / 'reference_index.json') == protocol['reference_index_sha256']
    assert digest(ROOT / 'assets/isaac/sesame/sesame.usd') == protocol['asset_sha256']
    output = study / 'analysis'
    for name in ('curves_by_steps', 'curves_by_time', 'stability_curves'):
        assert (output / f'{name}.pdf').read_bytes()[:4] == b'%PDF'
        assert (output / f'{name}.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    summary = read(output / 'summary.json')
    for algorithm, checked in method_checks.items():
        assert summary['methods'][algorithm]['status'] == 'completed'
        assert summary['methods'][algorithm]['total_real_steps'] == checked['total_real_steps']
        assert summary['methods'][algorithm]['test']['success_rate'] == checked['test_success_rate']
    with (output / 'test_condition_pass_counts.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(failure_rows[0]))
        writer.writeheader(); writer.writerows(failure_rows)
    sources = output / 'sources'; sources.mkdir(exist_ok=True)
    for name in ('analyze_supermimic_comparison.py', 'audit_supermimic_comparison.py', 'run_supermimic_comparison.py',
                 'validate_supermimic_small.py', 'run_supermimic_correction.py', 'validate_supermimic_boundary.py'):
        shutil.copy2(ROOT / 'scripts' / name, sources / name)
    weights = {str(p.relative_to(study)): digest(p) for p in sorted(study.glob('*_seed42/*.pt'))}
    report = {'passed': True, 'methods': method_checks, 'frozen_sources_match': True,
              'same_initial_actor': True, 'all_evaluations_paired': True, 'plot_files_valid': True,
              'dataset_manifests_unchanged': True, 'weight_sha256': weights,
              'ppo_reused_with_unchanged_numerical_implementation': bool(reuse),
              'analysis_sources_sha256': {p.name: digest(p) for p in sorted(sources.glob('*.py'))}}
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'methods': method_checks}, indent=2))


if __name__ == '__main__':
    main()
