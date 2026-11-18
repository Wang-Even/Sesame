#!/usr/bin/env python3
"""Rerun corrected SuperMimic and reuse the numerically unchanged PPO arm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
META = '/home/ubuntu/miniforge3/envs/metaworld_v010/bin/python'
ISAAC = '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-study', type=Path, required=True)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    original, study = args.original_study.resolve(), args.study_dir.resolve()
    assert study.is_relative_to(ROOT / 'runs') and original != study
    assert not (study / 'queue.json').exists()
    proof = json.loads((study / 'boundary_validation.json').read_text())
    assert proof['passed'] and proof['ppo_implementation_unchanged']
    def write(path, data):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)
    reason = {'superseded_supermimic': True, 'reason': 'An imagined trajectory reaching the reference endpoint was incorrectly charged the fall penalty. Corrected failure reward and completion continuation are now separate.',
              'corrected_study': str(study), 'ppo_update_affected': False}
    write(original / 'known_issue.json', reason)
    write(study / 'correction.json', dict(reason, original_study=str(original),
          corrected_supermimic_actor_initialization='Random from scratch, seed 42, same original data budget. Regression-test weights are not used for training.'))
    write(study / 'baseline_reuse.json', {'algorithm': 'ppo', 'original_study': str(original),
          'original_protocol_sha256': hashlib.sha256((original / 'protocol.json').read_bytes()).hexdigest(),
          'proof': 'boundary_validation.json', 'numerical_implementation_unchanged': True})
    (study / 'ppo_seed42').symlink_to(original / 'ppo_seed42', target_is_directory=True)
    for name in ('preflight_numerical_validation.json', 'paired_validation_seed81042.pt', 'paired_test_seed91042.pt'):
        shutil.copy2(original / name, study / name)
    resource = json.loads((original / 'resource_observation.json').read_text())
    resource['corrected_supermimic_overlaps_with_ppo'] = True
    resource['time_comparison_valid_as_controlled_benchmark'] = False
    write(study / 'resource_observation.json', resource)
    state = {'pid': os.getpid(), 'status': 'running', 'active_algorithm': 'supermimic',
             'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'methods': {'supermimic': {'status': 'running'}, 'ppo': {'status': 'running', 'reused_from': str(original / 'ppo_seed42')}}}
    write(study / 'queue.json', state)
    env = dict(os.environ, CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    command = ['/home/ubuntu/miniforge3/bin/conda', 'run', '-n', 'env_isaaclab', '--no-capture-output', ISAAC, '-u',
               str(ROOT / 'scripts/train_supermimic_comparison.py'), '--algorithm', 'supermimic', '--study-dir', str(study)]
    with (study / 'supermimic_console.log').open('w') as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((study / 'supermimic_seed42/result.json').read_text())
    state['methods']['supermimic'].update(status=result['status'], exit_code=code)
    state['active_algorithm'] = 'ppo'
    write(study / 'queue.json', state)
    while True:
        baseline = json.loads((original / 'ppo_seed42/result.json').read_text())
        if baseline['status'] != 'running':
            break
        time.sleep(10)
    state['methods']['ppo']['status'] = baseline['status']
    baseline_path = study / 'ppo_seed42'
    if baseline['status'] == 'completed' and baseline_path.is_symlink():
        temp = study / 'ppo_seed42_copy'
        shutil.copytree(baseline_path.resolve(), temp)
        baseline_path.unlink(); temp.rename(baseline_path)
        shutil.copy2(original / 'protocol.json', study / 'baseline_protocol.json')
        reuse = json.loads((study / 'baseline_reuse.json').read_text())
        reuse.update(local_protocol_copy='baseline_protocol.json', baseline_files_copied=True,
                     copied_at_utc=datetime.now(timezone.utc).isoformat())
        write(study / 'baseline_reuse.json', reuse)
    plotting = subprocess.call([META, '-s', str(ROOT / 'scripts/analyze_supermimic_comparison.py'),
                                '--study-dir', str(study)], cwd=ROOT, env=env)
    state['plot_exit_code'] = plotting
    state['status'] = 'completed' if plotting == 0 and all(v['status'] == 'completed' for v in state['methods'].values()) else 'failed'
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    state.pop('active_algorithm', None)
    write(study / 'queue.json', state)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
