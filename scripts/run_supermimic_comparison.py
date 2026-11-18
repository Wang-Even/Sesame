#!/usr/bin/env python3
"""Run the two CUDA experiments sequentially and preserve their logs/plots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDA = '/home/ubuntu/miniforge3/bin/conda'
ISAAC = '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'
METADATA = '/home/ubuntu/miniforge3/envs/metaworld_v010/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if not study.is_relative_to(ROOT / 'runs'):
        parser.error('Use Sesame/runs')
    study.mkdir(exist_ok=True, parents=True)
    if (study / 'queue.json').exists():
        parser.error('Queue already exists; choose a new study directory')
    env = dict(os.environ, CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    state = {'pid': os.getpid(), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'running', 'methods': {}}
    def save():
        temp = study / 'queue.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(study / 'queue.json')
    save()
    for algorithm in ('supermimic', 'ppo'):
        command = [CONDA, 'run', '-n', 'env_isaaclab', '--no-capture-output', ISAAC, '-u',
                   str(ROOT / 'scripts/train_supermimic_comparison.py'), '--algorithm', algorithm,
                   '--study-dir', str(study)]
        state['active_algorithm'] = algorithm
        state['methods'][algorithm] = {'command': command, 'status': 'running'}
        save()
        with (study / f'{algorithm}_console.log').open('w') as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_path = study / f'{algorithm}_seed42/result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        state['methods'][algorithm].update(exit_code=code, status=result.get('status', 'failed'))
        save()
        plotting = subprocess.call([METADATA, '-s', str(ROOT / 'scripts/analyze_supermimic_comparison.py'),
                                    '--study-dir', str(study)], cwd=ROOT, env=env)
        state['methods'][algorithm]['plot_exit_code'] = plotting
        save()
    state['status'] = 'completed' if all(v['status'] == 'completed' and v['plot_exit_code'] == 0
                                       for v in state['methods'].values()) else 'failed'
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    state.pop('active_algorithm', None)
    save()
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
