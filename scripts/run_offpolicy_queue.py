#!/usr/bin/env python3
"""Sequential frozen SAC/DQN training, common PPO reevaluation, and analysis."""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONDA = '/home/ubuntu/miniforge3/bin/conda'
ISAAC_PY = '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'
METADATA_PY = '/home/ubuntu/miniforge3/envs/metaworld_v010/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if not study.is_relative_to(ROOT / 'runs'):
        raise ValueError('Study must be inside Sesame runs')
    lock = (study / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = study / 'protocol.json'
    digest = sha(protocol_path)
    if digest != (study / 'protocol.sha256').read_text().strip():
        raise RuntimeError('Training protocol changed')
    protocol = json.loads(protocol_path.read_text())
    state = dict(status='running', phase='training', pid=os.getpid(), protocol_sha256=digest,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), completed=[], active=None)
    state_path = study / 'progress.json'
    write(state_path, state)
    environment = os.environ.copy()
    environment.update(CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')

    def check_sources():
        if sha(protocol_path) != digest:
            raise RuntimeError('Training protocol changed while running')
        for relative, expected in protocol['frozen_sources'].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError('Frozen source changed: '+relative)

    def run_child(command, log_path):
        state['command'] = command
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state['child_pid'] = process.pid
            write(state_path, state)
            code = process.wait()
        state.pop('child_pid', None)
        if code != 0:
            raise RuntimeError(f'Child exited {code}; see {log_path}')

    prefix = [CONDA, 'run', '-n', 'env_isaaclab', '--no-capture-output', ISAAC_PY, '-u']
    try:
        for entry in protocol['training_runs']:
            check_sources()
            run = Path(entry['run_dir'])
            run.mkdir(parents=True, exist_ok=True)
            result_path = run / 'result.json'
            state.update(phase='training', active=entry['id'], active_run_dir=str(run))
            write(state_path, state)
            if not result_path.exists():
                while True:
                    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                                        '--format=csv,noheader,nounits'], text=True).strip())
                    if free >= 7000:
                        state['status'] = 'running'
                        break
                    state.update(status='waiting_for_gpu_memory', free_memory_mib=free)
                    write(state_path, state)
                    time.sleep(15)
                cmd = prefix + [str(ROOT / 'scripts/train_offpolicy.py'), '--algorithm', entry['algorithm'],
                                '--seed', str(entry['seed']), '--run-dir', str(run)]
                print('START '+entry['id'], flush=True)
                run_child(cmd, run / 'train.log')
            result = json.loads(result_path.read_text())
            if result['status'] != 'completed' or result['smoke']:
                raise RuntimeError('Incomplete formal run; preserve and inspect: '+str(run))
            if result['environment_transitions_total'] != protocol['training']['transitions_per_run']:
                raise RuntimeError('Wrong training budget')
            if result['reload_max_abs_action_error'] != 0 or result['checkpoint_sha256'] != sha(run / 'final.pt'):
                raise RuntimeError('Checkpoint validation failed')
            if result['reward_version'] != 'absolute_heading_v1' or result['heading_penalty_scale'] != 0.5:
                raise RuntimeError('Training reward mismatch')
            state['completed'].append(dict(id=entry['id'], run_dir=str(run), checkpoint=str(run/'final.pt'),
                                           sha256=result['checkpoint_sha256']))
            state['active'] = None
            write(state_path, state)
            print('COMPLETED '+entry['id'], flush=True)
        check_sources()
        from evaluate_offpolicy import build_evaluation_protocol
        evaluation_protocol = build_evaluation_protocol(study)
        state.update(phase='evaluation', active='ppo_sac_dqn_paired_evaluation')
        write(state_path, state)
        run_child(prefix + [str(ROOT/'scripts/evaluate_offpolicy.py'), '--protocol', str(evaluation_protocol),
                            '--output-dir', str(study/'evaluation'), '--resume'], study/'evaluation.log')
        state.update(phase='analysis', active='summary_and_curves')
        write(state_path, state)
        run_child([METADATA_PY, '-s', str(ROOT/'scripts/analyze_offpolicy.py'), str(study)], study/'analysis.log')
        state.update(status='completed', phase='completed', active=None,
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        print('ALL_TRAINING_AND_EVALUATION_COMPLETED', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
