#!/usr/bin/env python3
"""Sequential ordinary-PPO training for a frozen heading-reward experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONDA = '/home/ubuntu/miniforge3/bin/conda'
PYTHON = '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if ROOT / 'runs' not in study.parents:
        raise ValueError('Study must be inside Sesame runs')
    lock = (study / 'training_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((study / 'protocol.json').read_text())
    protocol_hash = sha256(study / 'protocol.json')
    if protocol_hash != (study / 'protocol.sha256').read_text().strip():
        raise RuntimeError('Protocol changed')
    state = dict(status='running', queue_pid=os.getpid(), protocol_sha256=protocol_hash,
                 started_at_utc=datetime.now(timezone.utc).isoformat(), completed=[], active=None)
    state_path = study / 'training_progress.json'
    write(state_path, state)
    env = os.environ.copy()
    env.update(CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    cfg = protocol['training']
    try:
        for entry in protocol['training_runs']:
            for relative, expected in protocol['frozen_sources'].items():
                if sha256(ROOT / relative) != expected:
                    raise RuntimeError('Frozen source changed: ' + relative)
            run = Path(entry['run_dir'])
            result_path = run / 'result.json'
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old['status'] != 'completed':
                    raise RuntimeError('Incomplete previous run; inspect before retry: ' + str(run))
            else:
                # Wait for sufficient free VRAM; never stop unrelated jobs.
                while True:
                    output = subprocess.check_output([
                        'nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'
                    ], text=True)
                    free_mib = int(output.strip().splitlines()[0])
                    if free_mib >= 7000:
                        break
                    state.update(status='waiting_for_gpu_memory', active=entry['id'], free_memory_mib=free_mib)
                    write(state_path, state)
                    time.sleep(15)
                run.mkdir(parents=True, exist_ok=True)
                cmd = [CONDA, 'run', '-n', 'env_isaaclab', '--no-capture-output', PYTHON, '-u',
                       str(ROOT / 'scripts/train_heading_ppo.py'), '--seed', str(entry['seed']),
                       '--run-dir', str(run), '--num-envs', str(cfg['num_envs']),
                       '--iterations', str(cfg['iterations']), '--steps-per-env', str(cfg['steps_per_env']),
                       '--heading-penalty-scale', str(cfg['heading_penalty_scale'])]
                state.update(status='running', active=entry['id'], active_run_dir=str(run), command=cmd)
                with (run / 'train.log').open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL)
                    state['active_launcher_pid'] = proc.pid
                    write(state_path, state)
                    print('START ' + entry['id'], flush=True)
                    code = proc.wait()
                if code != 0:
                    raise RuntimeError(f'{entry["id"]} exited {code}; see train.log')
            result = json.loads(result_path.read_text())
            checkpoint = run / 'final.pt'
            if result['status'] != 'completed' or result['environment_transitions_total'] != cfg['transitions_per_run']:
                raise RuntimeError('Incomplete or wrong-budget result')
            if result['reload_max_abs_action_error'] != 0 or result['checkpoint_sha256'] != sha256(checkpoint):
                raise RuntimeError('Checkpoint validation failed')
            if result.get('algorithm') != 'PPO':
                raise RuntimeError('Expected ordinary PPO')
            actual_cfg = json.loads((run / 'ppo_config.json').read_text())
            reference_cfg = json.loads(Path(cfg['ppo_config_reference']).read_text())
            for key in ('policy', 'algorithm', 'obs_groups', 'num_steps_per_env'):
                if actual_cfg[key] != reference_cfg[key]:
                    raise RuntimeError('PPO settings changed: ' + key)
            state['completed'].append(dict(id=entry['id'], seed=entry['seed'], run_dir=str(run),
                                           checkpoint=str(checkpoint), sha256=sha256(checkpoint)))
            state['active'] = None
            state.pop('active_launcher_pid', None)
            write(state_path, state)
            print('COMPLETED ' + entry['id'], flush=True)
        state.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        write(study / 'resolved_training_checkpoints.json',
              dict(protocol_sha256=protocol_hash, checkpoints=state['completed']))
        print('ALL_TRAINING_COMPLETED', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
