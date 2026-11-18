"""Fresh, longer-budget PPO supplement after the serial course experiments.

This does not replace the predeclared equal-environment-step comparison.
Run only after other GPU experiments have finished.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT/'runs/course_comparison_20261115')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/course_ppo_extended_20261115')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    suite, output = args.suite.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Output must remain under Sesame runs')
    source = read(suite/'ppo/protocol.json')
    if read(suite/'ppo/status.json')['status'] != 'complete':
        raise ValueError('Original fixed-budget PPO must be complete')
    for relative, sha in source['sources'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Training source changed since original comparison: {relative}')
    command = list(read(suite/'protocol.json')['runs'][0]['command'])
    assert command[command.index('--method')+1] == 'ppo'
    command[command.index('--output')+1] = str(output)
    command[command.index('--steps')+1] = '8388608'
    cache = ROOT/'.runtime/course_comparison_cache'/output.name
    if output.exists() or cache.exists():
        raise FileExistsError('Use a fresh output/cache name')
    protocol = dict(
        motivation='Supplement added after the first fixed-budget results: extend the PPO time axis to cover the SuperMimic training window',
        prospective_steps=8388608, original_steps=2097152,
        changed_setting='Only the interaction budget; same training and evaluation configuration',
        comparison_scope='Additional time-window comparison; does not replace equal-step main results',
        initialization='Fresh seed42 run, not resumed or stitched onto the earlier PPO curve',
        cache='Independent fresh Inductor/Triton directories',
        original_protocol_sha256=hashlib.sha256((suite/'ppo/protocol.json').read_bytes()).hexdigest(),
        suite=str(suite), command=command, cache_root=str(cache), seed_count=1)
    if args.check_config:
        print(json.dumps(protocol, ensure_ascii=False)); return
    if read(suite/'status.json')['status'] != 'complete':
        raise ValueError('Wait for all seven main runs before using the GPU')
    # The motion-study owner releases the GPU only after its queue is complete.
    motion_status = ROOT/'runs/ppo_motion_study_20261115/queue_status.json'
    if not motion_status.exists() or read(motion_status)['status'] != 'completed':
        raise ValueError('Wait for the authorized Isaac motion study to finish')
    output.mkdir(parents=True)
    write(output/'supplement_protocol.json', protocol)
    env = os.environ.copy()
    env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
               TRITON_CACHE_DIR=str(cache/'triton'),
               XDG_CACHE_HOME=str(ROOT/'.runtime/research_cache'),
               WANDB_MODE='disabled', OMP_NUM_THREADS='4')
    with (output/'gpu_before.csv').open('w') as f:
        subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv'], stdout=f, check=True)
    state = dict(status='running', pid=os.getpid(), started_unix=time.time())
    with (output/'training.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state['child_pid'] = child.pid
        write(output/'launcher_status.json', state)
        code = child.wait()
    complete = (code == 0 and (output/'status.json').exists()
                and read(output/'status.json')['status'] == 'complete')
    state.update(status='complete' if complete else 'failed', returncode=code,
                 finished_unix=time.time())
    write(output/'launcher_status.json', state)
    if not complete:
        raise RuntimeError('Extended PPO failed; inspect preserved training.log')


if __name__ == '__main__':
    main()
