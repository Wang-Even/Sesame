#!/usr/bin/env python3
"""Record three preregistered demos after the common evaluation completes."""
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


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if not study.is_relative_to(ROOT/'runs'):
        raise ValueError('Study must remain inside Sesame runs')
    videos = study/'videos'
    videos.mkdir(parents=True, exist_ok=True)
    lock = (videos/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = videos/'progress.json'
    if state_path.exists():
        raise RuntimeError('Existing video queue; inspect before retry')
    recorder = ROOT/'scripts/record_offpolicy_video.py'
    digest = hashlib.sha256(recorder.read_bytes()).hexdigest()
    state = dict(status='waiting_for_study', pid=os.getpid(), completed=[], active=None,
                 recording_script_sha256=digest, started_at_utc=datetime.now(timezone.utc).isoformat())
    write(state_path, state)
    env = os.environ.copy()
    env.update(CONDA_NO_PLUGINS='true', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    try:
        while True:
            progress = json.loads((study/'progress.json').read_text())
            if progress['status'] == 'failed':
                raise RuntimeError('Training/evaluation queue failed; demos not started')
            if progress['status'] == 'completed':
                break
            time.sleep(10)
        for policy_id in ('ppo_seed42', 'sac_seed42', 'dqn_seed42'):
            if hashlib.sha256(recorder.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Recorder changed after queue launch')
            out = videos/policy_id
            out.mkdir(parents=True, exist_ok=True)
            state.update(status='recording', active=policy_id)
            write(state_path, state)
            command = ['/home/ubuntu/miniforge3/bin/conda', 'run', '-n', 'env_isaaclab', '--no-capture-output',
                       '/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python', '-u', str(recorder),
                       '--protocol', str(study/'evaluation_protocol.json'), '--policy-id', policy_id,
                       '--output-dir', str(out)]
            with (out/'recording.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)
                state['child_pid'] = proc.pid
                write(state_path, state)
                code = proc.wait()
            state.pop('child_pid', None)
            if code:
                raise RuntimeError(f'{policy_id} recorder exited {code}; preserve recording.log')
            report_path = out/f'offpolicy_{policy_id}_seed25042.json'
            report = json.loads(report_path.read_text())
            if report['status'] != 'completed':
                raise RuntimeError('Video report incomplete')
            state['completed'].append(dict(policy_id=policy_id, report=str(report_path), video=report['video'],
                                           video_sha256=report['video_sha256']))
            write(state_path, state)
            print('VIDEO_COMPLETED '+policy_id, flush=True)
        state.update(status='completed', active=None, finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
