#!/usr/bin/env python3
"""Sequential fresh PPO motion-study training, evaluation and reporting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
CONDA='/home/ubuntu/miniforge3/bin/conda'
PYTHON='/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'


def read(path):return json.loads(path.read_text())


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,required=True)
    args=parser.parse_args();study=args.study_dir.resolve()
    if not study.is_relative_to(ROOT/'runs'):raise ValueError('Study must be within Sesame runs')
    if (study/'queue_status.json').exists():raise FileExistsError('Use a fresh study directory')
    study.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CONDA_NO_PLUGINS='true',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    commands=[]
    for variant in ['baseline','smooth','fast']:
        folder=study/'training'/f'{variant}_seed42';folder.mkdir(parents=True,exist_ok=True)
        commands.append((variant,[CONDA,'run','-n','env_isaaclab','--no-capture-output',PYTHON,'-u',
            str(ROOT/'scripts/train_ppo_motion_study.py'),'--experiment',variant,'--seed','42','--run-dir',str(folder)]))
    commands.append(('evaluation',[CONDA,'run','-n','env_isaaclab','--no-capture-output',PYTHON,'-u',
        str(ROOT/'scripts/evaluate_ppo_motion_study.py'),'--study-dir',str(study)]))
    commands.append(('report',['python3',str(ROOT/'scripts/summarize_ppo_motion_study.py'),'--study-dir',str(study)]))
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/'scripts/train_ppo_motion_study.py',ROOT/'scripts/evaluate_ppo_motion_study.py',
        ROOT/'scripts/summarize_ppo_motion_study.py',ROOT/'scripts/run_ppo_motion_study.py']}
    write(study/'protocol.json',dict(seed=42,training_steps_per_arm=2048000,parallel_envs=256,
        rollout_steps=32,ppo_updates=250,heading_coefficient=.5,domain_randomization=True,
        arms=[dict(name='baseline',speed=.1,action_rate=.08),dict(name='smooth',speed=.1,action_rate=.16),
              dict(name='fast',speed=.2,action_rate=.08)],
        common_evaluation_action_rate=.08,commands=commands,sources=sources,
        note='Exploratory single-training-seed experiment. Existing baseline results not substituted.'))
    state=dict(status='running',pid=os.getpid(),completed=[],started_unix=time.time())
    write(study/'queue_status.json',state)
    try:
        for stage,command in commands:
            state.update(stage=stage,command=command)
            with (study/f'{stage}.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['child_pid']=process.pid;write(study/'queue_status.json',state)
                code=process.wait()
            if code:raise RuntimeError(f'{stage} returned {code}; see {stage}.log')
            if stage in ('baseline','smooth','fast'):
                result=read(study/'training'/f'{stage}_seed42'/'result.json')
                if result['status']!='completed' or result['environment_transitions_total']!=2048000:
                    raise RuntimeError('Incomplete or wrong-budget training')
                previous=[read(study/'training'/f'{v}_seed42'/'result.json') for v in state['completed'] if v in ('baseline','smooth','fast')]
                if any(r['initial_network_sha256']!=result['initial_network_sha256'] or r['initial_condition_sha256']!=result['initial_condition_sha256'] for r in previous):
                    raise RuntimeError('Initial network/state/domain mismatch between treatment arms')
            state['completed'].append(stage);write(study/'queue_status.json',state)
        state.update(status='completed',finished_unix=time.time());write(study/'queue_status.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write(study/'queue_status.json',state);raise


if __name__=='__main__':main()
