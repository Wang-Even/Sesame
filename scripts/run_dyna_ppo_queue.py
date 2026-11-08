#!/usr/bin/env python3
"""Sequential, resumable launch of the three preregistered Dyna-PPO CUDA runs."""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONDA='/home/ubuntu/miniforge3/bin/conda'
PYTHON='/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python'


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def checked_result(entry):
    path=Path(entry['run_dir'])/'result.json'
    if not path.exists(): return None
    d=json.loads(path.read_text())
    if d['status']!='completed': raise RuntimeError(f"Existing run {entry['id']} is {d['status']}; inspect before any retry")
    assert d['environment_transitions_total']==2048000
    checkpoint=Path(entry['checkpoint'])
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha==d['checkpoint_sha256']
    assert entry.get('sha256',sha)==sha
    assert d['reload_max_abs_action_error']==0
    if entry['variant']=='dyna_ppo':
        assert d['algorithm']=='DynaPPO'
        assert d['dyna_training_summary']['imagined_optimizer_steps']>0
        assert d['world_model_reload_max_abs_prediction_error']==0
    return {'id':entry['id'],'variant':entry['variant'],'train_seed':entry['train_seed'],
            'checkpoint':str(checkpoint),'sha256':sha,'reused_training':entry.get('reused_training',False)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--study-dir',type=Path,required=True);args=ap.parse_args()
    study=args.study_dir.resolve();protocol_bytes=(study/'protocol.json').read_bytes();protocol=json.loads(protocol_bytes)
    lock=(study/'training_queue.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    sha=hashlib.sha256(protocol_bytes).hexdigest()
    for relative,expected in protocol['frozen_sources'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative
    assert sha==(study/'protocol.sha256').read_text().split()[0]
    state={'status':'running','protocol_sha256':sha,'started_at_utc':datetime.now(timezone.utc).isoformat(),'completed':[],'active':None}
    status_path=study/'training_progress.json'
    env=os.environ.copy();env['CONDA_NO_PLUGINS']='true';env['PYTHONDONTWRITEBYTECODE']='1'
    try:
        # Verify all existing results before launching; never run a duplicate worker.
        done={}
        for entry in protocol['checkpoints']:
            result=checked_result(entry)
            if result: done[entry['id']]=result
        state['completed']=list(done.values());write(status_path,state)
        for entry in protocol['checkpoints']:
            if entry['id'] in done: continue
            state['active']=entry['id'];write(status_path,state)
            run=Path(entry['run_dir']);run.mkdir(parents=True,exist_ok=True)
            command=[CONDA,'run','-n','env_isaaclab','--no-capture-output',PYTHON,'-u',str(ROOT/'scripts/train_dyna_ppo.py'),
                     '--variant',entry['variant'],'--seed',str(entry['train_seed']),'--run-dir',str(run),
                     '--dyna-config',protocol['training']['dyna_config_reference']]
            print('START '+entry['id'],flush=True)
            with (run/'train.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['active_process_pid']=process.pid;write(status_path,state)
                exit_code=process.wait()
            if exit_code!=0: raise RuntimeError(f"{entry['id']} exited {exit_code}; see {run/'train.log'}")
            done[entry['id']]=checked_result(entry)
            state['completed']=list(done.values());state['active']=None;state.pop('active_process_pid',None);write(status_path,state)
            print('COMPLETED '+entry['id'],flush=True)
        state['status']='completed';state['finished_at_utc']=datetime.now(timezone.utc).isoformat();write(status_path,state)
        write(study/'resolved_training_checkpoints.json',{'protocol_sha256':sha,'checkpoints':state['completed']})
        print('ALL_TRAINING_COMPLETED',flush=True)
    except BaseException as exc:
        state['status']='failed';state['error']=repr(exc);write(status_path,state)
        raise


if __name__=='__main__': main()
