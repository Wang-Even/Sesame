"""Serial, fixed-budget current-reward experiments for the course draft."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv_supermimic_research/bin/python'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.dataset=a.dataset.resolve();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    cache_root=ROOT/'.runtime/course_comparison_cache'/a.output.name
    if cache_root.exists():raise FileExistsError('Choose a suite name without an existing compilation cache')
    a.output.mkdir(parents=True)
    audits=[]
    for name in ('supermimic_official_5efbf114','mjlab_research_dd51a9fc'):
        provenance=json.loads((ROOT/'third_party'/(name+'.provenance.json')).read_text())
        changed=[rel for rel,digest in provenance['files_sha256'].items()
                 if hashlib.sha256((ROOT/'third_party'/name/rel).read_bytes()).hexdigest()!=digest]
        if changed:raise RuntimeError(f'Upstream source differs: {name}: {changed}')
        audits.append({'source':name,'commit':provenance['commit'],'files_checked':len(provenance['files_sha256']),'changed':changed})
    write(a.output/'source_audit.json',audits)
    # One seed in the first draft; these are configuration ablations, not seed repetitions.
    specs=[('ppo','ppo','original',32,False),
           ('supermimic','supermimic','original',32,False),
           ('supermimic_deepseek','supermimic','deepseek',32,False),
           ('supermimic_codex','supermimic','codex',32,False),
           ('supermimic_h16','supermimic','original',16,False),
           ('supermimic_h64','supermimic','original',64,False),
           ('supermimic_no_critic','supermimic','original',32,True)]
    commands=[]
    for name,method,source,horizon,no_critic in specs:
        cmd=[str(PYTHON),'-u',str(ROOT/'scripts/train_research_sesame.py'),
             '--method',method,'--output',str(a.output/name),
             '--motion-file',str(a.dataset/(source+'_train')),
             '--validation-file',str(a.dataset/'common_validation'),
             '--test-file',str(a.dataset/'common_test'),
             '--seed','42','--envs','512','--steps','2097152','--width','512','--batch','2048',
             '--horizon',str(horizon),'--wm-batch','512','--wm-batches','10',
             '--eval-every','262144','--eval-steps','300','--physics-dt','0.0005',
             '--replay-ratio','16','--sampling-mode','uniform',
             '--diagnostic-only-metrics','velocity_rmse_m_s','contact_foot_speed_rms_m_s']
        if no_critic:cmd.append('--no-critic')
        commands.append({'name':name,'source':source,'command':cmd})
    protocol={'dataset':str(a.dataset),'dataset_protocol_sha256':hashlib.sha256((a.dataset/'protocol.json').read_bytes()).hexdigest(),
        'runs':commands,'seed_count':1,'execution':'serial on one CUDA GPU',
        'time':'sampling+updates, including fresh per-run Inductor/Triton compilation; initialization/evaluation separately reported',
        'primary_plot':'common validation trajectory return versus training seconds',
        'final_table':'final.pt on the same 12 held-out clips, 4 from each source',
        'gamma':.99,'no_critic_ablation':'gamma fixed to .99 for both actor-critic and no-critic',
        'scope':'local motion-tracking research, not course official scoring','cache_root':str(cache_root)}
    write(a.output/'protocol.json',protocol)
    status={'status':'running','pid':os.getpid(),'completed':[],'failed':[],'started_unix':time.time()}
    write(a.output/'status.json',status)
    for row in commands:
        env=os.environ.copy()
        cache=cache_root/row['name']
        env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                   XDG_CACHE_HOME=str(ROOT/'.runtime/research_cache'),WANDB_MODE='disabled',OMP_NUM_THREADS='4')
        subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],
                       stdout=(a.output/(row['name']+'_gpu_before.csv')).open('w'),check=False)
        status.update(current=row['name']);write(a.output/'status.json',status)
        with (a.output/(row['name']+'.log')).open('xb') as log:
            process=subprocess.Popen(row['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.update(child_pid=process.pid);write(a.output/'status.json',status)
            rc=process.wait()
        result_path=a.output/row['name']/'status.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'missing_status'}
        if rc!=0 or result['status']!='complete':
            status['failed'].append({'name':row['name'],'returncode':rc,'child_status':result['status']})
            status.update(status='failed',finished_unix=time.time());write(a.output/'status.json',status)
            sys.exit(1)
        status['completed'].append(row['name']);write(a.output/'status.json',status)
    status.update(status='complete',finished_unix=time.time());write(a.output/'status.json',status)


if __name__=='__main__':main()
