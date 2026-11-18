"""Serial CUDA rerun of section 6 with eight-times denser validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'.venv_supermimic_research/bin/python'


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/course_comparison_20261115')
    parser.add_argument('--suite', type=Path, default=ROOT/'runs/course_comparison_dense_20261116')
    parser.add_argument('--extended', type=Path, default=ROOT/'runs/course_ppo_extended_dense_20261116')
    parser.add_argument('--motion', type=Path, default=ROOT/'runs/course_motion_evaluation_dense_20261116')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    dataset, suite, extended, motion = [p.resolve() for p in
                                      (args.dataset,args.suite,args.extended,args.motion)]
    cache_root = ROOT/'.runtime/course_comparison_cache'/suite.name
    for path in (suite, extended, motion, cache_root):
        if path.exists():
            raise FileExistsError(f'Use a fresh output/cache path: {path}')
    if not all(p.is_relative_to(ROOT/'runs') for p in (suite, extended, motion)):
        raise ValueError('Outputs must be under Sesame runs')
    for name in ('original_train','deepseek_train','codex_train','common_validation','common_test'):
        if not (dataset/name/'meta.json').is_file():
            raise FileNotFoundError(dataset/name/'meta.json')
    audits=[]
    for name in ('supermimic_official_5efbf114','mjlab_research_dd51a9fc'):
        provenance=json.loads((ROOT/'third_party'/(name+'.provenance.json')).read_text())
        changed=[p for p,digest in provenance['files_sha256'].items()
                 if sha(ROOT/'third_party'/name/p)!=digest]
        if changed:raise RuntimeError(f'Upstream files changed: {name}: {changed}')
        audits.append(dict(source=name,commit=provenance['commit'],changed=changed))
    specs=[('ppo','ppo','original',32,False,2097152),
           ('supermimic','supermimic','original',32,False,2097152),
           ('supermimic_deepseek','supermimic','deepseek',32,False,2097152),
           ('supermimic_codex','supermimic','codex',32,False,2097152),
           ('supermimic_h16','supermimic','original',16,False,2097152),
           ('supermimic_h64','supermimic','original',64,False,2097152),
           ('supermimic_no_critic','supermimic','original',32,True,2097152),
           ('ppo_extended','ppo','original',32,False,8388608)]
    runs=[]
    for name,method,source,horizon,no_critic,steps in specs:
        output = extended if name=='ppo_extended' else suite/name
        cmd=[str(PYTHON),'-u',str(ROOT/'scripts/train_research_sesame.py'),
             '--method',method,'--output',str(output),
             '--motion-file',str(dataset/(source+'_train')),
             '--validation-file',str(dataset/'common_validation'),
             '--test-file',str(dataset/'common_test'),
             '--seed','42','--envs','512','--steps',str(steps),'--width','512','--batch','2048',
             '--horizon',str(horizon),'--wm-batch','512','--wm-batches','10',
             '--eval-every','32768','--eval-steps','300','--eval-after-first-update',
             '--physics-dt','0.0005','--replay-ratio','16','--sampling-mode','uniform',
             '--diagnostic-only-metrics','velocity_rmse_m_s','contact_foot_speed_rms_m_s']
        if no_critic:cmd.append('--no-critic')
        runs.append(dict(name=name,source=source,output=str(output),command=cmd))
    source_paths=[Path(__file__),ROOT/'scripts/train_research_sesame.py',*sorted((ROOT/'sesame_research').glob('*.py'))]
    protocol=dict(dataset=str(dataset),dataset_protocol_sha256=sha(dataset/'protocol.json'),
        runs=runs,seed_count=1,seed=42,execution='serial on one CUDA GPU',
        eval_every_real_steps=32768,previous_eval_every_real_steps=262144,
        first_update_evaluation=True,training_budgets_unchanged=True,
        time='Sampling and update clock excludes evaluation and checkpoint overhead; separate startup and post-startup clocks retained.',
        primary_plot='Actual first-update evaluation at post-startup time zero, then dense validation; continuous EMA curves.',
        evaluation='12 held-out validation clips x 300 steps; final test on separate 12 clips; all sources unchanged',
        cache_root=str(cache_root),source_audit=audits,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        motion_output=str(motion),extended_output=str(extended))
    if args.check_config:
        print(json.dumps(protocol,ensure_ascii=False));return
    suite.mkdir(parents=True)
    write(suite/'protocol.json',protocol)
    (suite/'launcher_source.py').write_bytes(Path(__file__).read_bytes())
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),completed=[],failed=[],total=8)
    write(suite/'status.json',state)
    for run in runs:
        # Changes to report/plot scripts cannot alter these frozen training sources.
        for relative,digest in protocol['sources_sha256'].items():
            if sha(ROOT/relative)!=digest:raise RuntimeError(f'Training source changed: {relative}')
        name=run['name'];cache=cache_root/name
        env=os.environ.copy()
        env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                   XDG_CACHE_HOME=str(ROOT/'.runtime/research_cache'),WANDB_MODE='disabled',OMP_NUM_THREADS='4')
        with (suite/(name+'_gpu_before.csv')).open('w') as f:
            subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],stdout=f,check=True)
        with (suite/(name+'.log')).open('x') as f:
            child=subprocess.Popen(run['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state.update(current=name,child_pid=child.pid,current_started_unix=time.time())
            write(suite/'status.json',state)
            code=child.wait()
        result_path=Path(run['output'])/'status.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        if code or result.get('status')!='complete':
            state['failed'].append(dict(name=name,returncode=code,status=result.get('status')))
            state.update(status='failed',finished_unix=time.time());write(suite/'status.json',state)
            raise RuntimeError(f'{name} failed; inspect {suite/(name+".log")}')
        state['completed'].append(name);write(suite/'status.json',state)
    state.update(current='motion_evaluation');write(suite/'status.json',state)
    with (suite/'motion_evaluation.log').open('x') as f:
        child=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'scripts/evaluate_course_motion.py'),
                                '--suite',str(suite),'--output',str(motion)],cwd=ROOT,
                               stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        state['child_pid']=child.pid;write(suite/'status.json',state);code=child.wait()
    state.update(status='complete' if code==0 else 'failed',motion_returncode=code,finished_unix=time.time())
    write(suite/'status.json',state)
    if code:raise RuntimeError('Motion reevaluation failed; inspect motion_evaluation.log')


if __name__=='__main__':main()
