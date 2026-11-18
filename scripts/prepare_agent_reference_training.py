"""Verify 32 admitted executions, split whole clips 24/4/4, optionally start training."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/supermimic_official_5efbf114/src')]
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.runtime/research_cache'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--training-output',type=Path,required=True)
    p.add_argument('--split-seed',type=int,default=20261114)
    p.add_argument('--launch',action='store_true')
    a=p.parse_args();run=a.collection.resolve();dataset=a.dataset.resolve();training=a.training_output.resolve()
    if dataset.exists() or training.exists():raise FileExistsError('Use fresh dataset and training directories')
    read=lambda f:json.loads(f.read_text())
    manifest=read(run/'accepted/manifest.json');status=read(run/'status.json');protocol=read(run/'protocol.json')
    if status['status']!='complete' or manifest['accepted_count']!=32 or len(manifest['entries'])!=32:
        raise ValueError('Exactly 32 accepted references are required before training')
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for array verification; no CPU fallback')
    torch.set_num_threads(4)
    from sesame_trajectory_agent.trajectories import FIELDS,export_accepted,sha256,write_json
    from sesame_trajectory_agent.validation import ExecutionEngine
    from mjlab.tasks.tracking.mdp.commands import MotionLoader
    robot=read(run/'context.json')['robot']
    library=SimpleNamespace(joint_names=robot['joint_names'],body_names=robot['physical_constraints']['body_names'])
    entries=[];hashes=set()
    for item in manifest['entries']:
        report=Path(item['validation']);trace=Path(item['trace']);r=read(report)
        if sha256(report)!=item['validation_sha256'] or sha256(trace)!=item['trace_sha256']:
            raise ValueError('Accepted manifest hash mismatch')
        if r['trajectory_sha256'] in hashes:raise ValueError('Duplicate motion')
        hashes.add(r['trajectory_sha256'])
        entries.append({'id':item['id'],'source_clip':item['source_clip'],'report':str(report),'trace':str(trace)})
    dataset.mkdir(parents=True)
    # Re-run the full export contract before deriving any training/test arrays.
    with tempfile.TemporaryDirectory(prefix='agent32_verified_',dir=ROOT/'.runtime') as temp:
        verified=Path(temp)/'accepted';export_accepted(verified,entries,library,run/'protocol.json')
        source=verified/'train';meta=read(source/'meta.json')
        data={k:torch.as_tensor(np.load(source/(k+'.npy'),allow_pickle=False),device='cuda:0') for k in FIELDS}
        if not all(bool(torch.isfinite(v).all()) for v in data.values()):raise ValueError('Nonfinite admitted arrays')
        ends=np.load(source/'motion_ends.npy',allow_pickle=False).tolist();starts=[0]+ends[:-1]
        order=list(range(32));random.Random(a.split_seed).shuffle(order)
        groups={'train':order[:24],'validation':order[24:28],'test':order[28:]};mapping={}
        for split,indices in groups.items():
            folder=dataset/split;folder.mkdir()
            total=0;new_ends=[]
            for i in indices:total+=ends[i]-starts[i];new_ends.append(total)
            for key in FIELDS:
                packed=torch.cat([data[key][starts[i]:ends[i]] for i in indices])
                np.save(folder/(key+'.npy'),packed.cpu().numpy(),allow_pickle=False)
            np.save(folder/'motion_ends.npy',np.asarray(new_ends,dtype=np.int64),allow_pickle=False)
            write_json(folder/'joint_names.json',library.joint_names)
            write_json(folder/'motion_names.json',[entries[i]['id'] for i in indices])
            write_json(folder/'meta.json',meta|{'clips':len(indices),'frames':total,'split':split,
                'duration_h':(total-len(indices))/meta['fps']/3600})
            mapping[split]=[{'index':i,'id':entries[i]['id'],'report':entries[i]['report'],
                'motion_sha256':read(Path(entries[i]['report']))['trajectory_sha256']} for i in indices]
    # Official loader/FK verification on CUDA, without collecting new trajectories.
    engine=ExecutionEngine(ROOT/protocol['config']['source'],protocol['config'])
    try:
        for split,rows in mapping.items():
            loader=MotionLoader(str(dataset/split),engine.fk,device='cuda:0')
            frames=loader.motion_ends-1;states=loader.body_state(frames)
            if len(frames)!=len(rows) or not all(bool(torch.isfinite(x).all()) for x in states):
                raise ValueError('Official motion loader verification failed')
    finally:engine.close()
    command=[str(ROOT/'.venv_supermimic_research/bin/python'),'-u',str(ROOT/'scripts/train_research_sesame.py'),
        '--method','supermimic','--output',str(training),'--motion-file',str(dataset/'train'),
        '--validation-file',str(dataset/'validation'),'--test-file',str(dataset/'test'),
        '--seed','42','--envs','512','--steps','2097152','--width','512','--batch','2048',
        '--horizon','32','--wm-batch','512','--wm-batches','10','--eval-every','262144',
        '--eval-steps','300','--physics-dt','0.0005','--replay-ratio','16',
        '--diagnostic-only-metrics','velocity_rmse_m_s','contact_foot_speed_rms_m_s']
    record={'source_collection':str(run),'source_protocol_sha256':sha256(run/'protocol.json'),
        'source_manifest_sha256':sha256(run/'accepted/manifest.json'),'total_unique_clips':32,
        'splits':mapping,'split_seed':a.split_seed,'split_rule':'whole programs/trajectories; no frame-level mixing',
        'limitations':'Gait parameter variants share a waveform family; this is not an unseen-skill generalization benchmark',
        'motion_loader_cuda_verified':True,'training_command':command,
        'array_sha256':{str(f.relative_to(dataset)):sha256(f) for split in mapping for f in (dataset/split).iterdir() if f.is_file()}}
    write_json(dataset/'dataset_protocol.json',record)
    if a.launch:
        log=training.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb') as output:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        write_json(dataset/'training_launch.json',{'pid':proc.pid,'command':command,'log':str(log),'output':str(training),'status':'launched'})
        print(json.dumps({'dataset':str(dataset),'clips':32,'train':24,'validation':4,'test':4,'training_pid':proc.pid,'log':str(log)}),flush=True)
    else:print(json.dumps({'dataset':str(dataset),'command':command}),flush=True)


if __name__=='__main__':main()
