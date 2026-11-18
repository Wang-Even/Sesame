"""Build frozen 6-second reference sets and shared held-out splits on CUDA."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def candidate_file(run, recorded):
    path = Path(recorded)
    if 'candidates' in path.parts:
        local_path = run.joinpath(*path.parts[path.parts.index('candidates'):])
        if local_path.is_file():
            return local_path
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deepseek',type=Path,required=True,help='Complete accepted collection root')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    # Data packing is performed on CUDA; NumPy is used only for file input/output.
    cache={}
    def load(folder):
        folder=folder.resolve()
        if folder not in cache:
            meta=read(folder/'meta.json');assert meta['fps']==50
            ends=np.load(folder/'motion_ends.npy',allow_pickle=False).tolist()
            starts=[0]+ends[:-1]
            data={k:torch.as_tensor(np.load(folder/(k+'.npy'),allow_pickle=False),device='cuda:0') for k in FIELDS}
            if not all(bool(torch.isfinite(x).all()) for x in data.values()):raise ValueError('Nonfinite data')
            names=read(folder/'motion_names.json')
            cache[folder]=(meta,starts,ends,data,names)
        return cache[folder]
    groups={};orig=ROOT/'datasets/supermimic_research_sesame_20261112'
    for split,n in [('train',24),('validation',4),('test',4)]:
        groups['original_'+split]=[(orig/split,i,'original') for i in range(n)]
    generated={
        'codex':ROOT/'runs/agent32_collection_20261114_reassessed_20261115',
        'deepseek':a.deepseek.resolve(),
    }
    collection_sources={};reference_hashes={};gate_configs={}
    for label,run in generated.items():
        status=read(run/'status.json');manifest=read(run/'accepted/manifest.json')
        assert status['status']=='complete' and manifest['accepted_count']>=32,(run,status)
        folder=run/'accepted/train'
        meta,_,_,_,names=load(folder)
        assert len(names)>=32
        # First 32 accepted in candidate order; fixed before viewing new training outcomes.
        chosen=sorted(range(len(names)),key=lambda i:names[i])[:32]
        manifest_by_id={item['id']:item for item in manifest['entries']}
        reference_hashes[label]={}
        for index in chosen:
            item=manifest_by_id[names[index]]
            report=candidate_file(run,item['validation']);trace=candidate_file(run,item['trace'])
            if sha(report)!=item['validation_sha256'] or sha(trace)!=item['trace_sha256']:
                raise ValueError('Collection provenance changed after acceptance')
            reference_hashes[label][names[index]]=read(report)['trajectory_sha256']
        source_cfg=read(run/'protocol.json')['config']
        gate_configs[label]={k:source_cfg[k] for k in ('seed','gates','duration_s','fps','robust_trials',
            'physics_dt_s','diagnostic_only_metrics','mean_speed_measurement','execution_feedback')}
        random.Random(20261115).shuffle(chosen)
        for split,ids in [('train',chosen[:24]),('validation',chosen[24:28]),('test',chosen[28:])]:
            groups[label+'_'+split]=[(folder,i,label) for i in ids]
        collection_sources[label]={'run':str(run),'manifest_sha256':sha(run/'accepted/manifest.json'),
            'protocol_sha256':sha(run/'protocol.json'),'accepted_available':len(names),'selected':32}
    assert gate_configs['deepseek']==gate_configs['codex'],'Agent groups must share execution and acceptance settings'
    assert not (set(reference_hashes['deepseek'].values()) & set(reference_hashes['codex'].values())), 'Identical generated references across Agent sources'
    for split in ('validation','test'):
        groups['common_'+split]=sum([groups[label+'_'+split] for label in ('original','deepseek','codex')],[])
    a.output.mkdir(parents=True)
    mapping={}
    for name,items in groups.items():
        out=a.output/name;out.mkdir()
        parts={k:[] for k in FIELDS};entries=[];meta0=None
        for folder,i,label in items:
            meta,starts,ends,data,names=load(folder)
            if ends[i]-starts[i]<301:raise ValueError('Every clip must contain 301 frames for 300 evaluation steps')
            if meta0 is not None:assert meta0['joint_names']==meta['joint_names'] and meta0['body_names']==meta['body_names']
            meta0=meta
            digest=hashlib.sha256()
            for key in FIELDS:
                clip=data[key][starts[i]:starts[i]+301]
                parts[key].append(clip)
                digest.update(clip.cpu().contiguous().numpy().tobytes())
            entries.append({'source':label,'name':names[i],'source_directory':str(folder),
                            'source_index':i,'frames':[starts[i],starts[i]+301], 'content_sha256':digest.hexdigest()})
        for key in FIELDS:np.save(out/(key+'.npy'),torch.cat(parts[key]).cpu().numpy(),allow_pickle=False)
        np.save(out/'motion_ends.npy',np.asarray([301*(i+1) for i in range(len(items))],dtype=np.int64),allow_pickle=False)
        write(out/'joint_names.json',meta0['joint_names'])
        write(out/'motion_names.json',[x['source']+'__'+x['name'] for x in entries])
        write(out/'meta.json',{'joint_names':meta0['joint_names'],'body_names':meta0['body_names'],
            'fps':50,'clips':len(items),'frames':301*len(items),'duration_h':6*len(items)/3600,
            'description':'Frozen 6-second clips; original first 6 seconds and accepted physical Agent executions'})
        write(out/'provenance.json',entries);mapping[name]=entries
    trained={e['content_sha256'] for name,items in mapping.items() if name.endswith('_train') for e in items}
    val={e['content_sha256'] for e in mapping['common_validation']}
    test={e['content_sha256'] for e in mapping['common_test']}
    assert len(trained)==72 and len(val)==12 and len(test)==12
    assert not (trained&val or trained&test or val&test),'Cross-split duplicate content'
    protocol={'selection_seed':20261115,'collection_sources':collection_sources,'mapping':mapping,
        'reference_sha256':reference_hashes,'shared_agent_acceptance':gate_configs['codex'],
        'train_clips_per_method':24,'train_duration_s':144,'common_validation_clips':12,'common_test_clips':12,
        'common_eval_order':['original 4','deepseek 4','codex 4'],
        'heldout':'whole trajectories, content hashes checked disjoint across all new training/validation/test splits',
        'limitations':['Agent programs are related gait variants, not independent unseen skills',
            'Historical reference generation was informed by earlier development experiments',
            'Original references originated from a learned Isaac policy; its training cost is a prior shared resource'],
        'array_sha256':{str(f.relative_to(a.output)):sha(f) for f in a.output.rglob('*') if f.is_file()}}
    write(a.output/'protocol.json',protocol)
    print(json.dumps({'dataset':str(a.output),'groups':{k:len(v) for k,v in mapping.items()},'disjoint':True}))


if __name__=='__main__':main()
