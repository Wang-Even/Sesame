#!/usr/bin/env python3
"""Validate saved states, actuator timing, splits and window loading on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sesame_rl.motion_dataset import SesameMotionDataset, SesameReferenceDataset


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args=parser.parse_args()
    root=args.dataset.resolve()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for numerical validation')
    torch.set_num_threads(4)
    schema=json.loads((root/'schema.json').read_text())
    config=json.loads((root/'environment_config.json').read_text())
    rows=json.loads((root/'combined_episode_index.json').read_text())
    refs=json.loads((root/'reference_index.json').read_text())
    checkfiles=json.loads((root/'checksums.json').read_text())
    for record in checkfiles:
        if digest(root/record['path'])!=record['sha256']: raise AssertionError(f'Hash mismatch: {record["path"]}')
    device='cuda:0'
    dt=.02
    stand=torch.tensor([config['robot']['init_state']['joint_pos'][k] for k in schema['joint_order']],device=device)
    limits=torch.tensor([[45,180],[0,135],[0,135],[45,180],[0,180],[0,180],[0,180],[0,180]],device=device)*torch.pi/180
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['path'] for r in rows})==len(rows)
    seed_sets={s:{r['reset_seed'] for r in rows if r['split']==s} for s in ('train','validation','test')}
    hashes={s:{r['sha256'] for r in rows if r['split']==s} for s in seed_sets}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        assert not seed_sets[a]&seed_sets[b], 'Reset seeds overlap across splits'
        assert not hashes[a]&hashes[b], 'Identical episode data overlaps across splits'
    max_error={'clip':0.0,'previous_action':0.0,'command_target':0.0,'history':0.0,'applied_target':0.0,'root_quaternion_norm':0.0,'body_quaternion_norm':0.0}
    nonzero_contact_episodes=0
    fall_windows=0
    initial_pairs={}
    paired_episodes_checked=0
    for index,row in enumerate(rows):
        with np.load(root/row['path'],allow_pickle=False) as f:
            data={k:torch.as_tensor(f[k],device=device) for k in f.files}
        T=row['transitions']
        for key,value in data.items():
            assert len(value)==T+(key.startswith('state_')), (row['id'],key,value.shape,T)
            if row['world_model_usable'] and value.is_floating_point(): assert bool(torch.isfinite(value).all()),row['id']
        done=data['terminated']|data['truncated']
        assert not bool(done[:-1].any()), 'Reset crossed'
        assert bool(done[-1]) or row['collector_cutoff'], 'Missing episode boundary'
        assert not bool((data['terminated']&data['truncated']).any())
        assert not row['collector_cutoff'], 'Production episode was cut before terminal/timeout'
        assert T<=500
        assert row['continuity_max_abs']<=1e-5
        if not row['world_model_usable']: continue
        meta=json.loads((root/row['metadata_path']).read_text())
        p=meta['domain_parameters']
        pair_key=(row['split'],row['reset_seed'],row['env_id'],row['profile'],row['condition'])
        initial_state=torch.cat((data['state_root_state'][0],data['state_joint_pos'][0],data['state_joint_vel'][0]))
        if pair_key in initial_pairs:
            earlier,earlier_domain=initial_pairs.pop(pair_key)
            assert torch.equal(initial_state,earlier), ('Policy-paired initial states differ',row['id'])
            assert earlier_domain==p, ('Policy-paired domain context differs',row['id'])
            paired_episodes_checked+=2
        else: initial_pairs[pair_key]=(initial_state.clone(),p)
        q0=data['state_command_target'][:-1]
        targets0=data['state_applied_target'][:-1]
        clipped=data['input_action'].clamp(-1,1)
        desired=(stand+.58*clipped).clamp(limits[:,0],limits[:,1])
        command_target=q0+(desired-q0).clamp(-10.471975512*dt*p['speed_scale'],10.471975512*dt*p['speed_scale'])
        history=torch.cat((command_target[:,None],data['state_target_history'][:-1,:-1]),dim=1)
        if row['condition']=='nominal':
            targets=targets0+(desired-targets0).clamp(-10.471975512*dt,10.471975512*dt)
        else:
            alpha=1-torch.exp(torch.tensor(-dt/max(p['target_tau_s'],1e-9),device=device))
            targets=targets0+alpha*(history[:,int(p['delay_control_steps'])]-targets0)
        errors={'clip':(clipped-data['executed_action']).abs().max(),
                'previous_action':(data['state_previous_action'][1:]-data['executed_action']).abs().max(),
                'command_target':(command_target-data['state_command_target'][1:]).abs().max(),
                'history':(history-data['state_target_history'][1:]).abs().max(),
                'applied_target':(targets-data['state_applied_target'][1:]).abs().max(),
                'root_quaternion_norm':(data['state_root_state'][:,3:7].norm(dim=-1)-1).abs().max(),
                'body_quaternion_norm':(data['state_body_link_state'][...,3:7].norm(dim=-1)-1).abs().max()}
        for key,value in errors.items():
            max_error[key]=max(max_error[key],float(value.item()))
            assert float(value.item())<2e-5,(row['id'],key,float(value.item()))
        # Independently rebuild both policy observations from the stored raw
        # states and the step's command, including terminal pre-reset states.
        def inverse_rotate(quat,vector):
            vector_part=-quat[...,1:]
            uv=2*torch.linalg.cross(vector_part,vector,dim=-1)
            return vector+quat[...,:1]*uv+torch.linalg.cross(vector_part,uv,dim=-1)
        for state_slice,obs_key in ((slice(None,-1),'observation'),(slice(1,None),'next_observation')):
            root_state=data['state_root_state'][state_slice]
            quat=root_state[:,3:7]
            gravity=torch.zeros_like(root_state[:,7:10]); gravity[:,2]=-1
            expected=torch.cat((inverse_rotate(quat,root_state[:,7:10])*5,
                inverse_rotate(quat,root_state[:,10:13])*.25, inverse_rotate(quat,gravity),
                inverse_rotate(quat,data['command_world'])*5,
                data['state_joint_pos'][state_slice]-stand,
                data['state_joint_vel'][state_slice]/10.471975512,
                data['state_previous_action'][state_slice]),dim=-1).clamp(-20,20)
            error=float((expected-data[obs_key]).abs().max().item())
            max_error[obs_key]=max(max_error.get(obs_key,0),error)
            assert error<3e-5,(row['id'],obs_key,error)
        has_contact=bool((data['body_contact_force'].norm(dim=-1)>0).any())
        if T==500: assert has_contact, ('No contact recorded in complete walking episode', row['id'])
        nonzero_contact_episodes+=int(has_contact)
        fall_windows+=int(row['metrics']['fell'] and T>=8)
        if index%200==0: print(f'VALIDATED_EPISODES={index+1}/{len(rows)}',flush=True)
    # Reference fields must exactly match the claimed source segment; no hidden
    # smoothing, time scaling or relabeling an unsuccessful high-speed attempt.
    row_map={r['path']:r for r in rows}
    for ref in refs:
        source=row_map[ref['source_episode']]
        assert source['split']==ref['split'] and not source['reference_rejection_reasons']
        assert source['sha256']==ref['source_sha256']
        with np.load(root/ref['path'],allow_pickle=False) as f, np.load(root/source['path'],allow_pickle=False) as episode:
            assert int(f['fps'])==50
            start=ref['source_state_start_index']
            for key,orig,slice_ in [('joint_pos','state_joint_pos',None),('joint_vel','state_joint_vel',None),
                                    ('root_pos','state_root_state',slice(0,3)),('root_quat','state_root_state',slice(3,7)),
                                    ('foot_pos','state_foot_pos',None),('foot_vel','state_foot_vel',None)]:
                a=torch.as_tensor(f[key],device=device)
                b=torch.as_tensor(episode[orig][start:],device=device)
                if slice_ is not None: b=b[...,slice_]
                assert torch.equal(a,b),(ref['id'],key)
    loader_checks=[]
    reference_loader_checks=[]
    for split in ('train','validation','test'):
        for horizon in (8,32):
            dataset=SesameMotionDataset(root,split=split,horizon=horizon,seed=123)
            batch=dataset.sample(16)
            assert batch['state'].shape==(16,horizon+1,77)
            assert batch['executed_action'].shape==(16,horizon,8)
            assert batch['context'].shape==(16,9)
            for value in batch.values():
                if isinstance(value,torch.Tensor): assert value.is_cuda
            allowed={r['id'] for r in rows if r['split']==split}
            assert set(batch['episode_ids'])<=allowed
            loader_checks.append({'split':split,'horizon':horizon,'state_shape':list(batch['state'].shape),'cuda':True})
            del dataset,batch
        reference_dataset=SesameReferenceDataset(root,split=split,future_frames=10)
        reference_batch=reference_dataset.sample(16)
        assert reference_batch['future']['joint_pos'].shape==(16,10,8)
        assert all(t.is_cuda for t in reference_batch['future'].values())
        allowed={r['id'] for r in refs if r['split']==split}
        assert set(reference_batch['motion_ids'])<=allowed
        reference_loader_checks.append({'split':split,'future_frames':10,'joint_pos_shape':[16,10,8],'cuda':True})
        del reference_dataset,reference_batch
    report={'status':'passed','device':torch.cuda.get_device_name(0),'files_hashed':len(checkfiles),
            'checksums_sha256':digest(root/'checksums.json'),
            'episodes_checked':len(rows),'references_checked':len(refs),'split_seed_disjoint':True,
            'no_exact_duplicate_episodes_across_splits':True,'no_auto_reset_transitions':True,
            'actuator_pipeline_max_abs_error':max_error,'nonzero_contact_episodes':nonzero_contact_episodes,
            'fall_episodes_with_at_least_8_transitions':fall_windows,'loader_checks':loader_checks,
            'reference_loader_checks':reference_loader_checks,
            'reference_exact_source_match':True,
            'policy_paired_episodes_checked':paired_episodes_checked,
            'limitations':['No simulation replay equivalence claim: PhysX hidden solver state is not serialized.',
                           'State is a practical dynamics-model input; it does not serialize all simulator internals.',
                           'No test of a trained SuperMimic policy; dataset/loader validation only.']}
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
