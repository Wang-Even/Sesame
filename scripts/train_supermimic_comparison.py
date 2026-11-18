#!/usr/bin/env python3
"""Train one small Sesame SuperMimic/PPO arm with fixed success criteria.

The SuperMimic arm warms its dynamics on the existing TRAIN transitions.
Those 300,847 supplied transitions count toward its real-data budget. Both
arms learn their actor from scratch with identical references/reward/evaluation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from train_ppo import ROOT,disable_git_imports,write_json
from train_ppo_study import require_runtime,sha256

SOURCES=('scripts/train_supermimic_comparison.py','sesame_rl/imitation_core.py',
         'sesame_rl/imitation_environment.py','sesame_rl/supermimic_small.py',
         'sesame_rl/environment.py','sesame_rl/dr_environment.py')


def append(path,record):
    with path.open('a') as file: file.write(json.dumps(record,allow_nan=False)+'\n')


class MemoryMonitor:
    def __init__(self):
        self.stop=threading.Event(); self.peak_process_mib=0; self.peak_total_mib=0
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def _run(self):
        while not self.stop.is_set():
            try:
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                for row in text.splitlines():
                    pid,value=[s.strip() for s in row.split(',')]
                    if int(pid)==os.getpid(): self.peak_process_mib=max(self.peak_process_mib,int(value))
                used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                self.peak_total_mib=max(self.peak_total_mib,int(used.splitlines()[0]))
            except (ValueError,subprocess.SubprocessError): pass
            self.stop.wait(1)


def protocol(root,dataset,smoke):
    from sesame_rl.imitation_core import SUCCESS,REWARD
    summary=json.loads((dataset/'summary.json').read_text())
    return {'schema_version':1,'study':'Small Sesame SuperMimic adaptation versus matched reference-tracking PPO',
        'created_date':'2026-11-10','smoke':smoke,'seed':42,'num_envs':16 if smoke else 512,
        'real_transition_budget':2048000,'steps_per_collection':32,'evaluation_every_updates':3 if smoke else 12,
        'evaluation_episodes':8 if smoke else 64,'evaluation_seed':81042,'test_seed':91042,
        'reference_reset':'Uniform reference-frame initialization for training; start at frame zero for validation/test, root XY rebased to origin.',
        'success':SUCCESS,'reward_weights':REWARD,
        'reward_scales':{'q_rad':.3,'dq_rad_s':5.,'root_pos_m':.1,'root_orientation_rad':.35,'root_vel_m_s':.08,'foot_pos_m':.02},
        'early_stop':'Two consecutive validation success rates >= 0.80; preserve checkpoint from first crossing. Otherwise stop at data budget.',
        'networks':{'world_hidden':[256,256,256],'motion_encoder':[256,128,32],'actor_decoder':[256,128],
                    'critic':[128,128],'observation_history':10,'future_reference_frames':10},
        'supermimic':{'offline_pretrain_updates':20 if smoke else 1000,'world_window':8,'batch':32 if smoke else 256,
                      'world_updates_per_collection':2 if smoke else 10,'imagination_horizon':32,'imagination_batch':32 if smoke else 256,
                      'learner_rounds_per_collection':1 if smoke else 4,
                      'critic_epochs':5,'critic_minibatches':4,'world_actor_lr':.0002,'actor_betas':[.9,.95],
                      'collection_action_noise_std':.1,'offline_supplied_transitions':summary['splits']['train']['transitions'],
                      'online_model_batches':'80% new on-policy replay, 20% original TRAIN data; no policy or world-model fitting on validation/test'},
        'ppo':{'lr':.0003,'clip':.2,'epochs':5,'minibatch_size':1024,'initial_std':.5,'entropy_coef':.005,
                'distribution':'tanh-Gaussian, exact transformed log probability; entropy regularization on underlying Gaussian'},
        'common':{'gamma':.99,'gae_lambda':.95,'full_domain_randomization':True,'action_scale':.58,'control_hz':50,
                  'actor_initialization':'Random, identical parameters before training; no teacher policy loaded.',
                  'data_budget_accounting':'SuperMimic cumulative=offline supplied TRAIN transitions + new real simulator transitions; PPO cumulative=new real transitions. Imagined transitions are separate.',
                  'time_accounting':'Training time includes offline data loading/pretraining, collection and optimization; excludes evaluation and simulator initialization. Program elapsed and evaluation time separately recorded.',
                  'reference_data_cost':'Reference generation and teacher training are shared prior costs, excluded from online training time; not a from-zero end-to-end compute comparison.'},
        'paper':'https://arxiv.org/abs/2610.28480','deviations':['Sesame/Isaac rather than G1/MuJoCo Warp; smaller networks/batches.',
            'Existing offline TRAIN data warm-starts dynamics; actor still starts randomly.',
            'Sesame-scaled analytic joint/root/foot tracking reward and soft analytic fall probabilities.',
            'Uniform reference resets; no return-based reset curriculum; no observation noise.',
            '512 real environments amortize simulation call overhead; four learner rounds per collection preserve the planned 128-environment update-to-data ratio.'],
        'dataset':str(dataset),'dataset_checksums_sha256':sha256(dataset/'checksums.json'),
        'reference_index_sha256':sha256(dataset/'reference_index.json'),
        'sources':{p:sha256(ROOT/p) for p in SOURCES},'asset_sha256':sha256(ROOT/'assets/isaac/sesame/sesame.usd')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=('supermimic','ppo'),required=True)
    parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/supermimic_sesame_20261110')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    study=args.study_dir.resolve(); dataset=args.dataset.resolve()
    if not study.is_relative_to(ROOT/'runs'): parser.error('Use a new directory under Sesame/runs')
    run=study/f'{args.algorithm}_seed42'
    if (run/'result.json').exists(): parser.error('Do not overwrite an existing run')
    run.mkdir(parents=True,exist_ok=True)
    require_runtime(); disable_git_imports()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_device(0); torch.set_num_threads(4)
    from sesame_rl.imitation_core import RobotMath,MotionLibrary,SUCCESS,DT,quat_matrix
    from sesame_rl.supermimic_small import Actor,Critic,WorldModel,OfflineReplay,OnlineReplay,fit_world,world_error,ppo_update,supermimic_update
    spec=protocol(study,dataset,args.smoke)
    if (study/'protocol.json').exists():
        if json.loads((study/'protocol.json').read_text())!=spec: raise RuntimeError('Frozen protocol/source mismatch')
    else: write_json(study/'protocol.json',spec)
    for relative in SOURCES:
        dest=run/'sources'/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,dest)
    runtime=ROOT/'.runtime/supermimic_training'
    for key,rel in {'XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config','XDG_DATA_HOME':'data','CUDA_CACHE_PATH':'cuda_cache'}.items():
        path=runtime/rel; path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
    started=time.monotonic(); app=env=None; monitor=MemoryMonitor(); exit_code=1
    result={'status':'running','algorithm':args.algorithm,'seed':42,'smoke':args.smoke,'pid':os.getpid(),
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(study/'protocol.json')}
    write_json(run/'result.json',result)
    training_seconds=evaluation_seconds=0.0; online_steps=0; imagined_steps=0; offline_steps=0
    try:
        from isaaclab.app import AppLauncher
        sys.argv=[sys.argv[0]]
        app=AppLauncher(headless=True,device='cuda:0',kit_args=' '.join(['--portable-root',str(runtime/'kit'),
            f'--/log/file={run/"isaac.log"}','--/app/settings/persistent=false','--/app/telemetry/enabled=false','--/renderer/multiGpu/enabled=false'])).app
        import warp as wp
        wp.config.kernel_cache_dir=str(runtime/'warp_cache')
        from sesame_rl.dr_environment import make_env_cfg
        from sesame_rl.imitation_environment import SesameImitationEnv
        mathlib=RobotMath('cuda:0')
        train_library=MotionLibrary(dataset,'train')
        val_library=train_library if args.smoke else MotionLibrary(dataset,'validation')
        cfg=make_env_cfg(num_envs=spec['num_envs'],seed=42,episode_length_s=9,domain_randomization=True)
        cfg.is_finite_horizon=True
        env=SesameImitationEnv(cfg,mathlib,train_library)
        env.reset(seed=42)
        write_json(run/'environment_config.json',cfg.to_dict())
        # Verify URDF-derived differentiable FK against imported Isaac link frames.
        from isaaclab.utils.math import quat_apply
        parents=[env.robot.body_names.index(n) for n in ('r3_link','r4_link','l3_link','l4_link')]
        body=env.robot.data.body_link_state_w[:,parents]
        offsets=mathlib.foot_offset[None].expand(env.num_envs,-1,-1)
        actual=body[...,:3]+quat_apply(body[...,3:7].reshape(-1,4),offsets.reshape(-1,3)).reshape(env.num_envs,4,3)-env.robot.data.root_pos_w[:,None]
        fk_error=float((actual-mathlib.feet_relative_world(env.state())).abs().max())
        obs_error=float((env._get_observations()['policy']-mathlib.observation(env.state())).abs().max())
        goal_error=float((mathlib.goal(env.state(),train_library.future(env.motion_ids,env.phase,env.offset))-train_library.goal(env.state(),env.motion_ids,env.phase,env.offset)).abs().max())
        if fk_error>1e-4 or obs_error>3e-5 or goal_error>3e-5: raise RuntimeError(f'Math mismatch FK={fk_error}, observation={obs_error}, goal={goal_error}')
        write_json(run/'math_validation.json',{'fk_max_abs_m':fk_error,'observation_max_abs':obs_error,'goal_max_abs':goal_error,'cuda':True})
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        with torch.device('cuda:0'):
            actor=Actor(); critic=Critic(); world=WorldModel(mathlib) if args.algorithm=='supermimic' else None
        initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in actor.parameters())).hexdigest()
        result['initial_actor_sha256']=initial_hash
        result['gpu']=torch.cuda.get_device_name(0)
        actor_opt=torch.optim.Adam(actor.parameters(),lr=.0002 if world else .0003,betas=(.9,.95) if world else (.9,.999))
        critic_opt=torch.optim.Adam(critic.parameters(),lr=.0002 if world else .0003)
        world_opt=torch.optim.Adam(world.parameters(),lr=.0002) if world else None
        replay=OnlineReplay(env.num_envs,'cuda:0') if world else None
        # JSONL is authoritative even if TensorBoard is unavailable.
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(run/'tensorboard'))
        except ImportError: writer=None
        def checkpoint(name,update):
            torch.save({'algorithm':args.algorithm,'actor':actor.state_dict(),'critic':critic.state_dict(),
                        'world':None if world is None else world.state_dict(),'protocol':spec,
                        'online_steps':online_steps,'offline_steps':offline_steps,'imagined_steps':imagined_steps,
                        'training_seconds':training_seconds,'update':update,'initial_actor_sha256':initial_hash,
                        'resume_supported':False},run/name)
        @torch.no_grad()
        def evaluate(library,seed,update,label='validation'):
            nonlocal evaluation_seconds
            begin=time.monotonic()
            env.set_library(library,evaluation=True); env.set_domain_randomization(True); env.reset(seed=seed)
            env.episode_records.clear()
            count=spec['evaluation_episodes']
            initial=torch.cat((env.state()[:count],env.context()[:count]),-1)
            pair=study/f'paired_{label}_seed{seed}.pt'
            if pair.exists():
                stored=torch.load(pair,map_location='cuda:0',weights_only=True)
                pairing_error=float((initial-stored).abs().max())
                if pairing_error>1e-6: raise RuntimeError(f'Unpaired evaluation reset {pairing_error}')
            else: torch.save(initial,pair); pairing_error=0.0
            history=mathlib.observation(env.state())[:,None].repeat(1,10,1)
            active=torch.ones(count,dtype=torch.bool,device='cuda:0')
            sums=torch.zeros((count,6),device='cuda:0'); returns=torch.zeros(count,device='cuda:0')
            lengths=torch.zeros(count,dtype=torch.long,device='cuda:0')
            fell=torch.zeros(count,dtype=torch.bool,device='cuda:0'); outside=fell.clone(); invalid=fell.clone()
            for _ in range(120 if args.smoke else library.frames-1):
                state=env.state(); goal=library.goal(state,env.motion_ids,env.phase,env.offset)
                action=actor.action(history,goal)
                _,reward,terminated,truncated,_=env.step(action)
                sums+=env.transition_metrics[:count]*active[:,None]
                returns+=reward[:count]*active; lengths+=active.long()
                fell|=env.transition_fell[:count]&active; outside|=env.transition_outside[:count]&active; invalid|=env.transition_invalid[:count]&active
                active&=~(terminated[:count]|truncated[:count])
                obs=mathlib.observation(env.state()); history=torch.cat((history[:,1:],obs[:,None]),1)
                done=terminated|truncated; history[done]=obs[done,None].expand(-1,10,-1)
            rms=(sums/lengths.clamp_min(1)[:,None]).sqrt()
            success=(lengths==450)&~fell&~outside&~invalid
            thresholds=[SUCCESS['velocity_rmse_m_s'],math.radians(SUCCESS['heading_rms_deg']),SUCCESS['joint_position_rmse_rad'],SUCCESS['foot_position_rmse_m'],SUCCESS['root_position_rmse_m']]
            success&=(rms[:,:5]<=torch.tensor(thresholds,device='cuda:0')).all(-1)
            seconds=time.monotonic()-begin; evaluation_seconds+=seconds
            row={'label':label,'update':update,'online_steps':online_steps,'offline_supplied_steps':offline_steps,
                 'cumulative_real_steps':online_steps+offline_steps,'imagined_steps':imagined_steps,
                 'training_seconds':training_seconds,'evaluation_seconds':seconds,'episodes':count,
                 'success_rate':float(success.float().mean()),'fall_rate':float(fell.float().mean()),
                 'mean_return':float(returns.mean()),'velocity_rmse_m_s':float(rms[:,0].mean()),
                 'heading_rms_deg':float(rms[:,1].mean()*180/math.pi),'joint_position_rmse_rad':float(rms[:,2].mean()),
                 'foot_position_rmse_m':float(rms[:,3].mean()),'root_position_rmse_m':float(rms[:,4].mean()),
                 'joint_acceleration_rms_rad_s2':float(rms[:,5].mean()),'initial_pairing_max_abs':pairing_error,
                 'episode_success':success.cpu().tolist(),'episode_metrics':rms.cpu().tolist(),
                 'episode_lengths':lengths.cpu().tolist(),'episode_returns':returns.cpu().tolist()}
            append(run/'evaluations.jsonl',row)
            if writer:
                for k in ('success_rate','fall_rate','mean_return','velocity_rmse_m_s','heading_rms_deg','joint_position_rmse_rad','foot_position_rmse_m'):
                    writer.add_scalar(label+'/'+k,row[k],row['cumulative_real_steps'])
            print('EVALUATION='+json.dumps({k:v for k,v in row.items() if not k.startswith('episode_')}),flush=True)
            return row
        def reset_training(update):
            env.set_library(train_library,evaluation=False); env.set_domain_randomization(True)
            env.reset(seed=42+100000+update)
            obs=mathlib.observation(env.state())
            return obs[:,None].repeat(1,10,1)
        initial_eval=evaluate(val_library,81042,0)
        checkpoint('initial.pt',0)
        best=initial_eval; stable=None; pending=None; consecutive=0
        offline=heldout=None
        if world:
            tick=time.monotonic()
            offline=OfflineReplay(dataset,'train','cuda:0',16 if args.smoke else None)
            heldout=OfflineReplay(dataset,'train' if args.smoke else 'validation','cuda:0',16 if args.smoke else None)
            offline_steps=offline.transitions
            for update in range(spec['supermimic']['offline_pretrain_updates']):
                loss=fit_world(world,world_opt,offline,spec['supermimic']['batch'])
                if (update+1)%100==0 or update==spec['supermimic']['offline_pretrain_updates']-1:
                    diagnostic=world_error(world,heldout,spec['supermimic']['batch'])
                    append(run/'world_pretraining.jsonl',{'update':update+1,'loss':loss,'validation':diagnostic,'elapsed_seconds':time.monotonic()-tick})
                    print('WORLD_PRETRAIN='+json.dumps({'update':update+1,'loss':loss,'validation':diagnostic}),flush=True)
            training_seconds+=time.monotonic()-tick
            result['offline_pretraining_seconds']=training_seconds
            # Policy is unchanged by world-model fitting; expose the real cost
            # as a second point, without spending another physics evaluation.
            warmed=dict(initial_eval,offline_supplied_steps=offline_steps,cumulative_real_steps=offline_steps,training_seconds=training_seconds,stage='world_pretrained_actor_unchanged')
            append(run/'evaluations.jsonl',warmed)
        history=reset_training(0)
        episodes=torch.zeros(env.num_envs,dtype=torch.long,device='cuda:0')
        chunk=spec['steps_per_collection']; per_update=chunk*env.num_envs
        updates=3 if args.smoke else (spec['real_transition_budget']-offline_steps)//per_update
        executed_updates=0
        for update in range(1,updates+1):
            tick=time.monotonic(); rollout=[]
            with torch.no_grad():
                for t in range(chunk):
                    state=env.state(); context=env.context()
                    goal=train_library.goal(state,env.motion_ids,env.phase,env.offset)
                    value=critic(mathlib.features(state),goal,mathlib.context_features(context))
                    if world:
                        action=(actor.action(history,goal)+.1*torch.randn((env.num_envs,8),device='cuda:0')).clamp(-1,1)
                    else: action,z,lp=actor.sample(history,goal)
                    old_history=history.clone()
                    _,reward,terminated,truncated,_=env.step(action)
                    done=terminated|truncated
                    if not bool(torch.isfinite(reward).all()&torch.isfinite(env.next_state).all()): raise RuntimeError('Nonfinite real transition')
                    nextgoal=train_library.goal(env.next_state,env.transition_motion_ids,env.next_phase,env.transition_offset)
                    nv=critic(mathlib.features(env.next_state),nextgoal,mathlib.context_features(context))
                    if world:
                        replay.add(state=state,next_state=env.next_state,action=action,context=context,history=old_history,
                                   motion=env.transition_motion_ids,phase=env.transition_phase,offset=env.transition_offset,
                                   episode=episodes,done=done)
                    else:
                        rollout.append({'state':state,'context':context,'history':old_history,'goal':goal,'z':z,'logprob':lp,
                                        'value':value,'next_value':nv,'reward':reward,'done':done})
                    episodes+=done.long()
                    obs=mathlib.observation(env.state()); history=torch.cat((history[:,1:],obs[:,None]),1)
                    history[done]=obs[done,None].expand(-1,10,-1)
                    online_steps+=env.num_envs
            collection_seconds=time.monotonic()-tick; tick=time.monotonic()
            if world:
                losses=[]
                for learner_round in range(spec['supermimic']['learner_rounds_per_collection']):
                    for j in range(spec['supermimic']['world_updates_per_collection']):
                        buffer=offline if j%5==0 else replay
                        losses.append(fit_world(world,world_opt,buffer,spec['supermimic']['batch']))
                    metrics=supermimic_update(actor,critic,world,actor_opt,critic_opt,replay,train_library,mathlib,
                                              spec['supermimic']['imagination_batch'],spec['supermimic']['imagination_horizon'])
                    imagined_steps+=metrics['imagined_transitions']
                metrics['world_loss']=sum(losses)/len(losses)
            else:
                tensor_rollout={k:torch.stack([r[k] for r in rollout]) for k in rollout[0]}
                metrics=ppo_update(actor,critic,actor_opt,critic_opt,mathlib,tensor_rollout)
            update_seconds=time.monotonic()-tick
            training_seconds+=collection_seconds+update_seconds
            row={'update':update,'online_steps':online_steps,'offline_supplied_steps':offline_steps,
                 'cumulative_real_steps':online_steps+offline_steps,'imagined_steps':imagined_steps,
                 'training_seconds':training_seconds,'collection_seconds':collection_seconds,'update_seconds':update_seconds,**metrics}
            append(run/'training.jsonl',row)
            if writer:
                for key,value in metrics.items(): writer.add_scalar('train/'+key,value,online_steps+offline_steps)
            executed_updates=update
            if update%10==0: print('TRAIN_UPDATE='+json.dumps(row),flush=True)
            if update%spec['evaluation_every_updates']==0 or update==updates:
                checkpoint('latest.pt',update)
                if world:
                    diagnostic=world_error(world,heldout,spec['supermimic']['batch'])
                    append(run/'world_validation.jsonl',{'update':update,'online_steps':online_steps,**diagnostic})
                ev=evaluate(val_library,81042,update)
                if (ev['success_rate'],ev['mean_return'])>(best['success_rate'],best['mean_return']):
                    best=ev; checkpoint('best.pt',update)
                if ev['success_rate']>=SUCCESS['validation_success_rate']:
                    consecutive+=1
                    if consecutive==1:
                        pending=ev; checkpoint('candidate_success.pt',update)
                    if consecutive>=SUCCESS['consecutive_evaluations']:
                        stable={'first_crossing':pending,'confirmation':ev}
                        shutil.copy2(run/'candidate_success.pt',run/'first_success.pt')
                        if not args.smoke: break
                else: consecutive=0; pending=None
                history=reset_training(update); episodes+=1
        checkpoint('final.pt',executed_updates)
        if not (run/'best.pt').exists(): shutil.copy2(run/'initial.pt',run/'best.pt')
        if not args.smoke:
            test_library=MotionLibrary(dataset,'test')
            selected=run/('first_success.pt' if stable else 'best.pt')
            saved=torch.load(selected,map_location='cuda:0',weights_only=False)
            actor.load_state_dict(saved['actor'])
            test=evaluate(test_library,91042,saved['update'],label='test')
            test['selected_checkpoint']=selected.name
            test['selected_checkpoint_online_steps']=saved['online_steps']
            test['selected_checkpoint_cumulative_steps']=saved['online_steps']+saved['offline_steps']
            write_json(run/'test_result.json',test)
        else: test=None
        result.update(status='completed',stop_reason='stable_validation_success' if stable else 'budget_exhausted',
                      stable_success=stable,best_validation=best,test=test,online_steps=online_steps,offline_supplied_steps=offline_steps,
                      cumulative_real_steps=online_steps+offline_steps,imagined_steps=imagined_steps,training_seconds=training_seconds,
                      evaluation_seconds=evaluation_seconds,updates=executed_updates,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      peak_process_gpu_mib=monitor.peak_process_mib,peak_total_gpu_mib=monitor.peak_total_mib,
                      wall_seconds_before_shutdown=time.monotonic()-started)
        if writer: writer.close()
        exit_code=0
    except (Exception,KeyboardInterrupt) as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),online_steps=online_steps,training_seconds=training_seconds)
        traceback.print_exc()
    finally:
        monitor.stop.set()
        result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        write_json(run/'result.json',result)
        print('TRAIN_RESULT='+json.dumps({k:v for k,v in result.items() if k not in ('stable_success','best_validation','test')},allow_nan=False),flush=True)
        if app is not None:
            timer=threading.Timer(15,lambda:os._exit(exit_code)); timer.daemon=True; timer.start()
            if env is not None: env.close()
            app.close(wait_for_replicator=False); timer.cancel()
    return exit_code


if __name__=='__main__': raise SystemExit(main())
