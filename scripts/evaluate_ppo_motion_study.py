#!/usr/bin/env python3
"""Common-reward evaluation for independent smoothness/speed PPO experiments."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from train_ppo import ROOT, disable_git_imports, write_json
from train_heading_ppo import require_runtime, sha256
from evaluate_heading_ppo import evaluate_case, canonical_hash, statistics_or_none


def resolve(study):
    entries=[]
    for name,speed,coefficient in [('baseline',.10,.08),('smooth',.10,.16),('fast',.20,.08)]:
        folder=study/'training'/f'{name}_seed42'
        result=json.loads((folder/'result.json').read_text())
        if result.get('status')!='completed' or result.get('environment_transitions_total')!=2048000:
            raise RuntimeError(f'Complete 2,048,000-step training required: {folder}')
        if result.get('command_forward_m_s')!=speed or result.get('action_rate_penalty_scale')!=coefficient:
            raise RuntimeError(f'Unexpected training treatment: {folder}')
        if result.get('train_seed')!=42 or result.get('reward_version')!='motion_study_heading_v1':
            raise RuntimeError(f'Unexpected seed/reward version: {folder}')
        if result.get('heading_penalty_scale')!=.5 or not result.get('domain_randomization'):
            raise RuntimeError(f'Heading reward and full domain randomization must match: {folder}')
        saved_cfg=json.loads((folder/'environment_config.json').read_text())
        if saved_cfg['command_forward_m_s']!=speed or saved_cfg['action_rate_penalty_scale']!=coefficient:
            raise RuntimeError('Saved environment treatment differs from result metadata')
        checkpoint=folder/'final.pt'
        if sha256(checkpoint)!=result['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        for source in result['sources']:
            if sha256(folder/'sources'/source['path'])!=source['sha256']:raise RuntimeError('Source snapshot changed')
            if source['path'].startswith(('sesame_rl/','assets/')) and sha256(ROOT/source['path'])!=source['sha256']:
                raise RuntimeError('Evaluation environment differs from training')
        entries.append(dict(id=f'{name}_seed42',variant=name,train_seed=42,checkpoint=str(checkpoint),
                            sha256=result['checkpoint_sha256'],run_dir=str(folder),
                            train_speed=speed,train_action_rate_coefficient=coefficient,
                            initial_network_sha256=result['initial_network_sha256'],
                            ppo_config=json.loads((folder/'ppo_config.json').read_text())))
    if len({e['initial_network_sha256'] for e in entries})!=1:
        raise RuntimeError('Initial policy parameters must be identical across treatment arms')
    if any(e['ppo_config']!=entries[0]['ppo_config'] for e in entries[1:]):
        raise RuntimeError('PPO configurations must match across treatments')
    return entries


def resolve_probe():
    """Reuse one existing actor strictly for evaluator validation, not training."""
    folder=ROOT/'runs/ppo_heading_20261105/training/heading_seed42'
    saved=json.loads((folder/'result.json').read_text())
    checkpoint=folder/'final.pt'
    if saved['status']!='completed' or sha256(checkpoint)!=saved['checkpoint_sha256']:
        raise RuntimeError('Existing probe policy is incomplete or changed')
    config=json.loads((folder/'ppo_config.json').read_text())
    return [dict(id=f'probe_{name}',variant=name,train_seed=42,checkpoint=str(checkpoint),
                 sha256=saved['checkpoint_sha256'],run_dir=str(folder),train_speed=.10,
                 train_action_rate_coefficient=.08,
                 probe_counterfactual_action_rate_coefficient=.16 if name=='smooth' else .08,
                 initial_network_sha256='not_checked_probe_reuses_existing_actor',ppo_config=config,
                 probe_note='All logical slots reuse the same old actor; coefficient changes only validate reporting.')
            for name in ['baseline','smooth','fast']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--probe',action='store_true',help='Short evaluator-only CUDA check with one existing policy; not an experiment')
    args=parser.parse_args()
    study=args.study_dir.resolve(); output=study/('evaluation_probe' if args.probe else 'evaluation')
    if not study.is_relative_to(ROOT/'runs'):raise ValueError('Study must be in Sesame runs')
    if (output/'result.json').exists():raise FileExistsError('Use a fresh evaluation directory')
    entries=resolve_probe() if args.probe else resolve(study)
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(num_envs=256,episodes_per_seed=64,seeds=[35042,36042],conditions=['nominal','randomized'],episode_length_s=10.)
    if args.probe:settings.update(num_envs=16,episodes_per_seed=4,seeds=[45042],episode_length_s=.4)
    protocol=dict(probe_only=args.probe,training_seed=42,entries=entries,evaluation=settings,
                  comparisons=[dict(speed=.10,models=['baseline','smooth']),dict(speed=.20,models=['baseline','fast'])],
                  evaluation_action_rate_coefficient=.08,
                  training_reward_return='Computed on exactly the same state/action rollout by replacing only its action-rate coefficient; not a second rollout.',
                  time_note='Trainings run sequentially. No existing training time substituted.',
                  score_note='Local research diagnostics, no custom success threshold or course score.')
    write_json(output/'protocol.json',protocol)
    protocol_hash=sha256(output/'protocol.json')
    code=['scripts/evaluate_ppo_motion_study.py','scripts/evaluate_heading_ppo.py','scripts/train_ppo_motion_study.py',
          'sesame_rl/environment.py','sesame_rl/dr_environment.py','sesame_rl/study_environment.py','sesame_rl/heading_environment.py']
    source_hashes={relative:sha256(ROOT/relative) for relative in code}
    for relative in code:
        target=output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    signature=canonical_hash(dict(protocol=protocol,source_hashes=source_hashes))
    write_json(output/'manifest.json',dict(protocol_sha256=protocol_hash,signature=signature,source_hashes=source_hashes))
    result=dict(status='initializing',probe_only=args.probe,pid=os.getpid(),cases=[],protocol_sha256=protocol_hash)
    write_json(output/'result.json',result)
    app=env=None;exit_code=1;started=time.monotonic()
    runtime=ROOT/'.runtime/ppo_motion_study_eval';runtime.mkdir(parents=True,exist_ok=True)
    for variable,subdir in {'XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config',
                            'XDG_DATA_HOME':'data','CUDA_CACHE_PATH':'cuda_cache'}.items():
        directory=runtime/subdir;directory.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(directory)
    sys.dont_write_bytecode=True
    try:
        require_runtime();disable_git_imports()
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU model/physics fallback forbidden')
        torch.cuda.set_device(0);torch.set_num_threads(4)
        from isaaclab.app import AppLauncher
        sys.argv=[sys.argv[0]]
        app=AppLauncher(headless=True,device='cuda:0',kit_args=f'--portable-root {runtime}/kit --/log/file={output}/isaac.log --/app/settings/persistent=false --/app/telemetry/enabled=false --/renderer/multiGpu/enabled=false').app
        import warp as wp
        wp.config.kernel_cache_dir=str(runtime/'warp_cache')
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.heading_environment import SesameHeadingEnv,make_heading_env_cfg

        class MotionEvaluationEnv(SesameHeadingEnv):
            def __init__(self,*a,**kw):
                self._motion_ready=False
                super().__init__(*a,**kw)
                self._motion_integrals=torch.zeros((self.num_envs,2),device=self.device)
                self._motion_ready=True

            def _get_rewards(self):
                reward=super()._get_rewards()
                if self._motion_ready:
                    self._motion_integrals[:,0].add_((self._actions-self._previous_actions).square().mean(dim=1)*self.step_dt)
                    self._motion_integrals[:,1].add_(self.robot.data.root_lin_vel_w[:,0]*self.step_dt)
                return reward

            def _reset_idx(self,env_ids):
                if not self._motion_ready:return super()._reset_idx(env_ids)
                ids=self._ids(env_ids)
                completed=ids[(self.episode_length_buf[ids]>0)&self.reset_buf[ids]]
                completed_ids=completed.tolist()
                if completed_ids:
                    expected=-self.cfg.action_rate_penalty_scale*self._motion_integrals[completed,0]
                    measured=self._reward_sums['action_rate'][completed]
                    if not torch.allclose(expected,measured,rtol=1e-4,atol=1e-6):
                        raise RuntimeError('New action integral differs from original environment reward accounting')
                integral_values=self._motion_integrals[completed].tolist()
                super()._reset_idx(ids)
                if completed_ids:
                    records=list(self.episode_records)[-len(completed_ids):]
                    for record,index,values in zip(records,completed_ids,integral_values,strict=True):
                        if record['env_id']!=index:raise RuntimeError('Terminal diagnostic ordering mismatch')
                        record['action_difference_squared_integral']=values[0]
                        record['sampled_mean_vx_m_s']=values[1]/record['duration_s']
                        record['normalized_action_rate_rms_per_s']=math.sqrt(values[0]/record['duration_s'])/self.step_dt
                self._motion_integrals[ids]=0.

        cfg=make_heading_env_cfg(num_envs=settings['num_envs'],seed=settings['seeds'][0],device='cuda:0',episode_length_s=settings['episode_length_s'],heading_penalty_scale=.5)
        cfg.action_rate_penalty_scale=.08
        env=MotionEvaluationEnv(cfg);wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        for comparison in protocol['comparisons']:
            speed=comparison['speed'];speed_dir=output/f'speed_{speed:.2f}';speed_dir.mkdir(exist_ok=True)
            # Isaac 4.5 keeps the USD stage after env.close(). Reuse the verified
            # scene and explicitly synchronize reward/diagnostic and observation
            # command sources before evaluate_case resets each seeded episode.
            cfg.command_forward_m_s=speed
            env.cfg.command_forward_m_s=speed
            env._commands_w.zero_()
            env._commands_w[:,0].fill_(speed)
            if not bool(torch.all(env._commands_w[:,0]==speed)):
                raise RuntimeError('Observation and reward command speeds differ')
            write_json(speed_dir/'environment_config.json',cfg.to_dict())
            eval_args=SimpleNamespace(output_dir=speed_dir,device='cuda:0')
            with torch.inference_mode():
                for entry in entries:
                    if entry['variant'] not in comparison['models']:continue
                    config=entry['ppo_config'];actor_config=copy.deepcopy(config['policy']);actor_config.pop('class_name')
                    with torch.device('cuda:0'):
                        actor=ActorCritic(wrapper.get_observations(),config['obs_groups'],8,**actor_config).to('cuda:0')
                    actor.load_state_dict(torch.load(entry['checkpoint'],map_location='cuda:0',weights_only=False)['model_state_dict']);actor.eval()
                    for condition in settings['conditions']:
                        case=evaluate_case(env,wrapper,actor,entry,condition,settings,eval_args,protocol_hash,signature,torch)
                        for record in case['episodes']:
                            record['return_under_training_action_rate_coefficient']=record['return']-(entry['train_action_rate_coefficient']-.08)*record['action_difference_squared_integral']
                            if args.probe:
                                coefficient=entry['probe_counterfactual_action_rate_coefficient']
                                record['probe_counterfactual_return']=record['return']-(coefficient-.08)*record['action_difference_squared_integral']
                        case['target_speed_m_s']=speed
                        case['common_evaluation_action_rate_coefficient']=.08
                        case['training_action_rate_coefficient']=entry['train_action_rate_coefficient']
                        case['statistics_note']='Episode variation for one trained policy; this study uses one training seed and does not estimate between-training-seed variability.'
                        case['metric_definitions']={**case['metric_definitions'],
                            'normalized_action_rate_rms_per_s':'sqrt(mean_time_and_8_actions((a_t-a_previous)^2))/0.02 s; normalized action per second, not a percentage',
                            'sampled_mean_vx_m_s':'Mean world-X velocity at 50 Hz control samples; diagnostic only; report uses net X displacement/duration as effective forward speed',
                            'action_difference_squared_integral':'sum_t mean_8_actions((a_t-a_previous)^2)*0.02 s; directly checked against original action-rate reward sum'}
                        case['additional_metrics']={key:statistics_or_none([r[key] for r in case['episodes']],len(case['episodes'])) for key in ['return_under_training_action_rate_coefficient','sampled_mean_vx_m_s','normalized_action_rate_rms_per_s']}
                        case['true_termination_rate']=statistics.mean(float(r['fell'] or r['out_of_bounds'] or r['invalid_state']) for r in case['episodes'])
                        path=speed_dir/f'{entry["id"]}_{condition}.json';write_json(path,case)
                        result['cases'].append(str(path));result['status']='evaluating';write_json(output/'result.json',result)
                        print('MOTION_CASE '+json.dumps(dict(model=entry['id'],speed=speed,condition=condition,return_mean=case['metrics']['return']['mean'])),flush=True)
                    del actor
        result.update(status='completed',wall_seconds=time.monotonic()-started,
                      episodes=len(result['cases'])*len(settings['seeds'])*settings['episodes_per_seed'],
                      cases_completed=len(result['cases']),action_rate_reward_accounting_verified=True)
        exit_code=0
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        write_json(output/'result.json',result)
        if app is not None:
            def finish_timeout():os._exit(exit_code)
            timer=threading.Timer(15.,finish_timeout);timer.daemon=True;timer.start()
            if env is not None:env.close()
            app.close(wait_for_replicator=False);timer.cancel()
    return exit_code


if __name__=='__main__':raise SystemExit(main())
