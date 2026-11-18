"""Reapply user-approved admission rules to saved metrics, preserving original runs.

No new trajectories, model inference, API calls, or physics rollouts are run.
Original program/plan/reference/trace hashes are verified before reassessment.
"""
import argparse
from collections import Counter
import copy
import csv
import json
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/supermimic_official_5efbf114/src')]
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.runtime/research_cache'))


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/trajectory_agent_unclipped.json')
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    if out.exists():p.error('Output exists; use a new directory')
    from sesame_trajectory_agent.schema import validate_config
    from sesame_trajectory_agent.decisions import decide
    from sesame_trajectory_agent.trajectories import sha256,write_json,export_accepted
    config=validate_config(read(a.config));old=read(source/'protocol.json')
    if read(source/'status.json')['status']!='complete':raise ValueError('Original collection is incomplete')
    removed_gates={'foot_tracking_rmse_max_m','joint_limit_violation_max_rad'}
    def clean(c):
        result={k:v for k,v in c.items() if k!='diagnostic_only_metrics'}
        result['gates']={k:v for k,v in c['gates'].items() if k not in removed_gates}
        return result
    if clean(config)!=clean(old['config']):raise ValueError('Only user-removed gates and diagnostic flags may change')
    archived=source/'sources/sesame_trajectory_agent/decisions.py'
    if not archived.exists() or sha256(archived)!=old['source_sha256']['sesame_trajectory_agent/decisions.py']:
        raise ValueError('Verified original decision source is required')
    spec=importlib.util.spec_from_file_location('archived_admission_rules',archived)
    original_rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_rules)
    expected_steps=round(config['duration_s']*config['fps'])
    before_hash=sha256(source/'protocol.json');verified=0;original_reports={}
    paths=sorted(source.glob('candidates/*/attempt_*/report.json'))
    # Validate every source artifact before creating an output or admitting data.
    for path in paths:
        r=read(path);original_reports[str(path.relative_to(source))]=sha256(path)
        if r['protocol_sha256']!=before_hash:raise ValueError('Original report protocol mismatch')
        for field,file in [('motion_program_sha256',path.parent/'motion.py'),
            ('candidate_reference_sha256',path.parent/'candidate_reference.npz'),
            ('plan_sha256',path.parent.parent/'plan.json'),
            ('nominal_trace_sha256',path.parent/'nominal_execution.npz'),
            ('randomized_trace_sha256',path.parent/'randomized_execution.npz')]:
            if field in r:
                if sha256(file)!=r[field]:raise ValueError('Changed source artifact: '+str(file))
                verified+=1
        if 'nominal' in r:
            if r.get('validation_complete') is not True or len(r.get('randomized',[]))!=config['robust_trials']:
                raise ValueError('Incomplete original validation')
            for trial in [r['nominal'],*r['randomized']]:
                original_reasons=original_rules.decide(trial['metrics'],old['config']['gates'],old['config'].get('diagnostic_only_metrics',[]))
                if original_reasons!=trial['reasons'] or trial['passed']!=(not original_reasons):
                    raise ValueError('Original decision does not match its own protocol')
                if trial['metrics']['steps']!=expected_steps:raise ValueError('Incomplete original rollout')
    out.mkdir(parents=True);started=time.monotonic()
    shutil.copytree(source/'candidates',out/'candidates')
    for name in ['context.json','simulation_context.json']:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    new_sources={}
    for file in [Path(__file__),*list((ROOT/'sesame_trajectory_agent').glob('*.py'))]:
        rel=file.relative_to(ROOT);dest=out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest);new_sources[str(rel)]=sha256(file)
    protocol=copy.deepcopy(old);protocol.update(config=config,source_sha256=new_sources,
        execution_source_sha256=old['source_sha256'],reassessment={
            'source_run':str(source),'source_protocol_sha256':before_hash,
            'source_reports_sha256':original_reports,'verified_source_artifact_count':verified,
            'user_instruction':'移除足端跟踪误差和关节限位检查；原两项速度指标仍仅作诊断',
            'new_physics_trials':0,'metrics_recomputed':False,'api_calls':0,
            'change':'User-removed metrics/gates are absent; other checks are retained'})
    write_json(out/'protocol.json',protocol);ph=sha256(out/'protocol.json')
    accepted=[];reports=[];nominal_failures=Counter();random_failures=Counter();seen=set()
    for old_path in paths:
        path=out/old_path.relative_to(source);r=read(old_path)
        r['previous_decision']={'source_report':str(old_path),'sha256':sha256(old_path),
            'accepted':r['accepted'],'reasons':r['reasons'],'protocol_sha256':before_hash}
        r['protocol_sha256']=ph;r['diagnostic_only_metrics']=config.get('diagnostic_only_metrics',[])
        if 'nominal' in r:
            for t in [r['nominal'],*r['randomized']]:
                for key in ('foot_tracking_rmse_m','joint_limit_violation_max_rad'):t['metrics'].pop(key,None)
                t['reasons']=decide(t['metrics'],config['gates'],config.get('diagnostic_only_metrics',[]))
                t['passed']=not t['reasons']
            reasons=r['nominal']['reasons']+[f"randomized_{t['trial']}:{x}" for t in r['randomized'] for x in t['reasons']]
            passed=r['kinematics']['passed'] and not reasons
            if passed and r['trajectory_sha256'] in seen:passed=False;reasons.append('duplicate_accepted_reference')
            if passed:seen.add(r['trajectory_sha256'])
            r.update(accepted=passed,stage='accepted' if passed else 'execution_rejected',reasons=reasons)
            nominal_failures.update(r['nominal']['reasons'])
            random_failures.update(x for t in r['randomized'] for x in t['reasons'])
            r['nominal_metrics']=r['nominal']['metrics']
        r.get('kinematics',{}).get('metrics',{}).pop('joint_limit_violation_max_rad',None)
        write_json(path,r);reports.append(r)
        if r['accepted']:
            accepted.append({'id':r['id'],'source_clip':'agent_motion_program','report':str(path),
                'trace':str(path.parent/'nominal_execution.npz')})
    context=read(out/'context.json')['robot']
    library=SimpleNamespace(joint_names=context['joint_names'],body_names=context['physical_constraints']['body_names'])
    export_accepted(out/'accepted',accepted,library,out/'protocol.json')
    executed=[r for r in reports if 'nominal' in r]
    summary={'status':'complete','source_run':str(source),'attempts':len(reports),
        'accepted_count':len(accepted),'nominal_pass_count':sum(r['nominal']['passed'] for r in executed),
        'unique_generated_references':len({r['trajectory_sha256'] for r in reports if 'trajectory_sha256' in r}),
        'new_physics_trials':0,'reassessed_existing_trials':sum(1+len(r['randomized']) for r in executed),
        'api_calls':0,'elapsed_s':time.monotonic()-started,'diagnostic_only_metrics':config.get('diagnostic_only_metrics',[]),
        'remaining_nominal_failures':dict(nominal_failures),'remaining_randomized_failures':dict(random_failures),
        'note':'Reassessment of saved measurements under current user-selected gates, not a new experiment; no speed measurement correction'}
    write_json(out/'summary.json',summary);write_json(out/'status.json',summary)
    with (out/'remaining_reasons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['id','nominal_passed','accepted','nominal_reasons','all_reasons']);w.writeheader()
        for r in reports:w.writerow({'id':r['id'],'nominal_passed':r.get('nominal',{}).get('passed'),
            'accepted':r['accepted'],'nominal_reasons':'; '.join(r.get('nominal',{}).get('reasons',[])),
            'all_reasons':'; '.join(r['reasons'])})
    for rel,h in original_reports.items():
        if sha256(source/rel)!=h:raise RuntimeError('Original report changed during reassessment')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
