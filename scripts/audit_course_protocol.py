"""CPU file/metadata audit of the current course comparison; no model loading."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MAIN = ('ppo','supermimic','supermimic_deepseek','supermimic_codex')
EXPECTED = MAIN+('supermimic_h16','supermimic_h64','supermimic_no_critic')


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): value.update(part)
    return value.hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,str): return re.sub(r'0x[0-9a-fA-F]+','0xADDRESS',value)
    return value
def jsonl(path):
    lines=Path(path).read_text().splitlines(); result=[]; partial=False
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
            partial=True
    return result,partial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,default=ROOT/'runs/course_comparison_20261115')
    p.add_argument('--output',type=Path,default=ROOT/'reports/course_draft_20261115/protocol_audit.json')
    a=p.parse_args(); suite=a.suite.resolve()
    sp=read(suite/'protocol.json'); dataset=Path(sp['dataset']); dp=read(dataset/'protocol.json')
    out=dict(audit_utc=datetime.now(timezone.utc).isoformat(), scope='CPU file/configuration/hash/log checks only; no model, physics, array arithmetic or GPU calls',
             suite=str(suite), suite_status=read(suite/'status.json'),
             suite_protocol_sha256=sha(suite/'protocol.json'), checks=[], runs={})
    checks=out['checks']
    def check(name,passed,evidence): checks.append(dict(name=name,status='pass' if passed else 'fail',evidence=evidence))
    check('seven_planned_groups_preserved',tuple(r['name'] for r in sp['runs'])==EXPECTED,[r['name'] for r in sp['runs']])
    check('single_seed_disclosed',sp['seed_count']==1,{'seed_count':sp['seed_count'],'seed':42})
    check('dataset_protocol_frozen',sha(dataset/'protocol.json')==sp['dataset_protocol_sha256'],sp['dataset_protocol_sha256'])
    changed=[name for name,digest in dp['array_sha256'].items() if not (dataset/name).is_file() or sha(dataset/name)!=digest]
    check('frozen_dataset_files_match_hashes',not changed,{'files_checked':len(dp['array_sha256']),'changed':changed})
    mapping=dp['mapping']
    training={e['content_sha256'] for name,entries in mapping.items() if name.endswith('_train') for e in entries}
    validation={e['content_sha256'] for e in mapping['common_validation']}
    testing={e['content_sha256'] for e in mapping['common_test']}
    overlaps={'train_validation':sorted(training&validation),'train_test':sorted(training&testing),'validation_test':sorted(validation&testing)}
    check('recorded_clip_content_hashes_disjoint',not any(overlaps.values()) and (len(training),len(validation),len(testing))==(72,12,12),
          {'counts':[len(training),len(validation),len(testing)],'overlaps':overlaps,
           'verification_scope':'Canonical per-clip content hashes were computed by the CUDA preparation script; this audit verifies recorded disjointness and the frozen output file hashes without recomputing arrays.'})
    counts={name:len(rows) for name,rows in mapping.items()}
    check('equal_train_data_and_balanced_common_eval',all(counts[k+'_train']==24 for k in ('original','codex','deepseek')) and
          all(Counter(e['source'] for e in mapping[k])=={'original':4,'codex':4,'deepseek':4} for k in ('common_validation','common_test')),
          {'counts':counts,'train_seconds':144,'eval_seconds_per_clip':6,'source_order':dp['common_eval_order']})
    check('agent_reference_exact_hashes_disjoint',not(set(dp['reference_sha256']['codex'].values())&set(dp['reference_sha256']['deepseek'].values())),
          {'codex_selected':len(dp['reference_sha256']['codex']),'deepseek_selected':len(dp['reference_sha256']['deepseek'])})
    acceptance=[]
    for source,record in dp['collection_sources'].items():
        run=Path(record['run']); protocol=read(run/'protocol.json')
        check(source+'_collection_provenance_frozen',sha(run/'protocol.json')==record['protocol_sha256'] and sha(run/'accepted/manifest.json')==record['manifest_sha256'],record)
        acceptance.append({k:protocol['config'][k] for k in dp['shared_agent_acceptance']})
    check('agent_acceptance_and_feedback_equal',all(v==dp['shared_agent_acceptance'] for v in acceptance),dp['shared_agent_acceptance'])
    out['positive_example_current_split']=[name for name,rows in mapping.items() if any(e['source']=='codex' and e['name']=='candidate_001_attempt_00' for e in rows)]
    out['upstream_sources']=[]
    for name in ('supermimic_official_5efbf114','mjlab_research_dd51a9fc'):
        provenance=read(ROOT/'third_party'/f'{name}.provenance.json')
        changes=[relative for relative,digest in provenance['files_sha256'].items() if sha(ROOT/'third_party'/name/relative)!=digest]
        item=dict(name=name,commit=provenance['commit'],files_checked=len(provenance['files_sha256']),changed=changes)
        out['upstream_sources'].append(item); check(name+'_unmodified',not changes,item)

    main_settings,env_hashes,init_hashes,source_groups={}, {}, {}, {}
    interval_rows=[]
    for name in EXPECTED:
        run=suite/name
        if not (run/'protocol.json').is_file():
            out['runs'][name]={'status':'not_started','final_test':None};continue
        saved=read(run/'protocol.json'); status=read(run/'status.json'); args=saved['args']
        snapshots=[relative for relative,digest in saved['sources'].items() if not(run/'sources'/relative).exists() or sha(run/'sources'/relative)!=digest]
        check(name+'_source_snapshot_integrity',not snapshots,{'files_checked':len(saved['sources']),'changed':snapshots})
        source_groups[name]=saved['sources']
        check(name+'_single_seed_record',args['seed']==42 and saved['seed_count']==1,{'seed':args['seed'],'seed_count':saved['seed_count']})
        intended=dict(envs=512,steps=2097152,width=512,batch=2048,eval_every=262144,eval_steps=300,sampling_mode='uniform',physics_dt=.0005,replay_ratio=16.)
        check(name+'_fixed_common_budget_settings',all(args[k]==v for k,v in intended.items()),{k:args[k] for k in intended})
        check(name+'_common_validation_test',Path(args['validation_file'])==dataset/'common_validation' and Path(args['test_file'])==dataset/'common_test',
              {'validation':args['validation_file'],'test':args['test_file']})
        env=clean(read(run/'environment_config.json'))
        env['commands']['motion']['motion_file']='INTENDED_TRAIN_SOURCE_DIFFERENCE'
        env_hashes[name]=canonical(env)
        init=read(run/'initial_actor.json') if(run/'initial_actor.json').is_file() else None
        if init: init_hashes[name]=init['parameter_sha256']
        curve,partial=jsonl(run/'curve.jsonl') if(run/'curve.jsonl').exists() else ([],False)
        expected_source='deepseek' if name=='supermimic_deepseek' else 'codex' if name=='supermimic_codex' else 'original'
        check(name+'_intended_training_reference',Path(args['motion_file'])==dataset/(expected_source+'_train'),args['motion_file'])
        reward=env['rewards']
        selected_bodies={body for key in ('motion_body_pos','motion_body_ori','motion_body_lin_vel','motion_body_ang_vel') for body in reward[key]['params']['body_names']}
        check(name+'_current_reward_terms',saved['reward_version']=='sesame_nonfoot_no_joint_limit_20261115' and 'joint_limit' not in reward and
              not(selected_bodies&{'r3_foot','r4_foot','l3_foot','l4_foot'}),{'version':saved['reward_version'],'terms':list(reward),'tracking_bodies':sorted(selected_bodies)})
        check(name+'_finite_monotone_curve_log',all(math.isfinite(r['return']) and math.isfinite(r['training_s']) for r in curve) and
              all(x['real_steps']<y['real_steps'] and x['training_s']<=y['training_s'] for x,y in zip(curve,curve[1:])),
              {'points':len(curve),'incomplete_last_line_while_running':partial})
        row=dict(status=status['status'],method=args['method'],source=expected_source,real_steps=status.get('real_steps'),
                 intended_real_steps=args['steps'],training_s=status.get('training_s'),end_to_end_s=status.get('end_to_end_s'),
                 initial_actor=init,initial_validation={k:curve[0][k] for k in ('real_steps','return','normalized_return','heading_rms_deg')} if curve else None,
                 validation_curve_points=len(curve),latest_validation=({k:curve[-1][k] for k in ('real_steps','return','normalized_return')} if curve else None),
                 final_test={k:status.get('final_test',{}).get(k) for k in ('return','normalized_return','heading_rms_deg','velocity_rmse_m_s','fall_rate')} if 'final_test'in status else None,
                 final_and_best_tests_recorded=('final_test'in status and 'best_validation_checkpoint_test'in status),
                 horizon=args['horizon'],no_critic=args['no_critic'],policy_gamma=saved['runner_config']['policy']['gamma'],
                 protocol_sha256=sha(run/'protocol.json'),source_snapshot_changed=snapshots)
        if status['status']=='complete':
            row['final_checkpoint_sha256']=sha(run/'final.pt')
            check(name+'_full_budget_and_unselected_final_test',status['real_steps']==args['steps'] and row['final_and_best_tests_recorded'],
                  {'real_steps':status['real_steps'],'final_and_best_recorded':row['final_and_best_tests_recorded'],'table_selection':'final.pt as declared by suite protocol'})
        out['runs'][name]=row
        interval_rows.append({'name':name,'start':status['started_unix'],'finish':status.get('finished_unix')})
        if name in MAIN:
            main_settings[name]={k:v for k,v in args.items() if k not in ('method','output','motion_file')}

    check('main_shared_run_arguments',len(main_settings)==4 and all(v==next(iter(main_settings.values())) for v in main_settings.values()),main_settings)
    check('main_environment_configs_equal_except_reference_path',len({env_hashes.get(k) for k in MAIN})==1 and all(k in env_hashes for k in MAIN),env_hashes)
    check('main_learned_actor_initial_parameters_match',len({init_hashes.get(k) for k in MAIN})==1 and all(k in init_hashes for k in MAIN),
          {'hashes':init_hashes,'scope':'Training-produced named-parameter hash records; this CPU audit does not instantiate networks or reload tensors. Running-statistic buffers and critics are not included.'})
    critical=('sesame_research/reward.py','sesame_research/task.py','sesame_research/evaluation.py','sesame_research/robot.py','sesame_research/runner.py','sesame_research/ppo.py','scripts/train_research_sesame.py')
    check('started_runs_share_training_and_reward_sources',all(len({v[k] for v in source_groups.values()})==1 for k in critical),
          {k:sorted({v[k] for v in source_groups.values()}) for k in critical})
    serial=all(left['finish'] is not None and left['finish']<=right['start'] for left,right in zip(interval_rows,interval_rows[1:]))
    check('run_intervals_are_serial',serial,interval_rows)
    for name,reference_horizon,no_critic in (('supermimic_h16',16,False),('supermimic_h64',64,False),('supermimic_no_critic',32,True)):
        row=out['runs'][name]
        if row['status']!='not_started':
            check(name+'_intended_ablation_only',row['source']=='original' and row['horizon']==reference_horizon and row['no_critic']==no_critic and row['policy_gamma']==.99,
                  {'source':row['source'],'horizon':row['horizon'],'no_critic':row['no_critic'],'gamma':row['policy_gamma']})
    out['timing_boundary']=dict(included='Sampling, optimizer updates, CUDA synchronization; fresh per-run Inductor/Triton compilation when first updates execute',
        excluded='Simulator/setup, validation/final-test evaluation, checkpoints/log serialization, reference acquisition and prior teacher training',
        end_to_end_s='Separately recorded training-process time including setup/evaluation; reference acquisition still separate',
        cache_root=sp['cache_root'],cache_directories={name:{'inductor':(Path(sp['cache_root'])/name/'inductor').exists(),'triton':(Path(sp['cache_root'])/name/'triton').exists()} for name in EXPECTED},
        evidence='run_course_comparison.py assigns distinct per-run TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR; train_research_sesame.py places the timer after evaluation and synchronizes before charging elapsed time',
        gpu_before_snapshots={name:(suite/(name+'_gpu_before.csv')).read_text().strip().splitlines() for name in EXPECTED if(suite/(name+'_gpu_before.csv')).exists()},
        caveat='Empty startup compute-process lists do not prove an exclusively reserved GPU throughout training. Physical Warp kernels and other shared caches are not all cold.' )
    out['interpretation_limits']=[
        'One training seed only. Neither twelve held-out trajectories nor seven configurations measure cross-seed stability or statistical significance.',
        'Actor learned-parameter hash equality is verified from logs, but zero-step validation returns differ; preserve original curves and do not claim bitwise-identical complete execution.',
        'PPO and SuperMimic share task, actor architecture, initialization records and real-interaction budget; they retain algorithm-specific optimizers and SuperMimic additionally uses imagined steps.',
        'Agent variants change training-reference distribution. They are data-generation comparisons on a common evaluation mixture, not isolated RL-algorithm changes.',
        'Frozen clip hashes exclude exact duplicate leakage; references remain related gait variants and historical development informed generation. This is not unseen-skill generalization.',
        'The official upstream sources are unchanged; Sesame task, current local reward, PPO adapter and evaluation are local adaptations, not a full paper-scale reproduction.',
        'Return and tracking RMSE alone do not establish actual forward progress. The separate final.pt motion diagnosis must report net speed, travel direction and completion.',
        'Training-time curves exclude data generation/teacher training. Do not claim end-to-end speedup without accounting for those costs.',
        'Log success_rate is a local configured threshold pass fraction, not course success or the paper benchmark.',
    ]
    out['reporting_policy']=dict(primary_order=list(MAIN),primary_test_checkpoint='final.pt',curves='Every logged common-validation point; no smoothing/seed bands or selected favorable endpoints',
        negative_results='All planned groups, timing regressions, lower-return ablations and completed-but-not-forward cases must remain visible; this audit records current artifacts, not a guarantee about future report prose',
        pending=[k for k,v in out['runs'].items() if v['status']!='complete'],
        motion_diagnosis_pending=not (ROOT/'runs/course_motion_evaluation_20261115/status.json').is_file())
    out['failed_checks']=[c['name'] for c in checks if c['status']=='fail']
    out['audit_status']='issues_found' if out['failed_checks'] else 'pass_for_started_runs_pending_completion' if out['reporting_policy']['pending'] else 'pass_for_all_seven_runs'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    details=a.output.with_name(a.output.stem+'_details.json')
    details.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    short={k:out[k] for k in ('audit_utc','audit_status','scope','suite','suite_status','suite_protocol_sha256','upstream_sources','runs','positive_example_current_split','timing_boundary','interpretation_limits','reporting_policy','failed_checks')}
    short['checks_summary']={'total':len(checks),'passed':sum(c['status']=='pass' for c in checks),'failed':out['failed_checks'],'detailed_evidence':str(details)}
    short['dataset_evidence']={'directory':str(dataset),'frozen_file_count':len(dp['array_sha256']),
        'changed_files':changed,'unique_train_validation_test_clip_counts':[len(training),len(validation),len(testing)],
        'cross_split_recorded_content_hash_overlaps':overlaps,'split_counts':counts,'common_validation_and_test_each':'4 original + 4 DeepSeek + 4 Codex, whole trajectories',
        'training_seconds_of_reference_per_group':144,'agent_acceptance_settings_equal':all(v==dp['shared_agent_acceptance'] for v in acceptance)}
    short['shared_main_settings']=next(iter(main_settings.values())) if main_settings else None
    short['initial_actor_evidence']={'parameter_hashes':init_hashes,'parameter_count':793032,
        'scope':'Learned parameter records only, not all normalization buffers or critic parameters. Zero-step evaluation returns differ; those original points are preserved.'}
    short['environment_semantic_hashes_except_training_reference_path']=env_hashes
    short['local_snapshot_file_counts']={name:len(v) for name,v in source_groups.items()}
    a.output.write_text(json.dumps(short,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'audit_status':out['audit_status'],'checks':len(checks),'failed_checks':out['failed_checks'],'pending':out['reporting_policy']['pending'],'output':str(a.output)},ensure_ascii=False))


if __name__=='__main__': main()
