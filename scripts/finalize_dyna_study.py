#!/usr/bin/env python3
"""Verify saved metadata and index completed Dyna results; no model computation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'runs/dyna_ppo_20261103'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    protocol = read(STUDY/'protocol.json')
    assert sha(STUDY/'protocol.json') == (STUDY/'protocol.sha256').read_text().split()[0]
    for relative, expected in protocol['frozen_sources'].items():
        assert sha(ROOT/relative) == expected, relative
    source_checks, training, pre_imagination = {}, [], []
    entries = {entry['id']:entry for entry in protocol['checkpoints']}
    for entry in entries.values():
        run = Path(entry['run_dir'])
        result = read(run/'result.json')
        assert result['status'] == 'completed'
        assert result['environment_transitions_total'] == 2048000
        assert result['training_iterations_this_run'] == 250
        assert result['reload_max_abs_action_error'] == 0
        assert sha(run/'final.pt') == result['checkpoint_sha256']
        rows = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
        assert len(rows) == 250
        assert all(r['iteration'] == i+1 and r['environment_transitions'] == 8192*(i+1)
                   and all(math.isfinite(v) for v in r['losses'].values()) for i,r in enumerate(rows))
        for relative in ('result.json','training.jsonl','final.pt','ppo_config.json'):
            source_checks[str(run/relative)] = sha(run/relative)
        record = {'id':entry['id'], 'real_steps':2048000, 'checkpoint_sha256':result['checkpoint_sha256'],
                  'reused_training':entry['reused_training'], 'actor_reload_error':0}
        if entry['variant'] == 'dyna_ppo':
            assert result['world_model_reload_max_abs_prediction_error'] == 0
            summary = result['dyna_training_summary']
            assert summary['imagined_optimizer_steps'] > 0
            assert summary['real_transitions_seen'] == 2048000
            assert sum(r['losses']['imagined_optimizer_steps'] for r in rows) == summary['imagined_optimizer_steps']
            assert sum(r['losses']['imagined_accepted'] for r in rows) == summary['imagined_accepted']
            assert all(r['losses']['imagined_optimizer_steps'] == 0 for r in rows[:40])
            assert all(r['losses']['imagined_normalization_unchanged'] == 1 for r in rows)
            assert sha(run/'dyna_config.json') == protocol['training']['dyna_config_sha256']
            for source in result['sources']:
                assert sha(run/'sources'/source['path']) == source['sha256']
            record.update(dyna=summary,world_model_reload_error=0)
            reference = Path(entries[f'full_dr_seed{entry["train_seed"]}']['run_dir'])
            control = [json.loads(line) for line in (reference/'training.jsonl').read_text().splitlines()]
            differences = {key:max(abs(rows[i]['losses'][key]-control[i]['losses'][key]) for i in range(40))
                           for key in ('value_function','surrogate','entropy')}
            valid = [i for i in range(40) if rows[i]['mean_episode_return'] is not None and control[i]['mean_episode_return'] is not None]
            differences['logged_episode_return'] = max(abs(rows[i]['mean_episode_return']-control[i]['mean_episode_return']) for i in valid)
            pre_imagination.append({'seed':entry['train_seed'],'updates':40,'maximum_absolute_log_differences':differences})
        training.append(record)
    evaluation = read(STUDY/'evaluation/result.json')
    assert evaluation['status']=='completed' and evaluation['cases_completed']==18
    assert evaluation['pairing_verified_all_cases']
    analysis = read(STUDY/'analysis/summary.json')
    assert analysis['status']=='completed' and analysis['total_evaluation_episodes']==2304
    for source,expected in analysis['inputs_sha256'].items():
        assert sha(Path(source))==expected, source
    video = read(STUDY/'videos/comparison_validation.json')
    assert video['status']=='completed' and video['exact_pairing_passed']
    assert video['visual_inspection']=='passed'
    for pathkey,hashkey in [('comparison_video','comparison_sha256'),('thumbnail','thumbnail_sha256')]:
        assert sha(Path(video[pathkey]))==video[hashkey]
    # Preserve the previous experiment, including its published figures/report.
    old = read(ROOT/'runs/robustness_ablation_20261103/result.json')
    for relative,meta in old['artifacts'].items():
        assert sha(ROOT/relative)==meta['sha256'], relative
    outcome = {'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'protocol_sha256':sha(STUDY/'protocol.json'),'training':training,
               'pre_imagination_control_comparison':pre_imagination,
               'evaluation':{'cases':18,'episodes':2304,'exact_initial_state_and_domain_pairing':True},
               'primary_results':{c:{v:metrics['velocity_x_rmse_m_s'] for v,metrics in variants.items()}
                                  for c,variants in analysis['groups'].items()},
               'source_checks_sha256':source_checks,'previous_study_preserved':True,
               'scope':'Local Dyna-style PPO, 3 seeds, extra model/imagination compute, fixed previously observed evaluation scenarios; no hardware or official-score claim'}
    include = [STUDY/'protocol.json',STUDY/'dyna_config.json',STUDY/'analysis/report.md',
               STUDY/'analysis/summary.json',STUDY/'training_audit.json',STUDY/'evaluation_audit.json',STUDY/'videos/comparison_validation.json',
               STUDY/'videos/ppo_study_comparison_seed3042.mp4', ROOT/'docs/DYNA_PPO_STUDY.md']
    include += list((STUDY/'analysis').glob('*.png')) + list((STUDY/'analysis').glob('*.pdf')) + list((STUDY/'analysis').glob('*.csv'))
    outcome['artifacts']={str(path.relative_to(ROOT)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in include}
    (STUDY/'result.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'completed','result':str(STUDY/'result.json'),'training_runs':9,'new_dyna_runs':3,'evaluation_episodes':2304},ensure_ascii=False))


if __name__=='__main__':
    main()
