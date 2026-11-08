#!/usr/bin/env python3
"""Analyze stored study metadata only; no model/simulator imports or computation."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['baseline', 'full_dr', 'fixed_delay']
LABELS = {'baseline': 'Fixed PPO', 'full_dr': 'Full DR', 'fixed_delay': 'Fixed-delay DR'}
COLORS = {'baseline': '#2563EB', 'full_dr': '#D97706', 'fixed_delay': '#059669'}
METRICS = {
    'velocity_x_rmse_m_s': 'Forward speed tracking RMSE (m/s)',
    'velocity_xy_tracking_rmse_m_s': 'Planar velocity tracking RMSE (m/s)',
    'distance_x_m': 'Forward displacement (m)',
    'abs_final_heading_yaw_deg': 'Absolute final heading (degrees)',
    'abs_lateral_displacement_m': 'Absolute final lateral offset (m)',
    'average_speed_abs_error_m_s': 'Episode-mean speed absolute error (m/s)',
    'return': 'Unchanged local evaluation return',
    'fall_rate': 'Observed fall fraction',
    'applied_target_rate_rms_rad_s': 'Applied target rate RMS (rad/s)',
    'applied_target_acceleration_rms_rad_s2': 'Applied target acceleration RMS (rad/s²)',
    'actual_joint_acceleration_rms_rad_s2': 'Actual joint acceleration RMS at 50 Hz (rad/s²)',
    'raw_action_clipped_fraction': 'Raw action clipping fraction',
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_value(row, key):
    if key == 'fall_rate':
        return float(row['fell'])
    value = row[key] if key in row else row['study_metrics'][key]
    return None if value is None else float(value)


def savefig(fig, directory, name):
    for suffix in ['png', 'pdf']:
        fig.savefig(directory / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('study_dir', type=Path)
    args = ap.parse_args()
    study = args.study_dir.resolve()
    protocol_path = study / 'protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    inputs = {str(protocol_path): protocol_hash}
    rows, cases, aggregates = [], {}, {}
    expected_n = len(protocol['evaluation']['seeds']) * protocol['evaluation']['episodes_per_seed']
    for entry in protocol['checkpoints']:
        for condition in protocol['evaluation']['conditions']:
            path = study / 'evaluation/cases' / entry['id'] / f'{condition}.json'
            raw = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            case = json.loads(raw)
            assert case['status'] == 'completed' and case['pairing_verified']
            assert case['protocol_sha256'] == protocol_hash
            episodes = case['episodes']
            assert len(episodes) == expected_n
            assert len({(e['evaluation_seed'], e['env_id']) for e in episodes}) == expected_n
            row = {'variant': entry['variant'], 'train_seed': entry['train_seed'], 'condition': condition,
                   'episodes': expected_n}
            for metric in METRICS:
                values = [value for e in episodes if (value := episode_value(e, metric)) is not None]
                assert all(math.isfinite(x) for x in values), metric
                row[metric] = stats.mean(values) if values else None
                row[metric+'_valid_episodes'] = len(values)
            row['mean_duration_s'] = stats.mean(e['duration_s'] for e in episodes)
            rows.append(row)
            cases[(entry['variant'], entry['train_seed'], condition)] = row
    group_rows = []
    for condition in protocol['evaluation']['conditions']:
        aggregates[condition] = {}
        for group in GROUPS:
            selected = [r for r in rows if r['condition'] == condition and r['variant'] == group]
            assert sorted(r['train_seed'] for r in selected) == protocol['training']['seeds']
            aggregates[condition][group] = {}
            for metric in METRICS:
                values = [r[metric] for r in selected if r[metric] is not None]
                value = {'mean': stats.mean(values) if values else None,
                         'sd_across_training_seeds': stats.stdev(values) if len(values)>1 else None,
                         'n_training_seeds': len(selected), 'n_valid_training_seeds': len(values),
                         'seed_means': {r['train_seed']:r[metric] for r in selected}}
                aggregates[condition][group][metric] = value
                group_rows.append({'condition': condition, 'variant': group, 'metric': metric,
                                   'mean': value['mean'], 'sd_across_training_seeds': value['sd_across_training_seeds'],
                                   'n_training_seeds': len(values)})
    differences = []
    for condition in protocol['evaluation']['conditions']:
        for reference, candidate in [('baseline', 'full_dr'), ('fixed_delay', 'full_dr')]:
            for metric in METRICS:
                deltas = []
                for seed in protocol['training']['seeds']:
                    first, second = cases[(candidate,seed,condition)][metric], cases[(reference,seed,condition)][metric]
                    deltas.append(None if first is None or second is None else first-second)
                valid_deltas = [d for d in deltas if d is not None]
                differences.append({'condition': condition, 'candidate': candidate, 'reference': reference,
                                    'metric': metric, 'mean_paired_difference': stats.mean(valid_deltas) if valid_deltas else None,
                                    'sd_paired_difference': stats.stdev(valid_deltas) if len(valid_deltas)>1 else None,
                                    'n_valid_pairs':len(valid_deltas),
                                    **{f'difference_seed{seed}': delta for seed, delta in zip(protocol['training']['seeds'], deltas)}})
    write_csv(out / 'per_training_seed.csv', rows)
    write_csv(out / 'group_summary.csv', group_rows)
    write_csv(out / 'paired_seed_differences.csv', differences)
    summary = {'protocol_sha256': protocol_hash, 'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'aggregation': 'mean of 128 episode metrics per policy, then mean and sample SD over three independent training seeds',
               'metric_definitions': METRICS, 'inputs_sha256': inputs, 'per_seed': rows,
               'groups': aggregates, 'paired_seed_differences': differences,
               'limitations': protocol['limitations']}
    dump(out / 'summary.json', summary)

    for name, metrics in [
        ('performance', ['velocity_x_rmse_m_s', 'distance_x_m', 'abs_final_heading_yaw_deg', 'abs_lateral_displacement_m', 'return', 'fall_rate']),
        ('smoothness', ['applied_target_rate_rms_rad_s', 'applied_target_acceleration_rms_rad_s2', 'actual_joint_acceleration_rms_rad_s2', 'raw_action_clipped_fraction']),
    ]:
        fig, axes = plt.subplots(len(metrics)//2, 2, figsize=(12, 3.7*(len(metrics)//2)), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            for index, group in enumerate(GROUPS):
                xs = [j + (index-1)*.24 for j in range(2)]
                vals = [aggregates[c][group][metric] for c in ['nominal', 'randomized']]
                ax.bar(xs, [v['mean'] if v['mean'] is not None else math.nan for v in vals], width=.22, color=COLORS[group], alpha=.8,
                       yerr=[v['sd_across_training_seeds'] if v['sd_across_training_seeds'] is not None else math.nan for v in vals], capsize=4, label=LABELS[group])
                for x, condition in zip(xs, ['nominal', 'randomized']):
                    ys = [cases[(group, seed, condition)][metric] if cases[(group, seed, condition)][metric] is not None else math.nan for seed in protocol['training']['seeds']]
                    ax.scatter([x-.045, x, x+.045], ys, s=25, color='black', zorder=3)
            ax.set_xticks([0,1], ['Nominal physics', 'Randomized physics'])
            ax.set_title(METRICS[metric])
            if metric == 'fall_rate':
                ax.set_ylim(0, 1)
                for index, group in enumerate(GROUPS):
                    for j, condition in enumerate(['nominal','randomized']):
                        rate = aggregates[condition][group][metric]['mean']
                        ax.text(j+(index-1)*.24, rate+.025, f'{rate:.0%}', ha='center', fontsize=9)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5,.96))
        fig.suptitle('Sesame PPO | 3 training seeds × 3 methods', fontsize=17)
        fig.text(.5,.01,'Dots: training-seed means (128 paired episodes each). Error bars: sample SD across 3 training seeds, not confidence intervals.\nSame evaluation physics for every method; unchanged local reward. Target/joint derivatives sampled at 50 Hz.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.065,1,.91])
        savefig(fig,out,name)

    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax, metric, title in zip(axes, ['mean_episode_return','value_function'], ['Training episode return','PPO value-function loss']):
        for group in GROUPS:
            series = []
            for entry in [e for e in protocol['checkpoints'] if e['variant'] == group]:
                source = Path(entry['run_dir'])/'training.jsonl'
                training = [json.loads(s) for s in source.read_text().splitlines()]
                values = [r.get('losses',{}).get(metric) if metric=='value_function' else r.get(metric) for r in training]
                smoothed = [stats.mean(v for v in values[max(0,i-9):i+1] if v is not None)
                            if any(v is not None for v in values[max(0,i-9):i+1]) else math.nan for i in range(len(values))]
                series.append(smoothed)
            assert all(len(s)==250 for s in series)
            xs=[(i+1)*8192/1e6 for i in range(250)]
            means=[stats.mean(values) if all(math.isfinite(v) for v in values) else math.nan for values in zip(*series)]
            sd=[stats.stdev(values) if all(math.isfinite(v) for v in values) else math.nan for values in zip(*series)]
            ax.plot(xs,means,label=LABELS[group],color=COLORS[group])
            ax.fill_between(xs,[m-s for m,s in zip(means,sd)],[m+s for m,s in zip(means,sd)],color=COLORS[group],alpha=.15)
        ax.set(title=title,xlabel='Environment transitions (millions)')
        ax.grid(alpha=.2)
        if metric=='value_function': ax.set_yscale('log')
    axes[0].legend()
    fig.suptitle('10-update trailing means; shading = sample SD across 3 training seeds')
    fig.text(.5,.01,'Training physical conditions differ across methods. Use paired held-out evaluation to compare performance.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94])
    savefig(fig,out,'learning_curves')

    lines=['# 多种子配对评测结果','', '每项先统计每个训练种子的 128 个完整评测回合，再计算三个训练种子的均值 ± 样本标准差。误差条不是置信区间。','',
           '| 条件 | 方法 | 逐时刻前向速度 RMSE (m/s) | 回合前向位移 (m) | 终点绝对偏航 (°) | 终点绝对侧移 (m) | 跌倒率 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for condition in ['nominal','randomized']:
        for group in GROUPS:
            a=aggregates[condition][group]
            values=[f"{a[m]['mean']:.4f} ± {a[m]['sd_across_training_seeds']:.4f}" for m in ['velocity_x_rmse_m_s','distance_x_m','abs_final_heading_yaw_deg','abs_lateral_displacement_m']]
            lines.append('| '+' | '.join([condition,group,*values,f"{a['fall_rate']['mean']:.2%}"])+' |')
    lines += ['', 'baseline 与 full_dr 比较的是整套训练方案；full_dr 与 fixed_delay 比较时延/响应随机变化。单次最佳模型不代表跨种子结论。',
              '', '[性能图](performance.png) · [平滑性图](smoothness.png) · [学习曲线](learning_curves.png)',
              '', '[逐训练种子数据](per_training_seed.csv) · [配对种子差值](paired_seed_differences.csv) · [完整统计](summary.json)']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed','analysis_dir':str(out),'cases':len(rows),'training_seeds':3},ensure_ascii=False))


if __name__=='__main__': main()
