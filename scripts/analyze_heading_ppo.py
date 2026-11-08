#!/usr/bin/env python3
"""Analyze saved heading-PPO evaluation metadata; never load models or simulators."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('full_dr', 'heading_v1')
LABELS = {'full_dr': 'Original PPO + DR', 'heading_v1': 'Heading PPO + DR'}
CN_LABELS = {'full_dr': '旧 PPO（域随机化）', 'heading_v1': '新 PPO（增加航向惩罚）'}
COLORS = {'full_dr': '#2563eb', 'heading_v1': '#d97706'}
METRICS = {
    'heading_rms_deg': ('Heading RMS (degrees)', '航向 RMS（°）'),
    'velocity_x_rmse_m_s': ('Forward velocity RMSE (m/s)', '前向速度 RMSE（m/s）'),
    'velocity_xy_tracking_rmse_m_s': ('Planar velocity RMSE (m/s)', '平面速度 RMSE（m/s）'),
    'distance_x_m': ('Forward displacement (m)', '前向位移（m）'),
    'abs_final_heading_yaw_deg': ('Absolute final heading (degrees)', '终点绝对航向（°）'),
    'abs_lateral_displacement_m': ('Absolute final lateral displacement (m)', '终点绝对侧移（m）'),
    'fall_rate': ('Observed fall fraction', '跌倒率'),
    'applied_target_rate_rms_rad_s': ('Applied target rate RMS (rad/s)', '执行目标变化率 RMS（rad/s）'),
    'applied_target_acceleration_rms_rad_s2': ('Applied target acceleration RMS (rad/s²)', '执行目标加速度 RMS（rad/s²）'),
    'actual_joint_acceleration_rms_rad_s2': ('Joint acceleration RMS at 50 Hz (rad/s²)', '实际关节加速度 RMS，50 Hz（rad/s²）'),
    'raw_action_clipped_fraction': ('Raw action clipping fraction', '原始动作裁剪比例'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(episode, metric):
    if metric == 'heading_rms_deg':
        return math.degrees(episode['study_metrics']['heading_rms_rad'])
    if metric == 'fall_rate':
        return float(episode['fell'])
    result = episode[metric] if metric in episode else episode['study_metrics'][metric]
    return None if result is None else float(result)


def savefig(plt, figure, out, name):
    for suffix in ('png', 'pdf'):
        figure.savefig(out / (name + '.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_dir', type=Path)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    require(ROOT in study.parents, 'Study must be inside Sesame')
    protocol_path = study / 'evaluation_protocol.json'
    protocol = read(protocol_path)
    protocol_hash = sha(protocol_path)
    evaluation = study / 'evaluation'
    manifest = read(evaluation / 'evaluation_manifest.json')
    result, progress = read(evaluation / 'result.json'), read(evaluation / 'progress.json')
    signature = manifest['manifest_signature']
    require(canonical_hash(manifest['core']) == signature, 'Manifest signature mismatch')
    require(manifest['core']['protocol_sha256'] == protocol_hash, 'Manifest protocol mismatch')
    require(result['status'] == progress['status'] == 'completed', 'Evaluation not complete')
    require(result['protocol_sha256'] == progress['protocol_sha256'] == protocol_hash, 'Result/progress protocol mismatch')
    require(result['manifest_signature'] == progress['manifest_signature'] == signature, 'Result/progress signature mismatch')
    require(not manifest['core']['smoke'] and not result['smoke'], 'Formal evaluation required')
    settings = protocol['evaluation']
    require(manifest['core']['effective_evaluation'] == settings, 'Effective settings mismatch')
    require(settings['seeds'] == [15042, 16042] and settings['episodes_per_seed'] == 64, 'Wrong held-out sample')
    require(settings['conditions'] == ['nominal', 'randomized'], 'Wrong conditions')
    seeds = protocol['training']['seeds']
    require(seeds == [42, 43, 44], 'Three preregistered training seeds required')
    expected_ids = {(seed, env_id) for seed in settings['seeds'] for env_id in range(64)}
    entries = protocol['checkpoints']
    require(len(entries) == 6 and {(e['variant'], e['train_seed']) for e in entries} == {(g, s) for g in GROUPS for s in seeds}, 'Wrong model set')
    require(len(progress['completed_cases']) == result['cases_completed'] == 12, 'Expected 12 completed cases')
    inputs = {str(protocol_path): protocol_hash, str(evaluation / 'evaluation_manifest.json'): sha(evaluation / 'evaluation_manifest.json')}
    rows, cases, trajectories, paired = [], {}, {}, {}
    resolved = {e['id']: e for e in manifest['core']['resolved_checkpoints']}
    for entry in entries:
        checkpoint = Path(entry['checkpoint'])
        checkpoint_hash = sha(checkpoint)
        training_result = read(Path(entry['run_dir']) / 'result.json')
        require(training_result['status'] == 'completed', 'Incomplete model training')
        require(checkpoint_hash == training_result['checkpoint_sha256'] == resolved[entry['id']]['sha256'], 'Checkpoint digest mismatch')
        require(entry.get('sha256', checkpoint_hash) == checkpoint_hash, 'Protocol checkpoint mismatch')
        inputs[str(checkpoint)] = checkpoint_hash
        for condition in settings['conditions']:
            path = evaluation / 'cases' / entry['id'] / (condition + '.json')
            case = read(path)
            digest = sha(path)
            require(progress['completed_cases'][entry['id'] + '/' + condition]['sha256'] == digest, 'Case digest mismatch')
            inputs[str(path)] = digest
            require(case['status'] == 'completed' and case['pairing_verified'] and case['all_actions_observations_rewards_finite'], 'Case failed validity checks')
            require(case['protocol_sha256'] == protocol_hash and case['manifest_signature'] == signature, 'Case provenance mismatch')
            require(case['checkpoint_sha256'] == checkpoint_hash and case['variant'] == entry['variant'] and case['train_seed'] == entry['train_seed'], 'Case identity mismatch')
            episodes = case['episodes']
            require(len(episodes) == 128 and {(e['evaluation_seed'], e['env_id']) for e in episodes} == expected_ids, 'Missing/duplicate first episodes')
            require(case['evaluation_environment_variant'] == 'full_dr', 'Noncommon evaluation environment')
            for initial, reference in zip(case['initial_conditions'], case['pairing_references']):
                ref_path = Path(reference['path'])
                require(reference['verified'] and sha(ref_path) == reference['sha256'], 'Pairing digest mismatch')
                document = read(ref_path)
                require(document['initial_condition'] == initial and document['protocol_sha256'] == protocol_hash and document['manifest_signature'] == signature, 'Pairing evidence mismatch')
                key = (condition, initial['evaluation_seed'])
                require(key not in paired or paired[key] == initial, 'Different initial states/physical parameters across policies')
                paired[key] = initial
                inputs[str(ref_path)] = sha(ref_path)
            require(len(case['initial_conditions']) == len(case['pairing_references']) == 2, 'Missing paired seed')
            row = {'variant': entry['variant'], 'train_seed': entry['train_seed'], 'condition': condition, 'episodes': len(episodes), 'falls': sum(bool(e['fell']) for e in episodes)}
            for metric in METRICS:
                values = [value(e, metric) for e in episodes]
                valid = [x for x in values if x is not None]
                require(all(math.isfinite(x) for x in valid), 'Nonfinite metric: ' + metric)
                row[metric] = stats.mean(valid) if valid else None
                row[metric + '_valid_episodes'] = len(valid)
            row['mean_duration_s'] = stats.mean(e['duration_s'] for e in episodes)
            rows.append(row)
            cases[(entry['variant'], entry['train_seed'], condition)] = row
            trajectory_path = Path(case['trajectory_file'])
            require(sha(trajectory_path) == case['trajectory_sha256'], 'Trajectory digest mismatch')
            trajectory = read(trajectory_path)
            require(trajectory['protocol_sha256'] == protocol_hash and trajectory['manifest_signature'] == signature, 'Trajectory provenance mismatch')
            trajectories[(entry['variant'], entry['train_seed'], condition)] = trajectory
            inputs[str(trajectory_path)] = case['trajectory_sha256']

    aggregates, group_rows, differences = {}, [], []
    for condition in settings['conditions']:
        aggregates[condition] = {}
        for group in GROUPS:
            selected = [cases[(group, seed, condition)] for seed in seeds]
            aggregates[condition][group] = {}
            for metric in METRICS:
                vals = [r[metric] for r in selected if r[metric] is not None]
                item = {'mean': stats.mean(vals) if vals else None, 'sd_across_training_seeds': stats.stdev(vals) if len(vals) > 1 else None,
                        'n_training_seeds': len(selected), 'n_valid_training_seeds': len(vals), 'seed_means': {r['train_seed']: r[metric] for r in selected}}
                aggregates[condition][group][metric] = item
                group_rows.append({'condition': condition, 'variant': group, 'metric': metric, 'mean': item['mean'], 'sd_across_training_seeds': item['sd_across_training_seeds'], 'n_valid_training_seeds': len(vals)})
        for metric in METRICS:
            deltas = [cases[('heading_v1', seed, condition)][metric] - cases[('full_dr', seed, condition)][metric]
                      if cases[('heading_v1', seed, condition)][metric] is not None and cases[('full_dr', seed, condition)][metric] is not None else None for seed in seeds]
            valid = [d for d in deltas if d is not None]
            old, new = (aggregates[condition][g][metric]['mean'] for g in GROUPS)
            differences.append({'condition': condition, 'metric': metric, 'candidate': 'heading_v1', 'reference': 'full_dr',
                                'mean_paired_difference': stats.mean(valid) if valid else None, 'sd_paired_difference': stats.stdev(valid) if len(valid) > 1 else None,
                                'percent_change_of_group_mean': 100.0 * (new - old) / abs(old) if old not in (None, 0) and new is not None else None,
                                'n_valid_pairs': len(valid), **{'difference_seed' + str(seed): delta for seed, delta in zip(seeds, deltas)}})
    limitations = [
        '每组只有 3 个独立训练种子；报告样本标准差，不宣称统计显著性。',
        '评估随机化范围与训练范围一致，不能外推到真实硬件或更极端的地形/扰动。',
        '旧组与新组都使用域随机化；本实验只新增航向惩罚，不能代表完整 Go2 奖励函数迁移。',
        '同一训练种子编号进行差值配对；不同策略的训练轨迹会随奖励改变而分化。',
        '目标/关节加速度使用 50 Hz 差分 RMS，排除重置跳变，不是 500 Hz 物理峰值或频谱抖动率。',
        '总训练回报使用不同奖励定义，不直接据此判断谁更好；本报告比较共同物理指标。',
        '零跌倒仅表示这些评估回合内未观察到跌倒，不能证明跌倒概率为零。',
        '固定展示训练种子 42、评估种子 15042、环境 0 的首回合，不按表现挑选。',
    ]
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    summary = {'status': 'completed', 'protocol_sha256': protocol_hash, 'manifest_signature': signature, 'analysis_source_sha256': sha(__file__),
               'total_episodes': sum(r['episodes'] for r in rows), 'total_falls': sum(r['falls'] for r in rows), 'pairing_verified_all_cases': True,
               'aggregation': '128 episodes per policy and condition; mean and sample SD across 3 independent training-seed means',
               'metric_definitions': METRICS, 'inputs_sha256': inputs, 'per_seed': rows, 'groups': aggregates,
               'paired_seed_differences': differences, 'limitations': limitations}
    dump(out / 'summary.json', summary)
    csv_write(out / 'per_training_seed.csv', rows)
    csv_write(out / 'group_summary.csv', group_rows)
    csv_write(out / 'paired_seed_differences.csv', differences)

    os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, metrics in [
        ('performance', ['heading_rms_deg', 'velocity_x_rmse_m_s', 'velocity_xy_tracking_rmse_m_s', 'distance_x_m', 'abs_final_heading_yaw_deg', 'abs_lateral_displacement_m']),
        ('smoothness', ['applied_target_rate_rms_rad_s', 'applied_target_acceleration_rms_rad_s2', 'actual_joint_acceleration_rms_rad_s2', 'raw_action_clipped_fraction']),
    ]:
        fig, axes = plt.subplots(len(metrics) // 2, 2, figsize=(12, 3.7 * (len(metrics) // 2)), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            for index, group in enumerate(GROUPS):
                xs = [j + (index - .5) * .30 for j in range(2)]
                vals = [aggregates[c][group][metric] for c in settings['conditions']]
                ax.bar(xs, [v['mean'] if v['mean'] is not None else math.nan for v in vals], width=.27, color=COLORS[group], alpha=.8,
                       yerr=[v['sd_across_training_seeds'] if v['sd_across_training_seeds'] is not None else math.nan for v in vals], capsize=4, label=LABELS[group])
                for x, condition in zip(xs, settings['conditions']):
                    ys = [cases[(group, seed, condition)][metric] for seed in seeds]
                    ax.scatter([x-.045, x, x+.045], ys, s=24, color='black', zorder=3)
            ax.set_xticks([0, 1], ['Nominal physics', 'Randomized physics'])
            ax.set_title(METRICS[metric][0]); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, .96))
        fig.suptitle('Sesame PPO | absolute-heading reward comparison', fontsize=17)
        fig.text(.5, .01, 'Dots: 3 independent training-seed means, 128 paired episodes per seed/condition. Error bars: sample SD, not confidence intervals.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .045, 1, .91]); savefig(plt, fig, out, name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
    for col, condition in enumerate(settings['conditions']):
        for group in GROUPS:
            trajectory = trajectories[(group, 42, condition)]
            samples = trajectory['samples']; ts = [f['time_s'] + .02 for f in samples]
            def yaw_deg(state):
                w, x, y, z = state[3:7]
                return math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
            axes[0, col].plot(ts, [yaw_deg(f['post_root_state_w']) for f in samples], color=COLORS[group], label=LABELS[group])
            axes[1, col].plot(ts, [f['post_root_state_w'][7] for f in samples], color=COLORS[group], label=LABELS[group])
        axes[0, col].axhline(0, color='black', ls='--', lw=1); axes[1, col].axhline(.1, color='black', ls='--', lw=1)
        axes[0, col].set(title=condition.capitalize(), ylabel='World yaw (degrees)')
        axes[1, col].set(xlabel='Time (s)', ylabel='World forward speed (m/s)')
        for ax in axes[:, col]: ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Fixed first episode | training seed 42, evaluation seed 15042, environment 0')
    fig.text(.5, .01, 'Selected before seeing results. Terminal sample included; this single trajectory does not replace the multi-seed evaluation.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95]); savefig(plt, fig, out, 'representative_trajectories')

    def fmt(item, metric):
        if item['mean'] is None:
            return '无有效样本'
        if metric == 'fall_rate':
            return '{:.2%}'.format(item['mean'])
        return '{:.4f} ± {:.4f}'.format(item['mean'], item['sd_across_training_seeds'])
    lines = ['# 航向奖励 PPO 对照评估', '', '旧 PPO 与新 PPO 都使用域随机化；新组只增加 −0.5 × (1 − cos(航向误差)) 的奖励率，控制步长为 0.02 秒。', '',
             '6 个最终模型 × 2 类物理条件 × 2 个评估种子 × 64 个首回合，共 1536 回合。所有模型在同一类条件下的初始状态和实际物理参数逐项一致。每项先取单个训练种子的 128 回合均值，再报告三个训练种子的均值 ± 样本标准差。', '']
    def change(condition, metric):
        return next(d['percent_change_of_group_mean'] for d in differences
                    if d['condition'] == condition and d['metric'] == metric)
    random_heading = aggregates['randomized']
    lines += ['## 本次结果解读', '',
              '随机化条件下，航向 RMS 从 {:.2f}° 变为 {:.2f}°（{:+.1f}%），三个训练种子均下降。标称条件下的航向 RMS 变化为 {:+.1f}%。这支持新增航向项改善本任务朝向控制，但只有三个训练种子，不作统计显著性声明。'.format(
                  random_heading['full_dr']['heading_rms_deg']['mean'], random_heading['heading_v1']['heading_rms_deg']['mean'],
                  change('randomized', 'heading_rms_deg'), change('nominal', 'heading_rms_deg')), '',
              '速度与平滑性存在取舍：随机化条件的前向速度 RMSE 变化为 {:+.1f}%，标称条件为 {:+.1f}%；执行目标加速度 RMS 分别变化 {:+.1f}% 和 {:+.1f}%。因此不能把航向改善等同于抖动减少或各项性能全面提升。'.format(
                  change('randomized', 'velocity_x_rmse_m_s'), change('nominal', 'velocity_x_rmse_m_s'),
                  change('randomized', 'applied_target_acceleration_rms_rad_s2'), change('nominal', 'applied_target_acceleration_rms_rad_s2')), '',
              '随机化条件的终点绝对侧移变化为 {:+.1f}%。目标速度为 0.10 m/s，10 秒期望位移约 1 m；旧、新策略平均位移分别为 {:.3f} m、{:.3f} m，仍有速度超调，不能把走得更远直接解释为前进效率更高。'.format(
                  change('randomized', 'abs_lateral_displacement_m'), random_heading['full_dr']['distance_x_m']['mean'],
                  random_heading['heading_v1']['distance_x_m']['mean']), '',
              '两个实验组各评估 768 回合，均未观测到跌倒。本实验不能据此说明新奖励降低了跌倒风险。', '',
              '后续建议保留航向项，再单独研究一个平滑性约束，并以速度 RMSE 和侧移作为约束指标；本报告未启动额外训练。', '']
    for condition, title in [('randomized', '随机化物理条件'), ('nominal', '标称物理条件')]:
        lines += ['## ' + title, '', '| 指标 | 旧 PPO | 新 PPO | 新相对旧的变化 |', '| --- | --- | --- | --- |']
        for metric in METRICS:
            change = next(d for d in differences if d['condition'] == condition and d['metric'] == metric)['percent_change_of_group_mean']
            change_text = '两组均为 0' if metric == 'fall_rate' and all(aggregates[condition][g][metric]['mean'] == 0 for g in GROUPS) else ('不计算（旧值为零/缺失）' if change is None else '{:+.1f}%'.format(change))
            lines.append('| ' + ' | '.join([METRICS[metric][1], fmt(aggregates[condition]['full_dr'][metric], metric), fmt(aggregates[condition]['heading_v1'][metric], metric), change_text]) + ' |')
        lines += ['', '变化百分比为 (新组均值 − 旧组均值) / |旧组均值|；误差类指标负值表示下降。前向位移的期望值约为 1 m/10 s，越大不一定越好。', '']
    lines += ['## 逐训练种子主指标', '', '| 条件 | 种子 | 旧航向 RMS（°） | 新航向 RMS（°） | 旧速度 RMSE（m/s） | 新速度 RMSE（m/s） |', '| --- | --- | --- | --- | --- | --- |']
    for condition in settings['conditions']:
        for seed in seeds:
            a, b = (cases[(g, seed, condition)] for g in GROUPS)
            lines.append('| {} | {} | {:.3f} | {:.3f} | {:.5f} | {:.5f} |'.format(condition, seed, a['heading_rms_deg'], b['heading_rms_deg'], a['velocity_x_rmse_m_s'], b['velocity_x_rmse_m_s']))
    lines += ['', '## 解释与边界', ''] + ['- ' + item for item in limitations]
    lines += ['', '[性能图](performance.png) · [平滑性图](smoothness.png) · [固定首回合轨迹](representative_trajectories.png)', '',
              '[逐种子数据](per_training_seed.csv) · [配对差值](paired_seed_differences.csv) · [完整统计](summary.json)', '']
    video = study / 'comparison_videos/ppo_heading_comparison_seed15042.mp4'
    if video.exists():
        lines += ['[新旧 PPO 并排视频](../comparison_videos/ppo_heading_comparison_seed15042.mp4)', '',
                  '视频固定使用训练种子 42、随机化评估种子 15042 的首回合；两侧物理参数和初始状态严格配对。视频使用独立单环境仿真，与批量统计评估的随机样本不完全相同，不能用单段视频代替多种子结论。', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    # Standalone HTML uses only local generated assets and escaped report text.
    blocks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Sesame 航向奖励 PPO 对照评估</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#182235}pre{white-space:pre-wrap}img{max-width:100%}a{color:#2563eb}</style><body>', '<pre>' + html.escape('\n'.join(lines)) + '</pre>']
    blocks.extend('<img src="' + name + '.png" alt="' + name + '">' for name in ('performance', 'smoothness', 'representative_trajectories'))
    blocks.append('</body></html>')
    (out / 'summary.html').write_text('\n'.join(blocks))
    print(json.dumps({'status': 'completed', 'analysis_dir': str(out), 'episodes': summary['total_episodes'], 'falls': summary['total_falls'], 'pairing_verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
