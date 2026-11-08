#!/usr/bin/env python3
"""Validate and summarize paired PPO/SAC/branching-DQN metadata without models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics as stats

from analyze_heading_ppo import ROOT, METRICS as HEADING_METRICS
from analyze_heading_ppo import sha, canonical_hash, require, read, dump, csv_write, value, savefig

GROUPS = ('ppo', 'sac', 'dqn')
LABELS = {'ppo': 'PPO', 'sac': 'SAC', 'dqn': 'DQN'}
COLORS = {'ppo': '#2563eb', 'sac': '#d97706', 'dqn': '#059669'}
METRICS = dict(HEADING_METRICS)
METRICS.update({
    'velocity_x_rmse_m_s': ('Forward velocity error, RMSE (m/s)', '前向速度误差（RMSE，m/s）'),
    'velocity_xy_tracking_rmse_m_s': ('Planar velocity error, RMSE (m/s)', '平面速度误差（RMSE，m/s）'),
    'heading_rms_deg': ('Heading error, RMS (degrees)', '航向误差（RMS，°）'),
    'abs_final_heading_yaw_deg': ('Absolute final heading error (degrees)', '终点偏航（°）'),
    'abs_lateral_displacement_m': ('Absolute final lateral displacement (m)', '终点侧移（m）'),
    'applied_target_rate_rms_rad_s': ('Joint target velocity, RMS (rad/s)', '关节目标变化率（RMS，rad/s）'),
    'applied_target_acceleration_rms_rad_s2': ('Joint target acceleration, RMS (rad/s²)', '关节目标加速度（RMS，rad/s²）'),
    'actual_joint_acceleration_rms_rad_s2': ('Joint acceleration at 50 Hz, RMS (rad/s²)', '实际关节加速度（RMS，rad/s²）'),
    'return': ('Episode return', '回合总奖励'),
    'duration_s': ('Episode duration (s)', '回合时长（s）'),
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_dir', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Analyze explicitly marked smoke artifacts only')
    args = parser.parse_args()
    study = args.study_dir.resolve()
    require(study.is_relative_to(ROOT), 'Study must remain inside Sesame')
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
    require(bool(manifest['core']['smoke']) == bool(result['smoke']) == args.smoke, 'Smoke/formal scope mismatch')
    require(manifest['core']['reward_version'] == 'absolute_heading_v1' and manifest['core']['heading_penalty_scale'] == 0.5,
            'Wrong common reward')
    settings = manifest['core']['effective_evaluation']
    if not args.smoke:
        require(settings == protocol['evaluation'], 'Effective settings mismatch')
        require(settings['seeds'] == [25042, 26042] and settings['episodes_per_seed'] == 64 and settings['num_envs'] == 256,
                'Wrong held-out sample')
        require(protocol['training']['seeds'] == [42, 43, 44], 'Three frozen training seeds required')
    require(settings['conditions'] == ['nominal', 'randomized'], 'Wrong conditions')
    require(settings['environment_variant'] == 'heading_full_dr_for_all_policies', 'Wrong common environment')
    entries = protocol['checkpoints']
    groups = tuple(g for g in GROUPS if any(e['variant'] == g for e in entries))
    seeds = sorted({e['train_seed'] for e in entries})
    require(len({(e['variant'], e['train_seed']) for e in entries}) == len(entries), 'Duplicate model')
    require({(e['variant'], e['train_seed']) for e in entries} == {(g, s) for g in groups for s in seeds},
            'Method/training seed grid is incomplete')
    if not args.smoke:
        require(groups == GROUPS and seeds == [42, 43, 44] and len(entries) == 9, 'Nine final models required')
    expected_case_count = len(entries) * len(settings['conditions'])
    require(len(progress['completed_cases']) == result['cases_completed'] == expected_case_count, 'Missing completed cases')
    expected_ids = {(seed, env_id) for seed in settings['seeds'] for env_id in range(settings['episodes_per_seed'])}
    inputs = {str(protocol_path): protocol_hash, str(evaluation / 'evaluation_manifest.json'): sha(evaluation / 'evaluation_manifest.json')}
    rows, episodes_flat, cases, trajectories, paired, compute = [], [], {}, {}, {}, []
    training_curves = {}
    resolved = {e['id']: e for e in manifest['core']['resolved_checkpoints']}
    for entry in entries:
        checkpoint = Path(entry['checkpoint'])
        checkpoint_hash = sha(checkpoint)
        training_path = Path(entry['run_dir']) / 'result.json'
        training = read(training_path)
        require(training['status'] == 'completed', 'Incomplete model training')
        require(checkpoint_hash == training['checkpoint_sha256'] == resolved[entry['id']]['sha256'], 'Checkpoint digest mismatch')
        require(entry.get('sha256', checkpoint_hash) == checkpoint_hash, 'Protocol checkpoint mismatch')
        require(sha(training_path) == resolved[entry['id']]['training_result_sha256'], 'Training result changed')
        require(training['reward_version'] == 'absolute_heading_v1' and training['heading_penalty_scale'] == 0.5,
                'Training reward mismatch')
        inputs[str(checkpoint)] = checkpoint_hash
        inputs[str(training_path)] = sha(training_path)
        compute.append({'variant': entry['variant'], 'train_seed': entry['train_seed'],
                        'environment_transitions_total': training.get('environment_transitions_total'),
                        'wall_seconds_before_shutdown': training.get('wall_seconds_before_shutdown'),
                        'training_wall_seconds': training.get('training_wall_seconds'),
                        'optimizer_updates': training.get('optimizer_updates'),
                        'torch_peak_allocated_bytes': training.get('training_torch_memory', {}).get('peak_allocated_bytes'),
                        'torch_peak_reserved_bytes': training.get('training_torch_memory', {}).get('peak_reserved_bytes')})
        training_log_path = Path(entry['run_dir']) / 'training.jsonl'
        history = [json.loads(line) for line in training_log_path.read_text().splitlines() if line.strip()]
        require(history and history[-1]['environment_transitions'] == training['environment_transitions_total'],
                'Training curve is incomplete')
        require(all(b['environment_transitions'] > a['environment_transitions'] for a, b in zip(history, history[1:])),
                'Training transition coordinates must increase')
        training_curves[(entry['variant'], entry['train_seed'])] = history
        inputs[str(training_log_path)] = sha(training_log_path)
        for condition in settings['conditions']:
            path = evaluation / 'cases' / entry['id'] / (condition + '.json')
            case = read(path)
            digest = sha(path)
            require(progress['completed_cases'][entry['id'] + '/' + condition]['sha256'] == digest, 'Case digest mismatch')
            inputs[str(path)] = digest
            require(case['status'] == 'completed' and case['pairing_verified'] and case['all_actions_observations_rewards_finite'],
                    'Case failed validity checks')
            require(case['protocol_sha256'] == protocol_hash and case['manifest_signature'] == signature, 'Case provenance mismatch')
            require(case['checkpoint_sha256'] == checkpoint_hash and case['variant'] == entry['variant'] and case['train_seed'] == entry['train_seed'],
                    'Case identity mismatch')
            require(case['reward_version'] == 'absolute_heading_v1' and case['heading_penalty_scale'] == 0.5, 'Case reward mismatch')
            episodes = case['episodes']
            require(len(episodes) == len(expected_ids) and {(e['evaluation_seed'], e['env_id']) for e in episodes} == expected_ids,
                    'Missing or duplicate first episodes')
            # "full_dr" is the inherited sampler's physical-domain label;
            # the manifest and case reward fields identify heading_v1.
            require(case['evaluation_environment_variant'] == 'full_dr', 'Noncommon physical environment')
            require(len(case['initial_conditions']) == len(case['pairing_references']) == len(settings['seeds']), 'Missing paired seed')
            for initial, reference in zip(case['initial_conditions'], case['pairing_references'], strict=True):
                ref_path = Path(reference['path'])
                require(reference['verified'] and sha(ref_path) == reference['sha256'], 'Pairing digest mismatch')
                document = read(ref_path)
                require(document['initial_condition'] == initial and document['protocol_sha256'] == protocol_hash and document['manifest_signature'] == signature,
                        'Pairing evidence mismatch')
                key = (condition, initial['evaluation_seed'])
                require(key not in paired or paired[key] == initial, 'Different initial states or physical parameters across methods')
                paired[key] = initial
                inputs[str(ref_path)] = sha(ref_path)
            row = {'variant': entry['variant'], 'train_seed': entry['train_seed'], 'condition': condition,
                   'episodes': len(episodes), 'falls': sum(bool(e['fell']) for e in episodes)}
            for metric in METRICS:
                vals = [value(e, metric) for e in episodes]
                valid = [v for v in vals if v is not None]
                require(all(math.isfinite(v) for v in valid), 'Nonfinite metric: ' + metric)
                row[metric] = stats.mean(valid) if valid else None
                row[metric + '_valid_episodes'] = len(valid)
            rows.append(row)
            cases[(entry['variant'], entry['train_seed'], condition)] = row
            for episode in episodes:
                episodes_flat.append({'variant': entry['variant'], 'train_seed': entry['train_seed'], 'condition': condition,
                                      'evaluation_seed': episode['evaluation_seed'], 'env_id': episode['env_id'],
                                      **{metric: value(episode, metric) for metric in METRICS}})
            trajectory_path = Path(case['trajectory_file'])
            require(sha(trajectory_path) == case['trajectory_sha256'], 'Trajectory digest mismatch')
            trajectory = read(trajectory_path)
            require(trajectory['protocol_sha256'] == protocol_hash and trajectory['manifest_signature'] == signature, 'Trajectory provenance mismatch')
            trajectories[(entry['variant'], entry['train_seed'], condition)] = trajectory
            inputs[str(trajectory_path)] = case['trajectory_sha256']

    aggregates, group_rows, differences = {}, [], []
    for condition in settings['conditions']:
        aggregates[condition] = {}
        for group in groups:
            selected = [cases[(group, seed, condition)] for seed in seeds]
            aggregates[condition][group] = {}
            for metric in METRICS:
                vals = [r[metric] for r in selected if r[metric] is not None]
                item = {'mean': stats.mean(vals) if vals else None,
                        'sd_across_training_seeds': stats.stdev(vals) if len(vals) > 1 else None,
                        'n_training_seeds': len(selected), 'n_valid_training_seeds': len(vals),
                        'seed_means': {r['train_seed']: r[metric] for r in selected}}
                aggregates[condition][group][metric] = item
                group_rows.append({'condition': condition, 'variant': group, 'metric': metric, 'mean': item['mean'],
                                   'sd_across_training_seeds': item['sd_across_training_seeds'], 'n_valid_training_seeds': len(vals)})
        if 'ppo' in groups:
            for candidate in (g for g in groups if g != 'ppo'):
                for metric in METRICS:
                    deltas = [cases[(candidate, seed, condition)][metric] - cases[('ppo', seed, condition)][metric]
                              if cases[(candidate, seed, condition)][metric] is not None and cases[('ppo', seed, condition)][metric] is not None
                              else None for seed in seeds]
                    valid = [d for d in deltas if d is not None]
                    reference, new = (aggregates[condition][g][metric]['mean'] for g in ('ppo', candidate))
                    differences.append({'condition': condition, 'metric': metric, 'candidate': candidate, 'reference': 'ppo',
                                        'mean_paired_difference': stats.mean(valid) if valid else None,
                                        'sd_paired_difference': stats.stdev(valid) if len(valid) > 1 else None,
                                        'percent_change_of_group_mean': 100 * (new - reference) / abs(reference)
                                        if reference not in (None, 0) and new is not None else None,
                                        'n_valid_pairs': len(valid),
                                        **{'difference_seed' + str(seed): delta for seed, delta in zip(seeds, deltas)}})
    limitations = [
        'DQN 使用分支网络和 Double DQN 更新，每个关节离散为 5 档动作；PPO、SAC 使用 8 维连续动作。',
        '三种方法使用相同的改进奖励、完整域随机化和真实仿真步预算；优化次数、网络结构、探索和观测归一化方式不同，计算量不等同。',
        'PPO 复用已有最终模型；SAC 与 DQN 从头训练。仅使用最终检查点，未按测试结果挑选模型。',
        '逐模型先平均首回合指标，再报告独立训练种子均值及样本标准差；不宣称统计显著性。',
        '相同训练种子编号用于描述差值，不意味着不同算法的训练轨迹配对。评估初态和实际物理参数已逐项配对。',
        '随机测试限于训练随机化范围、平地和固定 +X 方向 0.10 m/s 指令，不能代表真机或分布外效果。',
        '加速度是平滑性的测量指标，需结合速度误差和前向位移判断；不能只因加速度低就认为行走更好。',
        '零跌倒仅表示本次样本未观察到跌倒，不证明跌倒概率为零。',
        '训练曲线只作优化诊断：PPO 回报使用最近 100 回合，SAC/DQN 使用最近 256 回合；PPO 物理诊断取当次 rollout 完成回合，SAC/DQN 取最近 256 回合，探索策略也不同，不替代共同确定性评估。',
    ]
    if args.smoke:
        limitations.insert(0, '这是开发 smoke，不是正式效果实验；不得据此给出算法性能结论。')
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    summary = {'status': 'completed', 'smoke': args.smoke, 'protocol_sha256': protocol_hash, 'manifest_signature': signature,
               'analysis_source_sha256': sha(__file__), 'analysis_helper_sha256': sha(ROOT / 'scripts/analyze_heading_ppo.py'),
               'reward_version': 'absolute_heading_v1', 'heading_penalty_scale': 0.5,
               'total_episodes': sum(r['episodes'] for r in rows), 'total_falls': sum(r['falls'] for r in rows),
               'pairing_verified_all_cases': True,
               'aggregation': f"{len(expected_ids)} episodes per policy/condition; mean and sample SD across {len(seeds)} training-seed means",
               'metric_definitions': METRICS, 'inputs_sha256': inputs, 'per_seed': rows, 'groups': aggregates,
               'paired_seed_differences': differences, 'compute_accounting': compute, 'limitations': limitations}
    csv_write(out / 'per_episode.csv', episodes_flat)
    csv_write(out / 'per_training_seed.csv', rows)
    csv_write(out / 'group_summary.csv', group_rows)
    csv_write(out / 'training_compute.csv', compute)
    if differences:
        csv_write(out / 'paired_seed_differences.csv', differences)
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = ['velocity_x_rmse_m_s', 'heading_rms_deg', 'abs_final_heading_yaw_deg',
               'abs_lateral_displacement_m', 'applied_target_acceleration_rms_rad_s2', 'actual_joint_acceleration_rms_rad_s2',
               'distance_x_m', 'fall_rate']
    fig, axes = plt.subplots(4, 2, figsize=(12, 13.6))
    width = 0.8 / len(groups)
    for ax, metric in zip(axes.flat, metrics):
        for index, group in enumerate(groups):
            xs = [j + (index - (len(groups) - 1) / 2) * width for j in range(2)]
            vals = [aggregates[c][group][metric] for c in settings['conditions']]
            ax.bar(xs, [v['mean'] if v['mean'] is not None else math.nan for v in vals], width=width * .9,
                   color=COLORS[group], alpha=.8,
                   yerr=[v['sd_across_training_seeds'] or 0 for v in vals], capsize=3, label=LABELS[group])
            for x, condition in zip(xs, settings['conditions']):
                ys = [cases[(group, seed, condition)][metric] for seed in seeds]
                ax.scatter([x + (i - (len(seeds) - 1) / 2) * .03 for i in range(len(seeds))], ys,
                           s=20, color='black', zorder=3)
        ax.set_xticks([0, 1], ['Standard parameters', 'Randomized parameters'])
        ax.set_title(METRICS[metric][0]); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(groups), bbox_to_anchor=(.5, .972))
    fig.suptitle('Sesame | same reward, PPO / SAC / DQN', fontsize=16)
    fig.text(.5, .01, 'Dots: independent training-seed means. Error bars: sample SD, not confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .025, 1, .945]); savefig(plt, fig, out, 'performance')

    curve_specs = [('mean_episode_return', 'Training episode return', 1.0),
                   ('velocity_x_rmse_m_s', 'Training forward velocity RMSE (m/s)', 1.0),
                   ('heading_rms_rad', 'Training heading RMS (degrees)', 180.0 / math.pi),
                   ('applied_target_acceleration_rms_rad_s2', 'Training joint target acceleration RMS (rad/s²)', 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curve_rows = []
    for ax, (metric, title, scale) in zip(axes.flat, curve_specs):
        for group in groups:
            by_transition = {}
            for seed in seeds:
                selected = []
                for row in training_curves[(group, seed)]:
                    observed = row.get(metric) if metric == 'mean_episode_return' else row.get('diagnostics', {}).get(metric)
                    if observed is None:
                        continue
                    observed = float(observed) * scale
                    require(math.isfinite(observed), 'Nonfinite training diagnostic')
                    step = int(row['environment_transitions'])
                    selected.append((step, observed))
                    by_transition.setdefault(step, []).append(observed)
                    curve_rows.append({'variant': group, 'train_seed': seed, 'environment_transitions': step,
                                       'metric': metric, 'display_scale': scale, 'value': observed})
                ax.plot([s / 1e6 for s, _ in selected], [v for _, v in selected], color=COLORS[group],
                        linewidth=.75, alpha=.25)
            # Mean/SD only where all training seeds reported a valid value;
            # faint lines retain the individually available diagnostics.
            paired_steps = sorted(step for step, vals in by_transition.items() if len(vals) == len(seeds))
            means = [stats.mean(by_transition[step]) for step in paired_steps]
            deviations = [stats.stdev(by_transition[step]) if len(seeds) > 1 else 0.0 for step in paired_steps]
            xs = [step / 1e6 for step in paired_steps]
            ax.plot(xs, means, color=COLORS[group], linewidth=1.6, label=LABELS[group])
            ax.fill_between(xs, [m - s for m, s in zip(means, deviations)], [m + s for m, s in zip(means, deviations)],
                            color=COLORS[group], alpha=.10)
        ax.set(title=title, xlabel='Real simulation transitions (millions)')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Training diagnostics | same heading reward, different exploration and log windows', fontsize=14)
    fig.text(.5, .01, 'Faint: individual seeds. Bold/band: mean/sample SD at shared steps. PPO return window 100 episodes; SAC/DQN 256.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95]); savefig(plt, fig, out, 'training_curves')
    csv_write(out / 'training_curve_data.csv', curve_rows)

    def fmt(item, metric):
        if item['mean'] is None:
            return '无有效样本'
        if metric == 'fall_rate':
            return f"{item['mean']:.2%}"
        if item['sd_across_training_seeds'] is None:
            return f"{item['mean']:.5f}（单种子）"
        return f"{item['mean']:.5f} ± {item['sd_across_training_seeds']:.5f}"
    lines = ['# 同奖励下 PPO、SAC 与 DQN 仿真实验', '',
             '**范围：**' + ('开发 smoke，不能作为效果结论。' if args.smoke else '正式最终模型评估。'), '',
             '**基线方法：PPO、SAC、DQN。** 三种方法均使用加入航向项后的改进奖励（航向系数 0.5）和完整域随机化。正式实验中，每个训练种子使用 204.8 万真实仿真步。', '',
             'DQN 使用分支网络和 Double DQN 更新，每个关节离散为 5 档动作；PPO、SAC 使用连续动作。表中统一使用算法名称，具体实现差异保留在此说明。', '',
             f"{len(entries)} 个最终模型 × 2 种物理条件 × {len(settings['seeds'])} 个测试种子 × {settings['episodes_per_seed']} 个首回合，共 {summary['total_episodes']} 回合。测试种子 {settings['seeds']}；同条件下初态与实际物理参数逐项一致。", '',
             f"每项先求单模型的 {len(expected_ids)} 回合均值，再报告 {len(seeds)} 个训练种子的均值 ± 样本标准差。标准参数采用参考设置，随机参数在训练的域随机化范围内采样。", '',
             '## 测评指标', '',
             '**主指标为前向速度误差。** RMSE 表示均方根误差，RMS 表示均方根。目标是沿世界 +X 方向以 0.10 m/s 行走；下式的 mean 表示回合内控制步的平均。', '',
             '| 指标 | 计算方式 | 单位与判断标准 |',
             '| --- | --- | --- |',
             '| 前向速度误差（RMSE） | `√mean((vx−0.10)²)` | m/s，越低越好 |',
             '| 平面速度误差（RMSE） | `√mean((vx−0.10)²+vy²)` | m/s，同时检查前进和横向速度，越低越好 |',
             '| 航向误差（RMS） | `√mean(ψ²)`；ψ 是相对 +X 的偏航角，统一到 −180° 至 180° | °，越低越好 |',
             '| 终点偏航 | 终点偏航角的绝对值 | °，越低越好；与全回合航向 RMS 分开报告 |',
             '| 终点侧移 | 终点与起点的 y 坐标差的绝对值 | m，越低越好 |',
             '| 前向位移 | `x终点−x起点` | m，10 秒目标约 1 m，并非越远越好 |',
             '| 跌倒率 | 跌倒回合数 ÷ 总回合数 | %，越低越好 |',
             '| 关节目标加速度（RMS） | 对执行的关节目标角度做二阶差分，再求 RMS | rad/s²，结合速度和位移判断平滑性 |',
             '| 实际关节加速度（RMS） | 对仿真中的实际关节速度做一阶差分，再求 RMS | rad/s²，结合速度和位移判断平滑性 |', '',
             '跌倒判定为机身根节点高度低于 0.023 m，或直立度低于 0.5（约对应整体倾斜超过 60°）。越界和无效状态单独记录，不自动计入跌倒。', '',
             '加速度按 50 Hz（步长 0.02 秒）计算，对时间及 8 个关节取均方根，排除回合重置跳变；关节目标加速度从第 2 个控制步开始统计。这里的目标是经过动作裁剪、关节约束、时延和滞后后实际执行的目标，加速度数值不是抖动频率或物理瞬时峰值。', '',
             '辅助记录关节目标变化率 RMS、原始动作裁剪比例、回合总奖励和回合时长。训练曲线用于观察学习过程，最终效果以固定策略的共同评估为准。', '']
    for condition, title in [('nominal', '标准参数'), ('randomized', '随机参数')]:
        lines += ['## ' + title, '', '| 指标 | ' + ' | '.join(LABELS[g] for g in groups) + ' |',
                  '| --- | ' + ' | '.join('---' for g in groups) + ' |']
        for metric in METRICS:
            lines.append('| ' + ' | '.join([METRICS[metric][1]] + [fmt(aggregates[condition][g][metric], metric) for g in groups]) + ' |')
        lines += ['']
    lines += ['## 解释与边界', ''] + ['- ' + item for item in limitations]
    lines += ['', '[性能图](performance.png) · [训练曲线](training_curves.png) · [逐回合数据](per_episode.csv) · [逐种子数据](per_training_seed.csv) · [完整统计](summary.json)', '']
    text = '\n'.join(lines)
    (out / 'summary.md').write_text(text)
    import markdown
    body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    (out / 'summary.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sesame PPO/SAC/DQN</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 24px}table{border-collapse:collapse;display:block;overflow-x:auto}th,td{border:1px solid #dce4ed;padding:8px 12px;text-align:left}th{background:#edf3fa}pre{white-space:pre-wrap}img{max-width:100%}a{color:#1753b6}</style></head><body>' + body + '<img src="performance.png" alt="指标对照图"><img src="training_curves.png" alt="训练曲线"><p><a href="summary.json">JSON</a> · <a href="per_episode.csv">CSV</a></p></body></html>')
    artifact_names = ['per_episode.csv', 'per_training_seed.csv', 'group_summary.csv', 'training_compute.csv',
                      'training_curve_data.csv', 'performance.png', 'performance.pdf', 'training_curves.png',
                      'training_curves.pdf', 'summary.md', 'summary.html']
    if differences:
        artifact_names.append('paired_seed_differences.csv')
    summary['artifacts_sha256'] = {name: sha(out / name) for name in artifact_names}
    dump(out / 'summary.json', summary)
    print(json.dumps({'status': 'completed', 'smoke': args.smoke, 'analysis_dir': str(out),
                      'episodes': summary['total_episodes'], 'falls': summary['total_falls'], 'pairing_verified': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
