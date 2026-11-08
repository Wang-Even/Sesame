#!/usr/bin/env python3
"""Chinese report figures from frozen study JSON metadata; no model/simulator imports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

VARIANTS = ('baseline', 'full_dr', 'fixed_delay')
SEEDS = (42, 43, 44)
CONDITIONS = ('nominal', 'randomized')
LABELS = {'baseline': '固定参数基线', 'full_dr': '完整域随机化', 'fixed_delay': '固定延迟消融'}
COLORS = {'baseline': '#2563EB', 'full_dr': '#E07A1F', 'fixed_delay': '#17825D'}
CONDITION_LABELS = {'nominal': '标称物理条件', 'randomized': '随机物理条件'}
METRICS = {
    'velocity_x_rmse_m_s': ('瞬时前向速度 RMSE', 'm/s'),
    'velocity_xy_tracking_rmse_m_s': ('瞬时平面速度 RMSE', 'm/s'),
    'distance_x_m': ('世界 x 方向位移', 'm'),
    'abs_final_heading_yaw_deg': ('末端航向角绝对值', '°'),
    'abs_lateral_displacement_m': ('末端横向位移绝对值', 'm'),
    'average_speed_abs_error_m_s': ('回合平均速度绝对误差', 'm/s'),
    'return': ('本地评测回报', '无量纲'),
    'fall_rate': ('跌倒比例', '%'),
    'applied_target_rate_rms_rad_s': ('施加目标角速度 RMS', 'rad/s'),
    'applied_target_acceleration_rms_rad_s2': ('施加目标角加速度 RMS', 'rad/s²'),
    'actual_joint_acceleration_rms_rad_s2': ('实际关节角加速度 RMS（50 Hz）', 'rad/s²'),
    'raw_action_clipped_fraction': ('原始动作裁剪比例', '%'),
    'mean_duration_s': ('平均回合时长', 's'),
}
TRAIN_METRICS = {
    'return': ('训练回合回报', '无量纲'),
    'distance': ('训练回合前向位移', 'm'),
    'duration': ('训练回合时长', 's'),
    'value_loss': ('价值函数损失', '无量纲'),
}
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, sources):
    raw = Path(path).read_bytes()
    sources[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)


def read_jsonl(path, sources):
    raw = Path(path).read_bytes()
    sources[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def moments(values):
    valid = [float(v) for v in values if finite(v)]
    return {'mean': statistics.mean(valid) if valid else None,
            'sd': statistics.stdev(valid) if len(valid) > 1 else None, 'count': len(valid)}


def display_factor(metric):
    return 100 if metric in ('fall_rate', 'raw_action_clipped_fraction') else 1


def train_value(row, metric):
    if metric == 'return':
        return row.get('mean_episode_return')
    if metric == 'distance':
        return row.get('episode_metrics', {}).get('Episode/distance_x_m')
    if metric == 'duration':
        value = row.get('mean_episode_length_steps')
        return value / 50 if finite(value) else None
    return row.get('losses', {}).get('value_function')


def trailing(rows, metric, iteration, window):
    points = [(r['iteration'], train_value(r, metric)) for r in rows
              if iteration - window < r['iteration'] <= iteration]
    points = [(i, v) for i, v in points if finite(v)]
    return {'mean': statistics.mean(v for _, v in points) if points else None,
            'source_iterations': [i for i, _ in points], 'n_records': len(points)}


def configure_plotting():
    font_manager.fontManager.addfont(str(FONT))
    font_manager.fontManager.addfont(str(FONT.with_name('NotoSansCJK-Bold.ttc')))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 10.5, 'axes.titlesize': 12,
        'axes.titleweight': 'medium', 'axes.labelsize': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#A8B3C0', 'text.color': '#15283D',
        'axes.labelcolor': '#263B52', 'xtick.color': '#4B5E73',
        'ytick.color': '#4B5E73', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'grid.color': '#D9E1E9',
        'grid.alpha': .65, 'axes.unicode_minus': False,
        # NotoSansCJK is an OpenType/CFF collection; Matplotlib's Type42 path
        # mislabels its embedded outlines and corrupts text in Poppler readers.
        # Type3 embeds vector glyph outlines and renders the CJK text correctly.
        'pdf.fonttype': 3, 'ps.fonttype': 3, 'savefig.dpi': 180,
    })


def prettify(ax):
    ax.grid(axis='y', linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def header(fig, title, subtitle):
    fig.suptitle(title, x=.055, y=.974, ha='left', fontsize=18, fontweight='bold')
    fig.text(.055, .974-.52/fig.get_figheight(), subtitle, color='#52677D', fontsize=10.5)


def footer(fig, text):
    fig.text(.055, .025, text, va='bottom', fontsize=8.8, color='#52677D', linespacing=1.6)


def legend(fig, y=.88):
    fig.legend(handles=[Patch(facecolor=COLORS[v], label=LABELS[v]) for v in VARIANTS],
               loc='upper left', bbox_to_anchor=(.05, y), ncol=3, frameon=False,
               columnspacing=2.4)


def save_figure(fig, output, name, title, caption, manifest, extra=None):
    artifacts = {}
    for ext in ('png', 'pdf'):
        path = output / 'figures' / f'{name}.{ext}'
        fig.savefig(path, dpi=180, facecolor='white')
        artifacts[ext] = {'path': str(path.relative_to(output)), 'sha256': sha256(path),
                          'bytes': path.stat().st_size}
    manifest['figures'].append({'id': name, 'title': title, 'caption': caption,
                                'artifacts': artifacts, **(extra or {})})
    plt.close(fig)


def plot_training(training, output, manifest, window):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    header(fig, '01  训练过程：同等预算下的学习曲线',
           '每个策略 2.048M 环境步 · 250 次更新 · 256 个并行环境 · 每组 3 个训练种子')
    legend(fig)
    point_counts = {}
    for ax, metric in zip(axes.flat, TRAIN_METRICS):
        for variant in VARIANTS:
            # Distance is logged only when complete episodes end. Do not fabricate
            # observations on missing updates; lines merely connect the logged points.
            iterations = sorted(set.intersection(*[
                {r['iteration'] for r in training[(variant, seed)] if finite(train_value(r, metric))}
                for seed in SEEDS]))
            means, sds = [], []
            for it in iterations:
                values = [trailing(training[(variant, seed)], metric, it, window)['mean'] for seed in SEEDS]
                stats = moments(values)
                if stats['count'] != 3:
                    raise ValueError(f'Incomplete seed aggregation {variant}/{metric}/{it}')
                means.append(stats['mean']); sds.append(stats['sd'])
            xs = np.asarray(iterations) * 8192 / 1e6
            means, sds = np.asarray(means), np.asarray(sds)
            ax.fill_between(xs, means-sds, means+sds, color=COLORS[variant], alpha=.14, linewidth=0)
            ax.plot(xs, means, color=COLORS[variant], linewidth=2.1,
                    marker='o' if metric == 'distance' else None, markersize=3,
                    zorder=4 if variant == 'baseline' else 3)
            point_counts[f'{variant}/{metric}'] = len(iterations)
        name, unit = TRAIN_METRICS[metric]
        ax.set_title(name + ('（对数轴）' if metric == 'value_loss' else ''), loc='left', pad=10)
        ax.set_ylabel(unit)
        ax.set_xlabel('累计环境步数（百万）')
        ax.set_xlim(0, 2.07)
        ax.set_xticks([0, .5, 1, 1.5, 2])
        if metric == 'value_loss':
            ax.set_yscale('log')
        elif metric == 'duration':
            ax.set_ylim(9.6, 10.4)
            ax.set_yticks([9.6, 9.8, 10, 10.2, 10.4])
            ax.text(.04, .17, '三组曲线重叠：已记录回合均达到 10 s 上限',
                    transform=ax.transAxes, fontsize=9, color='#52677D')
        prettify(ax)
    footer(fig, f'各种子先取过去 {window} 个更新内已有值的均值，再计算 3 种子均值 ± 样本标准差（阴影）。位移仅在 16 个回合结束更新点有记录。\n'
           '三组训练物理分布不同；训练回报与价值损失不作为同条件性能排名。前 15 次更新尚无完整回合；缺失值不补零。')
    fig.subplots_adjust(left=.075, right=.97, top=.80, bottom=.15, hspace=.44, wspace=.25)
    save_figure(fig, output, '01_training_overview', '训练过程总览',
                f'三组各 3 个训练种子的训练曲线。每个种子先计算过去 {window} 次更新内有效日志值的均值，'
                '再计算种子均值及样本标准差（ddof=1）。回报为 RSL-RL 最近 100 个已完成回合缓冲均值；回合时长由该缓冲步数除以 50 得到；'
                '位移仅在已完成回合的更新点记录（16 点），连线不代表新增观测；价值函数损失使用对数纵轴。'
                '三组训练分布不同，跨组训练回报和损失不等价于同条件评测。没有训练过程中的瞬时速度 RMSE 记录，因此未生成此类曲线。',
                manifest, {'recorded_plot_points': point_counts})


def stat(summary, condition, variant, metric):
    entry = summary['groups'][condition][variant].get(metric)
    if entry is not None:
        return entry['mean'], entry['sd_across_training_seeds'], entry['seed_means']
    values = {str(row['train_seed']): row[metric] for row in summary['per_seed']
              if row['condition'] == condition and row['variant'] == variant}
    result = moments(values.values())
    return result['mean'], result['sd'], values


def seed_points(ax, x, seed_means, factor=1, spread=.12, size=140):
    for delta, seed in zip((-spread, 0, spread), SEEDS):
        value = seed_means[str(seed)]
        if finite(value):
            ax.scatter(x+delta, value*factor, marker=f'${seed}$', s=size,
                       color='#172C40', linewidths=.2, zorder=6)


def plot_primary(summary, output, manifest):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.6), sharey=True)
    header(fig, '02  主要指标：瞬时前向速度跟踪误差',
           '相同评测初始状态与物理参数 · 目标世界 x 速度 0.10 m/s · RMSE 越低越好')
    for ax, condition in zip(axes, CONDITIONS):
        for index, variant in enumerate(VARIANTS):
            mean, sd, seeds = stat(summary, condition, variant, 'velocity_x_rmse_m_s')
            ax.bar(index, mean, width=.62, color=COLORS[variant], alpha=.20, linewidth=0)
            ax.errorbar(index, mean, yerr=sd, color=COLORS[variant], fmt='s',
                        markersize=7, capsize=6, linewidth=2, zorder=5)
            seed_points(ax, index, seeds, spread=.16, size=155)
            ax.text(index, max(mean+sd, max(seeds.values()))+.006, f'{mean:.4f} ± {sd:.4f}',
                    ha='center', fontsize=9.3, color=COLORS[variant])
        ax.set_title(CONDITION_LABELS[condition], pad=14, loc='left')
        ax.set_xticks(range(3), [LABELS[v] for v in VARIANTS])
        ax.set_ylim(0, .127)
        ax.set_yticks([0, .02, .04, .06, .08, .10, .12])
        ax.set_ylabel('瞬时前向速度 RMSE（m/s）')
        prettify(ax)
    footer(fig, '方块与误差线：3 个训练种子的均值 ± 样本标准差；数字 42 / 43 / 44：各训练种子结果（每个点平均 128 回合）。\n'
           '共 18 个策略×条件组合、2304 回合，记录跌倒 0 次。随机条件在训练参数范围内；误差线不是置信区间。')
    fig.subplots_adjust(left=.075, right=.975, top=.80, bottom=.21, wspace=.19)
    save_figure(fig, output, '02_primary_comparison', '主要指标对照',
                '以每回合 50 Hz 世界 x 速度相对于 0.10 m/s 的 RMSE 为主要指标。每个策略/条件先平均 128 个回合，'
                '再对 3 个训练种子汇总均值和样本标准差；数值标记对应种子 42、43、44。'
                '全部 2304 回合记录跌倒 0 次，不能据此证明真实跌倒概率为零。', manifest)


def plot_ablation(summary, output, manifest):
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    header(fig, '03  延迟消融：同训练种子配对比较',
           '统一随机物理评测 · 仅比较固定中点响应/延迟与完整随机化 · RMSE 越低越好')
    _, _, fixed = stat(summary, 'randomized', 'fixed_delay', 'velocity_x_rmse_m_s')
    _, _, full = stat(summary, 'randomized', 'full_dr', 'velocity_x_rmse_m_s')
    differences = {}
    for index, seed in enumerate(SEEDS):
        a, b = fixed[str(seed)], full[str(seed)]
        differences[str(seed)] = b-a
        ax.plot([0, 1], [a, b], color='#94A3B3', linewidth=1.7,
                linestyle=('-', '--', ':')[index], zorder=2)
        for x, value, variant in ((0, a, 'fixed_delay'), (1, b, 'full_dr')):
            ax.scatter(x, value, s=420, marker='o', color='white', edgecolor=COLORS[variant], linewidth=2, zorder=3)
            ax.text(x, value, str(seed), ha='center', va='center', fontsize=10, color=COLORS[variant], zorder=4)
        ax.annotate(f'{a:.6f}', (0, a), xytext=(-22, 0), textcoords='offset points',
                    ha='right', va='center', color=COLORS['fixed_delay'], fontsize=10)
        ax.annotate(f'{b:.6f}', (1, b), xytext=(22, 0), textcoords='offset points',
                    ha='left', va='center', color=COLORS['full_dr'], fontsize=10)
    ax.text(.51, .0445, '种子 44 几乎持平\n差值 = -0.00001047 m/s', ha='center', fontsize=10,
            bbox={'facecolor': '#F1F5F9', 'edgecolor': 'none', 'pad': 6}, color='#52677D')
    ax.set_xticks([0, 1], ['固定延迟消融\nτ = 0.04 s；延迟 = 1 控制步', '完整域随机化\nτ 与延迟按训练范围随机采样'])
    ax.set_xlim(-.42, 1.42)
    ax.set_ylim(.034, .0455)
    ax.set_ylabel('瞬时前向速度 RMSE（m/s）')
    prettify(ax)
    footer(fig, '每条线连接同编号训练种子；每个端点平均 128 个配对评测回合。除响应时间常数与命令延迟外，其他训练随机化设置保持一致。\n'
           '种子 42、43 有更明显改善，种子 44 近乎持平；独立训练样本仅 n = 3，不据此宣称统计显著性。')
    fig.subplots_adjust(left=.12, right=.94, top=.80, bottom=.25)
    save_figure(fig, output, '03_delay_ablation', '延迟随机化配对消融',
                '随机条件下，同编号训练种子从固定延迟消融到完整域随机化的主要指标变化。'
                '种子 42、43 改善，种子 44 的完整随机化仅降低约 0.00001047 m/s，近乎持平。'
                '配对展示训练种子差异，不代表统计显著性检验。', manifest,
                {'full_dr_minus_fixed_delay_m_s': differences})


def grouped_panels(summary, metrics, title, subtitle, footer_text, name, caption, output, manifest):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.1))
    header(fig, title, subtitle)
    legend(fig)
    for ax, metric in zip(axes.flat, metrics):
        factor = display_factor(metric)
        maxima, minima = [], []
        for ci, condition in enumerate(CONDITIONS):
            for vi, variant in enumerate(VARIANTS):
                x = ci + (vi-1)*.23
                mean, sd, seeds = stat(summary, condition, variant, metric)
                ax.bar(x, mean*factor, width=.185, color=COLORS[variant], alpha=.8,
                       yerr=sd*factor, capsize=3,
                       error_kw={'ecolor': '#263B52', 'elinewidth': 1.2})
                maxima.append((mean+sd)*factor)
                minima.append((mean-sd)*factor)
        label, unit = METRICS[metric]
        ax.set_title(label, loc='left', pad=10)
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], [CONDITION_LABELS[c] for c in CONDITIONS])
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(min(0, min(minima)*1.15), max(maxima)*1.16)
        if metric == 'distance_x_m':
            ax.axhline(1.0, color='#71849A', linestyle=':', linewidth=1.1)
            ax.text(.98, 1.0, '  0.10 m/s × 10 s = 1 m', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=8.8, color='#52677D')
        prettify(ax)
    footer(fig, footer_text)
    fig.subplots_adjust(left=.075, right=.97, top=.80, bottom=.15, hspace=.4, wspace=.28)
    save_figure(fig, output, name, title[4:], caption, manifest)


def write_csv(path, fields, rows, output, manifest, caption):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    manifest['tables'].append({'path': str(path.relative_to(output)), 'sha256': sha256(path),
                               'rows': len(rows), 'encoding': 'UTF-8 with BOM', 'caption': caption})


def write_tables(summary, training, output, manifest, window):
    rows = []
    for condition in CONDITIONS:
        for variant in VARIANTS:
            for metric, (label, unit) in METRICS.items():
                mean, sd, seeds = stat(summary, condition, variant, metric)
                factor = display_factor(metric)
                rows.append({'评测条件': CONDITION_LABELS[condition], '策略组': LABELS[variant],
                             '指标': label, '单位': unit, '均值': mean*factor,
                             '样本标准差': sd*factor, '均值±样本标准差': f'{mean*factor:.6g} ± {sd*factor:.6g}',
                             '有效训练种子数': len([v for v in seeds.values() if finite(v)]),
                             '每种子评测回合数': 128, '指标键': metric})
    write_csv(output/'tables/01_evaluation_summary_zh.csv', list(rows[0]), rows, output, manifest,
              '每个条件、策略组和指标的种子均值 ± 样本标准差；比例转换为百分数。')
    rows = []
    for row in summary['per_seed']:
        rows.append({'策略组': LABELS[row['variant']], '训练种子': row['train_seed'],
                     '评测条件': CONDITION_LABELS[row['condition']], '瞬时前向速度RMSE_m每s': row['velocity_x_rmse_m_s'],
                     '瞬时平面速度RMSE_m每s': row['velocity_xy_tracking_rmse_m_s'],
                     '有效回合数': row['velocity_x_rmse_m_s_valid_episodes'], '总回合数': row['episodes'],
                     '跌倒比例': row['fall_rate']})
    write_csv(output/'tables/02_primary_per_seed_zh.csv', list(rows[0]), rows, output, manifest,
              '各训练种子的主要指标及平面速度 RMSE；每个结果平均 128 个评测回合。')
    milestone_rows, detail_rows = [], []
    for iteration in (64, 125, 188, 250):
        for variant in VARIANTS:
            for metric, (label, unit) in TRAIN_METRICS.items():
                windows = {seed: trailing(training[(variant, seed)], metric, iteration, window) for seed in SEEDS}
                stats = moments(v['mean'] for v in windows.values())
                milestone_rows.append({'更新次数': iteration, '实际累计环境步数': iteration*8192,
                                       '策略组': LABELS[variant], '指标': label, '单位': unit,
                                       '过去10更新均值': stats['mean'], '跨种子样本标准差': stats['sd'],
                                       '均值±样本标准差': f"{stats['mean']:.6g} ± {stats['sd']:.6g}",
                                       '有效训练种子数': stats['count'], '窗口起始更新': iteration-window+1,
                                       '窗口结束更新': iteration,
                                       '各种子有效记录数': '; '.join(f'{s}:{windows[s]["n_records"]}' for s in SEEDS)})
                for seed, record in windows.items():
                    detail_rows.append({'更新次数': iteration, '实际累计环境步数': iteration*8192,
                                        '策略组': LABELS[variant], '训练种子': seed, '指标': label, '单位': unit,
                                        '过去10更新均值': record['mean'], '有效记录数': record['n_records'],
                                        '原始记录更新编号': ';'.join(map(str, record['source_iterations']))})
    write_csv(output/'tables/03_training_milestones_zh.csv', list(milestone_rows[0]), milestone_rows, output, manifest,
              '更新 64/125/188/250 对应的实际环境步数；每种子先取过去 10 个更新内有效值，再对种子聚合。')
    write_csv(output/'tables/04_training_milestones_per_seed_zh.csv', list(detail_rows[0]), detail_rows, output, manifest,
              '训练里程碑逐种子明细，记录构成窗口均值的原始更新编号；缺失值未插值或补零。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT/'runs/robustness_ablation_20261103')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/ppo_study_summary_20261103')
    args = parser.parse_args()
    study, output, sources = args.study_dir.resolve(), args.output_dir.resolve(), {}
    if output == study or study in output.parents:
        raise ValueError('Report output must be outside frozen study artifacts')
    protocol = read_json(study/'protocol.json', sources)
    summary = read_json(study/'analysis/summary.json', sources)
    result = read_json(study/'evaluation/result.json', sources)
    if result['status'] != 'completed' or result['cases_completed'] != 18:
        raise ValueError('The formal evaluation must be complete')
    if summary['protocol_sha256'] != sha256(study/'protocol.json'):
        raise ValueError('Analysis/protocol hash mismatch')
    for path, expected in summary['inputs_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen analysis input hash mismatch: {path}')
    training = {}
    for checkpoint in protocol['checkpoints']:
        rows = read_jsonl(Path(checkpoint['run_dir'])/'training.jsonl', sources)
        if [row['iteration'] for row in rows] != list(range(1, 251)):
            raise ValueError(f'Unexpected training iterations: {checkpoint["id"]}')
        if any(row['environment_transitions'] != row['iteration']*8192 for row in rows):
            raise ValueError('Training transition count mismatch')
        training[(checkpoint['variant'], checkpoint['train_seed'])] = rows
    if set(training) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Expected all 9 formal training logs')
    if len(summary['per_seed']) != 18 or any(r['episodes'] != 128 for r in summary['per_seed']):
        raise ValueError('Expected 18 complete policy/condition summaries')
    if any(row['fall_rate'] != 0 for row in summary['per_seed']):
        raise ValueError('Caption fall count needs updating')
    for folder in ('figures', 'tables'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    configure_plotting()
    sources[str(Path(__file__).resolve())] = {'sha256': sha256(__file__)}
    sources[str(FONT)] = {'sha256': sha256(FONT)}
    manifest = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'title': 'Sesame PPO 三组实验：训练曲线与正式评测',
        'study_dir': str(study), 'output_dir': str(output),
        'protocol_sha256': summary['protocol_sha256'], 'sourcehashes': sources,
        'verified_upstream_analysis_inputs_sha256': summary['inputs_sha256'],
        'counts': {'training_runs': 9, 'training_seeds_per_group': 3, 'updates_per_run': 250,
                   'environment_transitions_per_run': 2048000, 'evaluation_cases': 18,
                   'episodes_per_case': 128, 'total_evaluation_episodes': 2304, 'observed_falls': 0},
        'preprocessing': {
            'training_smoothing': '先在每个种子内，对当前及之前9个更新中的有效值求均值；再对3个种子求均值±样本SD（ddof=1）。',
            'training_missing_values': '不补零、不插值；位移仅在有原始日志的16个更新点显示，连线仅为阅读引导。',
            'training_return': 'RSL-RL记录的最近100个已完成回合缓冲均值，再作10更新尾随平滑。',
            'training_duration': 'mean_episode_length_steps / 50 Hz，随后作10更新尾随平滑。',
            'training_value_loss': 'losses.value_function，尾随均值后以对数纵轴呈现。',
            'evaluation': '每个策略/条件先平均128个回合，再以3个训练种子为独立单位，计算均值和样本SD；不是回合SEM或置信区间。',
            'milestone_iterations': [64, 125, 188, 250],
            'milestone_transitions': [524288, 1024000, 1540096, 2048000],
            'training_velocity_rmse_curve': '未生成：已保存训练日志不含该指标。',
        },
        'style': {'colors': COLORS, 'variant_labels': LABELS, 'font': str(FONT),
                  'png_dpi': 180, 'pdf': 'vector plot elements with embedded Type3 CJK glyph outlines',
                  'seed_markers': {str(s): str(s) for s in SEEDS}},
        'limitations': summary['limitations'], 'figures': [], 'tables': [],
    }
    plot_training(training, output, manifest, 10)
    plot_primary(summary, output, manifest)
    plot_ablation(summary, output, manifest)
    grouped_panels(summary,
                   ['distance_x_m', 'abs_final_heading_yaw_deg', 'abs_lateral_displacement_m', 'return'],
                   '04  运动表现：位移、方向与本地回报',
                   '两个共同评测条件 · 柱高为 3 种子均值 · 误差线为种子间样本标准差',
                   '更远的位移不一定更接近 0.10 m/s 目标；航向与横向位移均为相对于世界坐标的末端绝对值。\n'
                   '方向相关奖励未修改，不能把速度跟踪改善等同于完全消除侧行。本地回报不代表官方课程分数；共 2304 回合记录跌倒 0 次。',
                   '04_motion_quality',
                   '世界 x 位移、末端航向角绝对值、末端横向位移绝对值与未修改的本地奖励回报。'
                   '每个条件/策略组报告 3 个训练种子均值±样本标准差。位移中的 1 m 参考线为目标速度乘以 10 s，位移越大不必然越好。'
                   '航向未增加纠正奖励，不能宣称已消除侧行；本地回报不是官方课程评分。', output, manifest)
    grouped_panels(summary,
                   ['applied_target_rate_rms_rad_s', 'applied_target_acceleration_rms_rad_s2',
                    'actual_joint_acceleration_rms_rad_s2', 'raw_action_clipped_fraction'],
                   '05  平滑性诊断：控制目标与关节响应',
                   '全部差分指标按 50 Hz 控制频率统计 · 均值 ± 3 种子样本标准差',
                   '目标角变化描述施加命令；实际关节角加速度由关节速度在 50 Hz 采样下差分，不能代表 500 Hz 物理峰值、功率或“抖动频率”。\n'
                   '原始动作裁剪比例在包装器限幅前统计；平滑指标需结合跟踪误差解读。全部 2304 回合记录跌倒 0 次。',
                   '05_smoothness',
                   '施加目标角的一阶/二阶差分 RMS、实际关节速度在 50 Hz 控制采样下的一阶差分 RMS，以及原始动作限幅比例。'
                   '这些指标不等同于抖动频率、机械功率或 500 Hz 物理加速度峰值。原始动作比例在包装器裁剪前计算，图中转换为百分数。'
                   '各组误差线为 3 个训练种子均值的样本标准差。', output, manifest)
    write_tables(summary, training, output, manifest, 10)
    path = output/'figures_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'manifest': str(path), 'figures': len(manifest['figures']),
                      'figure_formats': ['png', 'pdf'], 'tables': len(manifest['tables'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
