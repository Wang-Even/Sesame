"""Paper-style figures from recorded baseline/motion scalars only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from matplotlib.ticker import NullLocator, PercentFormatter

from course_paper_plot_utils import (
    ROOT, SOURCE, COLORS, HALF_LIFE_STEPS, plt, np, panel, ema, draw_curve,
    curve_records as make_curve_records,
)


def _csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _categories(ax):
    ax.set_xticks([0, 1], ['Fixed\nparameters', 'Domain\nrandomization'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-.38, 1.38)


def _change_inset(ax, original, smooth):
    """Show relative change alongside the unchanged absolute measurements."""
    if np.any(original == 0):
        raise ValueError('Relative-change inset requires nonzero original measurements')
    change = 100. * (smooth - original) / original
    limit = max(1., float(np.ceil(np.max(np.abs(change)) * 1.2)))
    detail = ax.inset_axes([.60, .72, .36, .24])
    detail.set_title('Change (%)', fontsize=10.5, pad=3)
    detail.axhline(0., color='#888888', lw=.65, ls='-', zorder=1)
    detail.plot([0., 1.], change, color=COLORS['smooth'], lw=1.35,
                ls='-', marker=None, zorder=3)
    detail.set_xlim(-.12, 1.12)
    detail.set_ylim(-limit, limit)
    detail.set_xticks([0., 1.], ['Fixed', 'DR'])
    detail.set_yticks([-limit, 0., limit])
    detail.xaxis.set_minor_locator(NullLocator())
    detail.yaxis.set_minor_locator(NullLocator())
    detail.tick_params(labelsize=9.5, length=2.5, width=.6, pad=2,
                       color='#777777')
    detail.grid(False)
    detail._curve_detail = dict(kind='relative_change',
                               formula='100 * (smooth - original) / original',
                               original=original.tolist(), smooth=smooth.tolist(),
                               change_percent=change.tolist(),
                               conditions=['nominal', 'randomized'])


def _startup_record(path, method, seed):
    """Use the first logged confirmation of a completed optimizer update.

    Off-policy logs are periodic: the first nonzero counter can already include
    several updates. Its timestamp is an observed boundary, not an estimate of
    the exact first optimizer-step completion time.
    """
    elapsed = 0.
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if 'training_wall_seconds' in row:
            elapsed = float(row['training_wall_seconds'])
            updated = row['optimizer_updates'] > 0
            rule = 'First periodic log with optimizer_updates > 0; exact first-update time is not recorded.'
        else:
            elapsed += row['collection_seconds'] + row['update_seconds']
            updated = bool(row['losses'])
            rule = 'Cumulative collection_seconds + update_seconds at the first completed PPO update.'
        if updated:
            return dict(method=method, seed=seed, startup_s=elapsed,
                        source=str(path.relative_to(ROOT)), rule=rule,
                        logged_real_steps=int(row['environment_transitions']),
                        logged_optimizer_updates=row.get('optimizer_updates', row.get('iteration')))
    raise ValueError(f'No completed update in {path}')


def draw_all(save, *, omit_high_speed=False, training_uncertainty=True,
             training_time_comparison=True):
    """Draw selected baseline figures and retain archived scalar provenance.

    Reports without the high-speed study omit both its comparison and the
    three-policy supplementary training figure. Their archived scalar rows
    remain available in the display-data export.
    """
    curve_file = SOURCE / 'baselines/baseline_curve_data.csv'
    test_file = SOURCE / 'baselines/baseline_test_results.csv'
    motion_file = SOURCE / 'improvements/motion_results.csv'
    sources = [curve_file, test_file, motion_file]
    records = []
    curves, tests, motion = _csv(curve_file), _csv(test_file), _csv(motion_file)
    methods, seeds = ['dqn', 'ppo', 'sac'], [42, 43, 44]
    origins, startup_audit = {}, []
    for method in methods:
        for seed in seeds:
            path = (ROOT / 'runs/ppo_heading_20261105/training' /
                    f'heading_seed{seed}/training.jsonl' if method == 'ppo' else
                    ROOT / 'runs/offpolicy_heading_20261107/training' /
                    f'{method}_seed{seed}/training.jsonl')
            sources.append(path)
            record = _startup_record(path, method, seed)
            origins[(method, seed)] = record['startup_s']
            startup_audit.append(record)
    metrics = [
        ('mean_episode_return', 'Return', 'Training return'),
        ('velocity_x_rmse_m_s', 'Velocity error', 'Velocity RMSE (m/s)'),
        ('heading_rms_rad', 'Heading error', 'Heading RMS (deg)'),
    ]
    training_rows = 2 if training_time_comparison else 1
    fig, axes = plt.subplots(training_rows, 3,
                             figsize=(11, 6.4 if training_time_comparison else 3.3),
                             layout='constrained', squeeze=False)
    for column, (metric, title, ylabel) in enumerate(metrics):
        panel(axes[0, column], 'abc'[column], title,
              'Environment steps (million)', ylabel)
        if training_time_comparison:
            panel(axes[1, column], 'def'[column], title, 'Training time after startup (min)', ylabel)
        for method in methods:
            by_seed, time_by_seed = {}, {}
            for seed in seeds:
                selected = sorted(
                    (r for r in curves if r['variant'] == method and
                     int(r['train_seed']) == seed and r['metric'] == metric),
                    key=lambda r: int(r['environment_transitions']))
                steps = np.array([int(r['environment_transitions']) for r in selected])
                seconds = np.array([float(r['logged_training_s']) for r in selected])
                values = np.array([float(r['value']) for r in selected])
                if len(set(steps)) != len(steps):
                    raise ValueError('Duplicate baseline method/seed/metric/step')
                # This archived CSV has ALREADY converted heading_rms_rad to degrees.
                if metric == 'heading_rms_rad':
                    assert all(np.isclose(float(r['display_scale']), 180 / np.pi) for r in selected)
                smooth = ema(steps, values)
                startup_s = origins[(method, seed)]
                keep = seconds >= startup_s
                by_seed[seed] = dict(zip(steps[keep].tolist(), smooth[keep].tolist()))
                time_by_seed[seed] = dict(zip(steps[keep].tolist(),
                                            (seconds[keep] - startup_s).tolist()))
                records.extend(make_curve_records(
                    'historical_baseline_training', method, seed,
                    'heading_rms_deg' if metric == 'heading_rms_rad' else metric,
                    steps, seconds, values, smooth, startup_s=startup_s))
            shared = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
            if not shared:
                raise ValueError('No common training steps across three seeds')
            values = np.array([[by_seed[s][step] for step in shared] for s in seeds])
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            x = np.asarray(shared) / 1e6
            if training_uncertainty:
                axes[0, column].fill_between(x, mean - std, mean + std,
                                             color=COLORS[method], alpha=.13, lw=0)
            axes[0, column].plot(x, mean, color=COLORS[method], lw=1.8,
                                label=method.upper(), zorder=3)
            if training_time_comparison:
                mean_seconds = np.array([[time_by_seed[seed][step] for step in shared]
                                         for seed in seeds]).mean(axis=0)
                if training_uncertainty:
                    axes[1, column].fill_between(mean_seconds / 60, mean - std, mean + std,
                                                 color=COLORS[method], alpha=.13, lw=0)
                axes[1, column].plot(mean_seconds / 60, mean, color=COLORS[method],
                                    lw=1.8, ls='-', marker=None,
                                    label=method.upper(), zorder=3)
        for row in range(training_rows):
            axes[row, column].set_ylim(bottom=0)
            axes[row, column].set_xlim(left=0)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    if training_time_comparison:
        axes[1, 0].legend(loc='lower right', fontsize=10)
    save(fig, 'baselines/baseline_learning_curves')

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout='constrained')
    for index, (metric, title, ylabel) in enumerate([
        ('return', 'Return', 'Test return'),
        ('velocity_x_rmse_m_s', 'Velocity error', 'Velocity RMSE (m/s)'),
        ('heading_rms_deg', 'Heading error', 'Heading RMS (deg)'),
    ]):
        ax = axes[index]
        panel(ax, 'abc'[index], title, ylabel=ylabel)
        for j, method in enumerate(methods):
            selected = [next(r for r in tests if r['method'] == method and r['condition'] == c)
                        for c in ['nominal', 'randomized']]
            xs = np.array([0., 1.])
            mean = np.array([float(r[metric + '_mean']) for r in selected])
            std = np.array([float(r[metric + '_sd']) for r in selected])
            ax.fill_between(xs, mean - std, mean + std,
                            color=COLORS[method], alpha=.13, lw=0)
            ax.plot(xs, mean, color=COLORS[method], lw=1.8, ls='-',
                    marker=None, label=method.upper(), zorder=3)
        _categories(ax)
        maximum = max(float(r[metric + '_mean']) + float(r[metric + '_sd']) for r in tests)
        ax.set_ylim(0, maximum * 1.12)
    axes[0].legend(loc='lower left', fontsize=10)
    save(fig, 'baselines/baseline_test_metrics')

    for speed, models, stem, specs in [
        (.10, ['baseline', 'smooth'], 'smoothness_comparison', [
            ('target_acceleration_rms_rad_s2', 'Target acceleration', r'Target acceleration RMS (rad/s$^2$)'),
            ('actual_joint_acceleration_rms_rad_s2', 'Joint acceleration', r'Joint acceleration RMS (rad/s$^2$)'),
            ('velocity_rmse_m_s', 'Velocity error', 'Velocity RMSE (m/s)'),
            ('heading_rms_deg', 'Heading error', 'Heading RMS (deg)'),
        ]),
        (.20, ['baseline', 'fast'], 'speed_comparison', [
            ('mean_forward_speed_m_s', 'Forward speed', 'Effective forward speed (m/s)'),
            ('velocity_rmse_m_s', 'Velocity error', 'Velocity RMSE (m/s)'),
            ('heading_rms_deg', 'Heading error', 'Heading RMS (deg)'),
            ('true_termination_rate', 'Early termination', 'Early termination'),
        ]),
    ]:
        if omit_high_speed and stem == 'speed_comparison':
            continue
        labels = {'baseline': 'Original PPO' if speed == .10 else 'PPO trained at 0.10 m/s',
                  'smooth': 'Action penalty doubled', 'fast': 'PPO trained at 0.20 m/s'}
        is_smoothness = stem == 'smoothness_comparison'
        if is_smoothness:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.1))
            fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.86,
                                wspace=.30, hspace=.50)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), layout='constrained')
        for i, (metric, title, ylabel) in enumerate(specs):
            ax = axes.flat[i]
            panel(ax, 'abcd'[i], title, ylabel=ylabel)
            absolute_values = {}
            for j, model in enumerate(models):
                selected = [next(r for r in motion if r['model'] == model and
                                 r['condition'] == c and float(r['target_speed_m_s']) == speed)
                            for c in ['nominal', 'randomized']]
                xs = np.array([0., 1.])
                values = np.array([float(r[metric]) for r in selected])
                absolute_values[model] = values
                linewidth = (2.4 if model == 'baseline' else 1.5) if is_smoothness else 2.2
                ax.plot(xs, values, color=COLORS[model], lw=linewidth, ls='-',
                        marker=None, label=labels[model], zorder=4 + j, clip_on=False)
            _categories(ax)
            maximum = max(float(r[metric]) for r in motion
                          if r['model'] in models and float(r['target_speed_m_s']) == speed)
            ax.set_ylim(0, max(maximum * 1.15, .01))
            if is_smoothness and i < 3:
                _change_inset(ax, absolute_values['baseline'], absolute_values['smooth'])
            if metric == 'mean_forward_speed_m_s':
                ax.axhline(.20, color='#777777', lw=1.0, ls='-',
                           label='Command 0.20 m/s', zorder=1)
            if metric == 'true_termination_rate':
                ax.set_ylim(-.06, 1.)
                ax.set_yticks([0, .25, .50, .75, 1.])
                ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
        if is_smoothness:
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='upper center',
                       bbox_to_anchor=(.5, .995), ncol=2, fontsize=11)
        else:
            axes[0, 0].legend(loc='best', fontsize=10)
        save(fig, f'improvements/{stem}')

    if omit_high_speed:
        fig, axes = None, None
    else:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout='constrained')
    titles = ['Original PPO', r'Action penalty $\times 2$', 'PPO at 0.20 m/s']
    for i, model in enumerate(['baseline', 'smooth', 'fast']):
        path = ROOT / 'runs/ppo_motion_study_20261115/training' / f'{model}_seed42' / 'training.jsonl'
        sources.append(path)
        record = _startup_record(path, model, 42)
        startup_audit.append(record)
        startup_s = record['startup_s']
        elapsed, points = 0., []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            elapsed += row['collection_seconds'] + row['update_seconds']
            if row['mean_episode_return'] is not None:
                points.append((row['environment_transitions'], elapsed, row['mean_episode_return']))
        steps, seconds, values = map(np.asarray, zip(*points))
        if axes is None:
            smooth = ema(steps, values)
        else:
            ax = axes[i]
            panel(ax, 'abc'[i], titles[i], 'Training time after startup (min)', 'Own training return')
            smooth, _ = draw_curve(ax, steps, (seconds - startup_s) / 60,
                                   values, COLORS[model], start_x=0.)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        records.extend(make_curve_records('motion_training_own_reward', model, 42,
                                           'mean_episode_return', steps, seconds, values, smooth,
                                           startup_s=startup_s))
    if fig is not None:
        save(fig, 'improvements/motion_training_curves')

    captions = {
        '2': f'三种基线的训练曲线。上、下排横轴分别为绝对真实环境步数和扣除启动阶段后的训练时间，纵轴依次为训练回报、前向速度 RMSE 与航向 RMS。每个种子以首次记录到参数更新完成的日志时刻为时间零点，仅绘制此后的已有记录，不补造零点回报。SAC/DQN 采用首次 optimizer_updates 非零的周期日志，其时刻不等同于精确的第一次优化完成时刻。因果 EMA 在完整记录上计算，半衰期为 {HALF_LIFE_STEPS:,} 步，再裁剪显示区间；三个训练种子在共同保留步数处计算均值与样本标准差，下排横轴取各自扣除启动时间后的均值。连续实线连接已有 EMA 值。PPO 误差指标共 16 个诊断时刻；回报窗口为 PPO 最近 100 回合、SAC/DQN 最近 256 回合。最终测试统计不作平滑。',
        '3': '三种基线的共同测试结果。固定参数与域随机化评估设置见第 2.1 节，训练均采用域随机化。连续实线连接两种评估条件下三个训练种子的测试均值，阴影表示样本标准差；每种方法、每种条件共 384 回合。横轴为分类条件，连线用于比较。数值为未经平滑的原始测试统计。',
        '4': '动作差惩罚加倍与原设置 PPO 的平滑性对照。(a) 执行目标角加速度 RMS；(b) 实际关节加速度 RMS；(c) 前向速度 RMSE；(d) 航向 RMS。固定参数与域随机化评估设置见第 2.1 节。连续实线连接两种评估条件下单个训练种子、128 回合的原始测试均值，无跨种子误差带。横轴为分类条件，连线用于比较；平滑性结合速度误差评估。',
        '5': '0.20 m/s 指令下高速训练与低速策略直接迁移的对照。(a) 有效前进速度，灰色实线为目标；(b) 速度 RMSE；(c) 航向 RMS；(d) 提前终止比例，包括跌倒、越界和无效状态。固定参数与域随机化评估设置见第 2.1 节。连续实线连接两种评估条件下单个训练种子、128 回合的原始测试均值，无跨种子误差带。横轴为分类条件，连线用于比较；提前终止比例均为零，两组曲线重合。',
    }
    return {'captions': captions, 'sources': sources, 'curve_records': records,
            'startup_audit': startup_audit}
