#!/usr/bin/env python3
"""Paper-style figures from completed Sesame logs; no model computation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


BLUE, ORANGE, GRAY = '#1054ad', '#c65520', '#858585'
METHODS = ('supermimic', 'ppo')
LABELS = {'supermimic': 'SuperMimic (small)', 'ppo': 'PPO'}
COLORS = {'supermimic': BLUE, 'ppo': ORANGE}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def panel(ax, letter, title, xlabel, ylabel):
    ax.set_title(r'$\mathbf{(' + letter + r')}$ ' + title, loc='left', pad=13)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(which='major', length=5, width=.8, color=GRAY)
    ax.tick_params(which='minor', length=3, width=.6, color=GRAY)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#626262')
    ax.grid(False)


def curve(ax, algorithm, x, y, **kwargs):
    return ax.plot(x, y, color=COLORS[algorithm], label=LABELS[algorithm],
                   linestyle='-' if algorithm == 'supermimic' else '--',
                   marker='o' if algorithm == 'supermimic' else 's',
                   markersize=3.4, linewidth=2.0, **kwargs)


def log_millions(ax, ticks, limits):
    ax.set_xscale('log')
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else study / 'paper_figures_20261111'
    output.mkdir(parents=True, exist_ok=True)
    spec = read(study / 'protocol.json')
    assert read(study / 'analysis/audit.json')['passed']
    assert not spec['smoke']
    # Positive tracking terms are at most one; all other reward terms are <= 0.
    reward_ceiling = spec['success']['episode_seconds'] * sum(
        weight for weight in spec['reward_weights'].values() if weight > 0)
    assert math.isclose(reward_ceiling, 33.75)
    data, source_hashes = {}, {'protocol.json': sha(study / 'protocol.json')}
    validation, throughput, timing, final_metrics = [], [], [], []
    transition_batch = spec['num_envs'] * spec['steps_per_collection']
    for algorithm in METHODS:
        run = study / f'{algorithm}_seed42'
        result = read(run / 'result.json')
        assert result['status'] == 'completed' and result['seed'] == 42
        training = lines(run / 'training.jsonl')
        evaluations = [r for r in lines(run / 'evaluations.jsonl') if r['label'] == 'validation']
        test = read(run / 'test_result.json')
        for name in ('result.json', 'training.jsonl', 'evaluations.jsonl', 'test_result.json'):
            source_hashes[f'{algorithm}_seed42/{name}'] = sha(run / name)
        for row in evaluations:
            validation.append({'algorithm': algorithm, 'update': row['update'],
                'real_transitions': row['cumulative_real_steps'], 'imagined_transitions': row['imagined_steps'],
                'training_hours': row['training_seconds'] / 3600, 'raw_return': row['mean_return'],
                'normalized_return': row['mean_return'] / reward_ceiling, 'success_rate': row['success_rate'],
                'stage': row.get('stage', 'policy_evaluation')})
        # Fixed, nonoverlapping blocks. Ratio of totals, not average of rates.
        for start in range(0, len(training), 12):
            block = training[start:start + 12]
            seconds = sum(r['collection_seconds'] for r in block)
            throughput.append({'algorithm': algorithm, 'first_update': block[0]['update'],
                'last_update': block[-1]['update'], 'updates_in_block': len(block),
                'real_transitions': block[-1]['cumulative_real_steps'],
                'collected_transitions': len(block) * transition_batch,
                'collection_seconds': seconds, 'steps_per_second': len(block) * transition_batch / seconds})
        components = {'offline_pretraining_seconds': result.get('offline_pretraining_seconds', 0),
                      'collection_seconds': sum(r['collection_seconds'] for r in training),
                      'optimization_seconds': sum(r['update_seconds'] for r in training)}
        assert math.isclose(sum(components.values()), result['training_seconds'], abs_tol=1e-5)
        timing.append({'algorithm': algorithm, **components, 'training_seconds': result['training_seconds'],
                       'evaluation_seconds': result['evaluation_seconds'],
                       'program_elapsed_seconds': result['wall_seconds_before_shutdown']})
        final_metrics.append({'algorithm': algorithm, 'episodes': test['episodes'],
            'checkpoint': test['selected_checkpoint'],
            'checkpoint_real_transitions': test['selected_checkpoint_cumulative_steps'],
            **{k: v for k, v in test.items() if k.endswith(('_m_s', '_deg', '_rad', '_m', '_rad_s2', '_rate'))}})
        data[algorithm] = {'result': result, 'validation': [v for v in validation if v['algorithm'] == algorithm],
                           'test': test}
    world_path = study / 'supermimic_seed42/world_validation.jsonl'
    source_hashes['supermimic_seed42/world_validation.jsonl'] = sha(world_path)
    world = [{**row, 'real_transitions': row['online_steps'] + spec['supermimic']['offline_supplied_transitions']}
             for row in lines(world_path)]
    for filename, rows in [('validation.csv', validation), ('collection_throughput.csv', throughput),
                           ('training_time.csv', timing), ('world_prediction.csv', world),
                           ('test_metrics.csv', final_metrics)]:
        write_csv(output / filename, rows)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 19,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.facecolor': 'white'})
    saved = []
    with PdfPages(output / 'all_figures.pdf') as book:
        def save(fig, name, footer):
            fig.text(.07, .035, footer, ha='left', va='bottom', fontsize=10.5, color='#444444')
            fig.savefig(output / f'{name}.png', facecolor='white')
            fig.savefig(output / f'{name}.pdf', facecolor='white')
            book.savefig(fig)
            saved.append(name)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
        fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.24, wspace=.34)
        panel(axes[0], 'a', 'Normalized return', 'Training time (h)', 'Return / 33.75')
        panel(axes[1], 'b', 'Normalized return', 'Real transitions (millions)', 'Return / 33.75')
        panel(axes[2], 'c', 'Validation success', 'Real transitions (millions)', 'Success rate')
        for algorithm in METHODS:
            rows = data[algorithm]['validation']
            curve(axes[0], algorithm, [r['training_hours'] for r in rows], [r['normalized_return'] for r in rows])
            positive = [r for r in rows if r['real_transitions'] > 0]
            curve(axes[1], algorithm, [r['real_transitions']/1e6 for r in positive], [r['normalized_return'] for r in positive])
            curve(axes[2], algorithm, [r['real_transitions']/1e6 for r in rows], [r['success_rate'] for r in rows])
        for ax in axes[:2]:
            ax.set_ylim(0, .78)
            ax.set_yticks([0, .2, .4, .6])
        axes[0].set_xlim(0, 1.02)
        axes[0].legend(frameon=False, loc='lower right')
        baseline = validation[0]['normalized_return']
        axes[1].axhline(baseline, color=GRAY, linewidth=.9, linestyle=(0, (5, 4)))
        axes[1].text(.21, baseline - .045, 'Initial policy', fontsize=11, color=GRAY)
        log_millions(axes[1], [.2, .5, 1, 2], (.17, 2.3))
        axes[2].set_ylim(-.025, 1.04)
        axes[2].set_xlim(0, 2.13)
        axes[2].axhline(.8, color=GRAY, linewidth=1, linestyle=(0, (5, 4)))
        axes[2].text(.08, .83, 'Target: 80% twice in a row', fontsize=11, color=GRAY)
        axes[2].text(.08, .40, 'Target not reached\nBoth methods: 0 / 64', fontsize=13)
        save(fig, 'figure_1_learning', 'Sesame reference tracking; seed 42; 64 validation trajectories. No smoothing or multi-seed confidence band.\nTraining time excludes evaluation; shared GPU. SuperMimic real-data counts include 300,847 offline transitions.')

        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
        fig.subplots_adjust(left=.085, right=.98, top=.81, bottom=.25, wspace=.32)
        panel(axes[0], 'a', 'Return vs. imagined steps', 'Imagined transitions (millions)', 'Normalized validation return')
        rows = [r for r in data['supermimic']['validation'] if r['imagined_transitions'] > 0]
        curve(axes[0], 'supermimic', [r['imagined_transitions']/1e6 for r in rows], [r['normalized_return'] for r in rows])
        log_millions(axes[0], [.4, 1, 2, 4], (.35, 4.15))
        axes[0].set_ylim(0, .78)
        axes[0].legend(frameon=False, loc='lower left')
        panel(axes[1], 'b', '8-step prediction error', 'Real transitions (millions)', 'Normalized state RMSE')
        axes[1].plot([r['real_transitions']/1e6 for r in world], [r['normalized_rmse'] for r in world],
                     '-o', color=BLUE, linewidth=2, markersize=5)
        axes[1].set_ylim(0, max(r['normalized_rmse'] for r in world)*1.15)
        save(fig, 'figure_2_world_model', 'SuperMimic only; PPO has no imagined transitions. Prediction error uses held-out offline validation sequences.\nError is averaged over an 8-step rollout with Sesame state scaling; it is not the paper\'s G1 metric.')

        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
        fig.subplots_adjust(left=.085, right=.98, top=.80, bottom=.25, wspace=.40)
        panel(axes[0], 'a', 'Simulation collection speed', 'Real transitions (millions)', 'Collected transitions / second')
        for algorithm in METHODS:
            rows = [r for r in throughput if r['algorithm'] == algorithm]
            curve(axes[0], algorithm, [r['real_transitions']/1e6 for r in rows], [r['steps_per_second'] for r in rows])
        axes[0].set_ylim(bottom=0)
        axes[0].legend(frameon=False, loc='lower left')
        panel(axes[1], 'b', 'Training time', 'Training time (h)', '')
        stages = [('offline_pretraining_seconds', 'Offline pretraining', '#9db7d7'),
                  ('collection_seconds', 'Simulation collection', BLUE),
                  ('optimization_seconds', 'Online optimization', ORANGE)]
        for i, row in enumerate(timing):
            start = 0
            for key, label, color in stages:
                width = row[key]/3600
                axes[1].barh(i, width, left=start, height=.38, color=color,
                              label=label if i == 0 else None)
                start += width
            axes[1].text(start + .025, i, f'{start:.3f} h', va='center', fontsize=12)
        axes[1].set_yticks([0, 1], [LABELS[m] for m in METHODS])
        axes[1].set_xlim(0, 1.12)
        axes[1].set_ylim(-.60, 2.0)
        axes[1].invert_yaxis()
        axes[1].legend(frameon=False, loc='lower left', fontsize=11)
        save(fig, 'figure_3_efficiency', '512 Isaac environments; collection speed = transitions / collection time in nonoverlapping 12-update blocks.\nShared GPU load changed during the runs. These are observed timings, not controlled algorithm-speed benchmarks.')

        metrics = [('velocity_rmse_m_s', 'Planar velocity', 'RMSE (m/s)', .05),
                   ('heading_rms_deg', 'Heading', 'RMS error (deg)', 10.),
                   ('root_position_rmse_m', 'Root position', 'RMSE (m)', .10),
                   ('joint_position_rmse_rad', 'Joint position', 'RMSE (rad)', .25),
                   ('foot_position_rmse_m', 'Foot position', 'RMSE (m)', .02),
                   ('joint_acceleration_rms_rad_s2', 'Joint acceleration', r'RMS (rad/s$^2$)', None)]
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.9))
        fig.subplots_adjust(left=.075, right=.985, top=.91, bottom=.15, hspace=.73, wspace=.38)
        for i, (ax, (key, title, ylabel, threshold)) in enumerate(zip(axes.flat, metrics)):
            panel(ax, chr(97+i), title, '', ylabel)
            values = [data[m]['test'][key] for m in METHODS]
            ax.bar([0, 1], values, color=[BLUE, ORANGE], width=.52)
            maximum = max(values + ([threshold] if threshold is not None else []))
            ax.set_ylim(0, maximum*1.25)
            ax.set_xticks([0, 1], ['SuperMimic\n(small)', 'PPO'])
            ax.tick_params(axis='x', length=0)
            for x, value in enumerate(values):
                label = f'{value:.1f}' if key.endswith('s2') else (f'{value:.2f}' if key.endswith('deg') else f'{value:.3f}')
                ax.text(x, value+maximum*.035, label, ha='center', va='bottom', fontsize=12,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .5}, zorder=4)
            if threshold is not None:
                ax.axhline(threshold, color=GRAY, linewidth=.9, linestyle=(0, (5, 4)))
        save(fig, 'figure_4_test_metrics', 'Independent test: 64 trajectories, checkpoints selected on validation only; seed 42. Lower is better.\nDashed lines are per-trajectory success thresholds; bars are means. Both methods: 0% success and 0% falls.')

    notes = '''绘图说明（仅重画已有正式实验，无新增训练）

参考：用户提供的 SuperMimic 原论文截图；采用衬线字体、蓝色/橙色、左侧子图标题、简化坐标轴和适用的对数横轴。
本项目是 Sesame 小配置改编，图中数值完全来自修正后实验，不是论文的 G1 结果。

图 1：验证回报随训练时间/真实训练数据变化，以及验证成功率。
归一化回报 = 原始 9 秒轨迹回报 / 33.75。
33.75 = 9 × (1 + 0.25 + 0.5 + 0.5 + 1 + 0.5)，即所有跟踪奖励最大、惩罚为零时的理论上限。
没有使用观测到的最小/最大回报缩放，也没有截断或平滑曲线。
图 1b 使用对数横轴，零步点不能出现在该坐标中；初始策略回报另以灰色横线标出。
SuperMimic 的离线预训练点沿用初始策略评估，因为该阶段尚未改变策略。
每个数据点为单个训练 seed 42 下的 64 条验证轨迹均值；没有多 seed 误差带。
成功标准是连续两次验证成功率达到 80%，两种方法均未达到，不能画成功所需步数/时间的数值排名。

图 2：SuperMimic 验证回报随想象步数变化；世界模型的 8 步预测误差随累计真实训练数据变化。
PPO 无世界模型及想象步数，因此不出现在图 2。
预测误差来自固定的离线验证集采样，不是论文中的在线策略轨迹误差。
normalized_rmse 为 29 维物理状态按 RobotMath.delta_scale 逐维缩放后、对 8 步有效状态求均方再开根号。
这是 Sesame 定义的 RMSE，与原论文的误差定义/量级不能直接比较；故不照搬论文的 ×10^-3 标注。
横轴是训练进度，不是数据集长度扫描。原始世界模型预测误差及米/弧度误差保存在 world_prediction.csv。

图 3：实际仿真采样吞吐量和训练时间拆分。
吞吐量 = 每个不重叠 12 轮更新区间采集的真实状态转移数 / 区间采样总时间；末段不足 12 轮按实计。
训练时间含离线加载/预训练、在线仿真采样和优化，排除验证、测试、初始化和日志等开销。
当前日志无法分离纯世界模型前向耗时；不能将包含模型学习、想象及策略/价值更新的 update_seconds 当成世界模型仿真速度。
因此未伪造 World model 与 MuJoCo Warp 的速度曲线，也没有声称世界模型加速倍数。
只有 512 个真实环境的配置，未做并行环境数、数据集规模、想象长度或多 GPU 扫描。
GPU 与其他作业共享，两组还有部分时间重叠；时间/吞吐量为本轮观测，不是受控速度基准或独占 GPU-hours。

图 4：验证集所选权重在独立测试集的跟踪误差及关节加速度；所有 64 条轨迹均计入。
SuperMimic 权重对应 2,037,551 条真实数据，PPO 权重对应 1,769,472 条；PPO 最终训练总量为 2,048,000。
虚线是单条轨迹的成功阈值，柱子是所有轨迹的误差均值；均值低于阈值不等于全部轨迹成功。
关节加速度无预设成功阈值。所有图均为单 seed，不提供统计显著性结论。

PNG：300 dpi；PDF：矢量曲线和嵌入字体；all_figures.pdf 合并四页。
复现命令：
  /home/ubuntu/miniforge3/envs/metaworld_v010/bin/python -s scripts/plot_supermimic_paper_style.py --study-dir runs/supermimic_comparison_corrected_20261110
'''
    (output / 'figure_notes.txt').write_text(notes)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    manifest = {'study': str(study), 'training_seed': 42, 'return_denominator': reward_ceiling,
                'multi_seed_confidence_bands': False, 'curves_smoothed': False,
                'throughput_block_updates': 12, 'figures': saved, 'source_sha256': source_hashes,
                'plot_script_sha256': sha(Path(__file__)),
                'validation_points': {m: len(data[m]['validation']) for m in METHODS},
                'world_prediction_points': len(world)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'figures': saved, 'return_denominator': reward_ceiling}, indent=2))


if __name__ == '__main__':
    main()
