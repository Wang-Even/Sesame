#!/usr/bin/env python3
"""Plot saved metadata only; never runs a simulator or model on CPU."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live writer may have an incomplete final line.
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    spec = read_json(study / 'protocol.json')
    if not spec:
        raise RuntimeError('Missing frozen protocol')
    output = study / 'analysis'
    output.mkdir(exist_ok=True)
    known_issue = read_json(study / 'known_issue.json')
    correction = read_json(study / 'correction.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    labels = {'supermimic': 'Small SuperMimic adaptation', 'ppo': 'PPO'}
    colors = {'supermimic': '#1769aa', 'ppo': '#e67e22'}
    records = {}
    flattened = []
    for algorithm in labels:
        run = study / f'{algorithm}_seed42'
        evaluations = read_jsonl(run / 'evaluations.jsonl')
        training = read_jsonl(run / 'training.jsonl')
        result = read_json(run / 'result.json')
        records[algorithm] = {'result': result, 'training': training,
                              'validation': [r for r in evaluations if r['label'] == 'validation'],
                              'test': read_json(run / 'test_result.json')}
        for row in evaluations:
            if row['label'] == 'test' and records[algorithm]['test']:
                row = {**row, **{k: v for k, v in records[algorithm]['test'].items()
                                 if k.startswith('selected_checkpoint')}}
            flattened.append({'algorithm': algorithm, **{k: v for k, v in row.items()
                              if not isinstance(v, (list, dict))},
                              'evaluation_executed': row.get('stage') != 'world_pretrained_actor_unchanged',
                              'evaluation_seconds_charged': 0 if row.get('stage') == 'world_pretrained_actor_unchanged' else row['evaluation_seconds']})
    if flattened:
        fields = list(dict.fromkeys(k for r in flattened for k in r))
        with (output / 'evaluation_metrics.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader(); writer.writerows(flattened)
    timing = []
    for algorithm, data in records.items():
        r = data['result']
        collection = sum(t['collection_seconds'] for t in data['training'])
        optimization = sum(t['update_seconds'] for t in data['training'])
        offline = r.get('offline_pretraining_seconds', next(
            (v['training_seconds'] for v in data['validation'] if v.get('stage') == 'world_pretrained_actor_unchanged'), 0.))
        timing.append({'algorithm': algorithm, 'status': r.get('status', 'not_started'),
                       'offline_loading_and_pretraining_seconds': offline,
                       'online_simulation_collection_seconds': collection,
                       'online_optimization_seconds': optimization,
                       'training_seconds': offline + collection + optimization,
                       'evaluation_seconds': r.get('evaluation_seconds'),
                       'program_elapsed_before_shutdown_seconds': r.get('wall_seconds_before_shutdown'),
                       'observed_peak_process_gpu_mib': r.get('peak_process_gpu_mib')})
    with (output / 'time_breakdown.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(timing[0])); writer.writeheader(); writer.writerows(timing)
    panels = [('success_rate', 'Validation success rate', .8),
              ('mean_return', 'Validation return', None),
              ('velocity_rmse_m_s', 'Planar velocity RMSE (m/s)', .05),
              ('heading_rms_deg', 'Heading RMS error (deg)', 10.),
              ('joint_position_rmse_rad', 'Joint position RMSE (rad)', .25),
              ('root_position_rmse_m', 'Root position RMSE (m)', .10)]
    for xkey, xscale, xlabel, name in [
        ('cumulative_real_steps', 1e6, 'Cumulative real transitions (millions)', 'curves_by_steps'),
        ('training_seconds', 60., 'Training time excluding evaluation (min)', 'curves_by_time')]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
        for ax, (key, ylabel, threshold) in zip(axes.flat, panels):
            for algorithm, data in records.items():
                rows = data['validation']
                if not rows:
                    continue
                ax.plot([r[xkey] / xscale for r in rows], [r[key] for r in rows],
                        '-o' if algorithm == 'supermimic' else '--s', markersize=3,
                        linewidth=1.7, color=colors[algorithm], label=labels[algorithm])
            if threshold is not None:
                ax.axhline(threshold, color='#666666', linestyle='--', linewidth=.8,
                           label='Success threshold' if key == 'success_rate' else None)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.ticklabel_format(axis='y', useOffset=False, style='plain')
            if key == 'success_rate':
                ax.set_ylim(-.02, 1.02)
            elif key != 'mean_return':
                ax.set_ylim(bottom=0)
        axes.flat[0].legend(fontsize=8)
        status = ', '.join(f'{labels[k]}: {v["result"].get("status", "not started")}' for k, v in records.items())
        prefix = 'Superseded debug run: ' if known_issue else ('Corrected implementation: ' if correction else '')
        timing_note = '\nShared GPU workload changed; timing is descriptive, not a controlled speed benchmark.' if xkey == 'training_seconds' and (study / 'resource_observation.json').exists() else ''
        fig.suptitle(prefix + 'Sesame reference tracking, full domain randomization, seed 42\n' + status + timing_note, fontsize=12)
        fig.savefig(output / f'{name}.png'); fig.savefig(output / f'{name}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for algorithm, data in records.items():
        rows = data['validation']
        for ax, key, title in zip(axes, ['fall_rate', 'foot_position_rmse_m', 'joint_acceleration_rms_rad_s2'],
                                  ['Fall rate', 'Foot position RMSE (m)', 'Joint acceleration RMS (rad/s²)']):
            ax.plot([r['cumulative_real_steps'] / 1e6 for r in rows], [r[key] for r in rows],
                    '-o' if algorithm == 'supermimic' else '--s', markersize=3,
                    label=labels[algorithm], color=colors[algorithm])
            ax.set(xlabel='Cumulative real transitions (millions)', ylabel=title)
            ax.set_ylim(-.02, 1.02) if key == 'fall_rate' else ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'stability_curves.png'); fig.savefig(output / 'stability_curves.pdf'); plt.close(fig)
    summary = {'protocol': str(study / 'protocol.json'), 'smoke': spec['smoke'],
               'single_training_seed': 42, 'methods': {}}
    all_complete = all(d['result'].get('status') == 'completed' for d in records.values())
    achieved = [labels[k] for k, d in records.items() if d['result'].get('stable_success')]
    if known_issue:
        outcome = '本目录已被修正版实验替代，原始 SuperMimic 记录只供调试追溯。'
    elif spec['smoke']:
        outcome = '这是短测产物，仅用于验证程序，不是正式实验结论。'
    elif not all_complete:
        outcome = '实验仍在进行；已完成方法和阶段结果见下表。'
    elif not achieved:
        outcome = '结果：在约 205 万条真实训练数据的预算内，两组均未达到“连续两次验证成功率 ≥ 80%”的预设标准，尚不能给出稳定成功所需步数。'
    else:
        outcome = '结果：' + '、'.join(achieved) + '达到预设的稳定成功标准；首次达标步数和独立测试结果见下文。'
    summary['outcome'] = outcome
    lines = ['# 小配置 SuperMimic 与 PPO 对比', '',
             outcome, '',
             '任务：Sesame 在平地跟踪 0.10 m/s 直行参考动作，控制频率 50 Hz，完整域随机化。两组使用相同参考动作、奖励、策略网络、随机初始权重和配对评估初始状态。本轮 PPO 在本轮参考动作跟踪奖励下重新训练；旧版固定速度 PPO 的奖励与成功标准不同。', '',
             '## 成功标准', '',
             '一次评估包含 64 条完整 9 秒轨迹。单条轨迹需全程不摔倒、不越界，并同时满足：平面速度 RMSE ≤ 0.05 m/s、航向 RMS 误差 ≤ 10°、关节位置 RMSE ≤ 0.25 rad、足端相对机身的位置 RMSE ≤ 0.02 m、机身位置 RMSE ≤ 0.10 m。误差均相对于参考动作计算。验证成功率连续两次达到 80% 才记为达标。', '',
             '这些阈值在正式运行前固定；曲线上的误差是所有评估轨迹的均值，成功率则逐条检查全部条件。训练采样从随机参考帧开始，评估从参考第 0 帧完整跑 9 秒。', '',
             '## 结果', '',
             '| 方法 | 状态 | 累计真实交互数据 | 训练时间 | 最佳验证成功率 | 测试成功率 | 摔倒率（测试） |',
             '|---|---|---:|---:|---:|---:|---:|']
    if known_issue:
        lines[4:4] = ['本目录的 SuperMimic 结果已被替代：想象轨迹正常到达参考末尾时曾误加摔倒惩罚。请使用修正版目录的最终对比。PPO 更新函数不受影响。', '']
    elif correction:
        lines[4:4] = ['实现更正：正常到达参考末尾只停止价值自举；摔倒惩罚仅用于失败。SuperMimic 已从随机策略重新训练。PPO 复用本轮同任务的对照训练，其数值实现未变；来源见 ../baseline_reuse.json，GPU 边界验证见 ../boundary_validation.json。', '']
    for algorithm, data in records.items():
        r, test = data['result'], data['test']
        last = data['training'][-1] if data['training'] else {}
        best = r.get('best_validation') or max(data['validation'], key=lambda x: (x['success_rate'], x['mean_return']), default={})
        def percent(value):
            return '—' if value is None else f'{100 * value:.2f}%'
        steps = r.get('cumulative_real_steps', last.get('cumulative_real_steps', 0))
        seconds = r.get('training_seconds', last.get('training_seconds', 0))
        stable = r.get('stable_success')
        first = next((e for e in data['validation'] if e['success_rate'] >= .8), None)
        entry = {'status': r.get('status', 'not_started'), 'total_real_steps': steps,
                 'training_seconds': seconds, 'first_observed_threshold_crossing': first,
                 'stable_success': stable, 'best_validation': best, 'test': test,
                 'peak_process_gpu_mib': r.get('peak_process_gpu_mib'),
                 'evaluation_seconds': r.get('evaluation_seconds'),
                 'wall_seconds_before_shutdown': r.get('wall_seconds_before_shutdown')}
        summary['methods'][algorithm] = entry
        lines.append(f'| {labels[algorithm]} | {r.get("status", "未开始")} | {steps:,} | {seconds / 60:.2f} 分钟 | {percent(best.get("success_rate"))} | {percent(test.get("success_rate"))} | {percent(test.get("fall_rate"))} |')
    lines += ['', '测试使用验证集选出的权重；训练总预算与所选权重的训练步数分别记录：', '']
    for algorithm, data in records.items():
        test = data['test']
        if test:
            lines.append(f'- {labels[algorithm]}：{test["selected_checkpoint_cumulative_steps"]:,} 条累计真实训练数据时的 `{test["selected_checkpoint"]}`。')
    lines += ['', '所选权重在独立测试集上的跟踪误差（全部测试轨迹均纳入）：', '',
              '| 方法 | 速度 RMSE（m/s） | 航向 RMS（°） | 关节位置 RMSE（rad） | 足端位置 RMSE（m） | 机身位置 RMSE（m） |',
              '|---|---:|---:|---:|---:|---:|']
    for algorithm, data in records.items():
        test = data['test']
        values = [test.get(k) for k in ('velocity_rmse_m_s', 'heading_rms_deg', 'joint_position_rmse_rad',
                                       'foot_position_rmse_m', 'root_position_rmse_m')]
        lines.append('| ' + labels[algorithm] + ' | ' + ' | '.join('—' if v is None else f'{v:.4f}' for v in values) + ' |')
    lines += ['', '独立测试中各项条件分别通过的轨迹数（分母均为 64；这些条件需同时通过才计成功）：', '',
              '| 方法 | 速度 | 航向 | 关节位置 | 足端位置 | 机身位置 |',
              '|---|---:|---:|---:|---:|---:|']
    for algorithm, data in records.items():
        test = data['test']
        thresholds = [.05, math.radians(10), .25, .02, .10]
        counts = [str(sum(r[i] <= threshold for r in test['episode_metrics'])) if test else '—'
                  for i, threshold in enumerate(thresholds)]
        lines.append('| ' + labels[algorithm] + ' | ' + ' | '.join(counts) + ' |')
    lines += ['', '### 达标步数', '']
    for algorithm, data in records.items():
        r = data['result']; stable = r.get('stable_success')
        if stable:
            first, confirmation = stable['first_crossing'], stable['confirmation']
            earlier = [e for e in data['validation'] if e['cumulative_real_steps'] < first['cumulative_real_steps']]
            lower = earlier[-1]['cumulative_real_steps'] if earlier else 0
            lines.append(f'- {labels[algorithm]}：首次记录达标为 {first["cumulative_real_steps"]:,} 条累计真实数据（训练 {first["training_seconds"] / 60:.2f} 分钟），在 {confirmation["cumulative_real_steps"]:,} 条确认。实际跨越阈值发生在两次评估之间，区间为 ({lower:,}, {first["cumulative_real_steps"]:,}]，不能精确到单步。')
        elif r.get('status') == 'completed':
            lines.append(f'- {labels[algorithm]}：预算内未达到“连续两次验证成功率 ≥ 80%”，不能给出成功所需步数。')
        else:
            lines.append(f'- {labels[algorithm]}：实验尚未完整结束，暂不判断成功所需步数。')
    lines += ['', '## 统计口径与限制', '',
              '- “步数”是用于训练的各并行环境状态转移条数；512 个环境各执行一次控制动作记为 512 条。两组上限均为 2,048,000 条，按完整采样批次停止，可能略少于上限。评估交互不进入训练缓存，也不计入训练数据横轴。',
              '- SuperMimic 的横轴包括已有 TRAIN 数据 300,847 条，再加新仿真交互；PPO 从零开始采集。想象轨迹单独记录，不计入真实交互。离线预训练后的第二个点沿用未改变的初始策略评估，避免误认为产生了性能提升。',
              '- 训练时间包括离线数据加载及世界模型预训练、在线采样与优化；排除评估、仿真初始化、日志与保存等开销。程序总耗时和评估耗时另存于 result.json。已有参考动作的生成和教师策略训练是两组共享的前置成本，未计入本轮训练时间。',
              '- 仅训练随机种子 42；GPU 与其他作业共享。时间为本机本轮实测，不能视为独占 GPU 的速度排名，也不支持统计显著性结论。',
              '- 测试集不用于拟合或选权重。若达标，测试首次达标权重；否则测试验证集最优权重。test_result.json 单独记录所选权重的训练步数。',
              '- 这是针对 Sesame 的小规模 [SuperMimic](https://arxiv.org/abs/2610.28480) 改编：保留世界模型、32 步可微分想象和价值自举，使用小 MLP 与已有离线数据预热；不是论文原配置的复现。', '',
              '## 文件', '',
              '- [按训练数据量的曲线（PNG）](curves_by_steps.png) / [PDF](curves_by_steps.pdf)：成功率、回报与跟踪误差。',
              '- [按训练时间的曲线（PNG）](curves_by_time.png) / [PDF](curves_by_time.pdf)：相同指标，耗时受共享资源影响。',
              '- [稳定性曲线（PNG）](stability_curves.png) / [PDF](stability_curves.pdf)：摔倒率、足端跟踪和关节加速度。',
              '- [评估数据 CSV](evaluation_metrics.csv)：所有评估的汇总数据；原始逐轨迹指标在各方法 evaluations.jsonl。',
              '- [时间拆分 CSV](time_breakdown.csv)：预训练、采样、优化、评估耗时与进程显存峰值。离线预训练后的曲线点复用了初始评估，其 evaluation_seconds_charged 为 0。',
              '- 各方法 training.jsonl、world_pretraining.jsonl、world_validation.jsonl（后两项仅 SuperMimic）：训练日志。',
              '- 各方法 best.pt / final.pt，达标时另有 first_success.pt：模型权重。', '',
              '重新训练（必须使用新的运行目录；脚本自动调用 Isaac CUDA 环境）：', '', '```bash',
              '/home/ubuntu/miniforge3/envs/metaworld_v010/bin/python -s scripts/run_supermimic_comparison.py --study-dir runs/supermimic_comparison_new',
              '```', '', '从已有日志重新绘图：', '', '```bash',
              f'python scripts/analyze_supermimic_comparison.py --study-dir {study}', '```', '']
    for algorithm, data in records.items():
        selected = data['test'].get('selected_checkpoint', 'best.pt')
        description = '测试使用的权重' if data['test'] else '当前最佳验证权重'
        lines += [f'- [{labels[algorithm]}：{description}](../{algorithm}_seed42/{selected})。']
    lines += ['']
    resources = read_json(study / 'resource_observation.json')
    if resources:
        summary['resource_observation'] = resources
        lines += ['实测资源变化：PPO 期间监测到新增其他 GPU 作业，详见 ../resource_observation.json。因此本轮耗时受到不同外部负载影响，不能据此判断算法本身谁更快；优先比较真实训练数据量与任务表现。', '']
        if resources.get('corrected_supermimic_overlaps_with_ppo'):
            lines += ['修正版 SuperMimic 补跑期间还与 PPO 部分并行，共享同一 GPU；时间曲线用于记录本次运行，不能作为受控速度基准。', '']
    if correction:
        summary['correction'] = correction
        summary['baseline_reuse'] = read_json(study / 'baseline_reuse.json')
    if known_issue:
        summary['superseded_debug_run'] = known_issue
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
