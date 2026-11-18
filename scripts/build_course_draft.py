"""Assemble the course report from recorded results; never runs an experiment.

Usage: python3 scripts/build_course_draft.py
Missing/unfinished experiments are displayed as pending. Rerun after training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import markdown

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ('ppo', 'PPO', '#c45c27'),
    ('supermimic', 'SuperMimic（官方代码）', '#155da6'),
    ('supermimic_deepseek', 'SuperMimic（DeepSeek）', '#188766'),
    ('supermimic_codex', 'SuperMimic（Codex）', '#8d4ca5'),
    ('supermimic_h16', 'H=16，有 Critic', '#2d8b91'),
    ('supermimic_h64', 'H=64，有 Critic', '#b27920'),
    ('supermimic_no_critic', 'H=32，无 Critic', '#9b4f5b'),
]
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT = FontProperties(fname=FONT_PATH)
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
matplotlib.font_manager.fontManager.addfont(FONT_PATH)


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def read_lines(path: Path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
            # A live writer can have a partial last line. Keep complete records.
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def minutes(value):
    return '—' if value is None else f'{value/60:.2f}'


def save_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def load_runs(suite):
    result = []
    for name, label, color in METHODS:
        directory = suite/name
        status = read_json(directory/'status.json')
        curve = read_lines(directory/'curve.jsonl')
        if status.get('status') == 'complete':
            metrics = status.get('final_test')
            assert metrics and (directory/'final.pt').is_file(), f'{name}: completed run lacks final evaluation/checkpoint'
            assert metrics['episodes'] == 12 and metrics['steps_per_episode'] == 300, f'{name}: unexpected test scope'
            assert len(metrics['episode_returns']) == 12, f'{name}: incomplete test episodes'
            assert status['real_steps'] == 2097152, f'{name}: budget differs'
            assert curve and curve[-1]['real_steps'] == status['real_steps'], f'{name}: curve budget differs'
            assert math.isclose(curve[-1]['training_s'], status['training_s'], rel_tol=1e-8), f'{name}: training clocks differ'
            assert 0 < status['training_s'] <= status['end_to_end_s'], f'{name}: invalid timing'
            for key in ('return', 'velocity_rmse_m_s', 'heading_rms_deg', 'root_position_rmse_m', 'fall_rate'):
                assert math.isfinite(metrics[key]), f'{name}: invalid {key}'
        assert all(a['training_s'] <= b['training_s'] for a, b in zip(curve, curve[1:])), f'{name}: nonmonotonic training clock'
        result.append(dict(name=name, label=label, color=color, directory=directory,
                           status=status, curve=curve,
                           metrics=status.get('final_test') if status.get('status') == 'complete' else None))
    return result


def plot_curves(runs, out, ablation=False):
    selected = [runs[i] for i in ([1, 4, 5, 6] if ablation else [0, 1, 2, 3])]
    fields = [('return', '验证轨迹回报（Return）'),
              ('velocity_rmse_m_s', '速度 RMSE（m/s）'),
              ('heading_rms_deg', '偏航 RMS（°）'),
              ('root_position_rmse_m', '机身位置 RMSE（m）')]
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.1), layout='constrained')
    for ax, (field, ylabel) in zip(axes.flat, fields):
        count = 0
        observed = []
        for run in selected:
            points = [p for p in run['curve'] if field in p and 'training_s' in p]
            if not points:
                continue
            label = 'H=32，有 Critic' if ablation and run['name'] == 'supermimic' else run['label']
            pending = run['status'].get('status') != 'complete'
            if pending:
                label += '（中止/失败）' if run['status'].get('status') == 'failed' else '（未完成）'
            ax.plot([p['training_s']/60 for p in points], [p[field] for p in points],
                    lw=2, marker='.', markersize=4, color=run['color'],
                    linestyle='--' if pending else '-', label=label)
            count += 1
            observed.extend(p[field] for p in points)
        ax.set_xlabel('训练时间（分钟）')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.16)
        if field == 'return':
            # Negative rewards are possible; keep all recorded values visible.
            ax.set_ylim(bottom=min([0] + observed), top=max([30] + observed))
        if count:
            ax.legend(frameon=False, fontsize=8, loc='best')
        else:
            ax.text(.5, .5, '实验尚未完成\n不绘制推测曲线', ha='center', va='center',
                    color='#64748b', transform=ax.transAxes)
    name = 'supermimic_ablation' if ablation else 'supermimic_comparison'
    fig.savefig(out/f'{name}.png', dpi=170)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)


def extended_ppo_comparison(directory, suite, runs, out):
    """Optional post-hoc budget extension; never splice the original PPO curve."""
    status = read_json(directory/'status.json')
    if status.get('status') != 'complete' or any(r['status'].get('status') != 'complete' for r in runs[1:4]):
        return None, []
    protocol = read_json(directory/'protocol.json')
    original = read_json(suite/'ppo/protocol.json')
    for key, value in original['args'].items():
        if key not in ('steps', 'output'):
            assert protocol['args'][key] == value, f'Extended PPO differs in {key}'
    for key in ('ppo_config', 'reward_version', 'runner_config', 'official_commit', 'mjlab_commit'):
        assert protocol[key] == original[key], f'Extended PPO differs in {key}'
    assert protocol['args']['steps'] == status['real_steps'] == 8388608, 'Extended PPO budget differs'
    assert (directory/'final.pt').is_file(), 'Extended PPO final checkpoint missing'
    assert (read_json(directory/'initial_actor.json')['parameter_sha256'] ==
            read_json(suite/'ppo/initial_actor.json')['parameter_sha256']), 'Extended PPO initial actor differs'
    curve = read_lines(directory/'curve.jsonl')
    assert curve and curve[0]['real_steps'] == 0 and curve[-1]['real_steps'] == status['real_steps']
    assert all(a['training_s'] <= b['training_s'] for a,b in zip(curve,curve[1:]))
    assert math.isclose(curve[-1]['training_s'],status['training_s'],rel_tol=1e-8)
    assert 0 < status['training_s'] <= status['end_to_end_s']
    assert status['final_test']['episodes'] == 12 and status['final_test']['steps_per_episode'] == 300
    extended = dict(name='ppo_extended',label='PPO（延长预算）',color=METHODS[0][2],curve=curve,status=status)
    selected = [extended,*runs[1:4]]
    # Cover the complete SuperMimic budget for all three sources. Using the
    # shortest SM time would discard the other final points for a ~1 s timing
    # difference and replace them with checkpoints ~23 s earlier.
    cutoff = max(run['curve'][-1]['training_s'] for run in runs[1:4])
    assert curve[-1]['training_s'] >= cutoff, 'Extended PPO has not covered the SuperMimic time window'
    rows, recorded = [], []
    fig, ax = plt.subplots(figsize=(10.3,3.5),layout='constrained')
    for run in selected:
        points = run['curve']
        assert all(math.isfinite(p['return']) for p in points)
        ax.plot([p['training_s']/60 for p in points],[p['return'] for p in points],
                color=run['color'],lw=1.8,marker='o',markersize=3,label=run['label'])
        point = [p for p in points if p['training_s'] <= cutoff+1e-8][-1]
        rows.append(dict(method=run['name'],label=run['label'],window_end_s=cutoff,
                         actual_evaluation_s=point['training_s'],real_steps=point['real_steps'],
                         validation_return=point['return']))
        recorded.extend(dict(method=run['name'],real_steps=p['real_steps'],training_s=p['training_s'],
                             validation_return=p['return']) for p in points)
    values = [row['validation_return'] for row in recorded]
    ax.set(xlabel='训练时间（分钟）',ylabel='验证轨迹回报（Return）',
           ylim=(min([0]+values),max([30]+values)))
    ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8,loc='best')
    fig.savefig(out/'ppo_extended_time_comparison.png',dpi=170)
    fig.savefig(out/'ppo_extended_time_comparison.pdf');plt.close(fig)
    save_csv(out/'ppo_extended_time_points.csv',rows)
    save_csv(out/'ppo_extended_curve_data.csv',recorded)
    save_csv(out/'ppo_extended_final_result.csv',[dict(
        method='ppo_extended',real_steps=status['real_steps'],training_s=status['training_s'],
        end_to_end_s=status['end_to_end_s'],test_return=status['final_test']['return'],
        checkpoint_sha256=sha(directory/'final.pt'))])
    sources=[directory/name for name in ('status.json','protocol.json','curve.jsonl','initial_actor.json')]
    return dict(rows=rows,cutoff=cutoff,status=status), sources


def extended_section(data):
    if data is None:
        return []
    status = data['status']
    lines = ['', '### 6.8 延长 PPO 预算的时间对照（补充实验）', '',
             '观察首轮计时后，补做一组独立从头训练的 PPO：seed 42、数据、网络、超参和独立缓存规则均保持一致，仅将真实步预算延长到 8,388,608。该实验用于补足时间窗口，原七组固定预算结果保留。', '',
             '![延长 PPO 预算的训练时间对照](ppo_extended_time_comparison.png)', '',
             '图 9：延长预算 PPO 与三组 SuperMimic 的原始验证曲线。点为实际评估时刻，连线仅辅助阅读，长首段包含首次编译等待及采样更新。未拼接旧 PPO 曲线，各曲线止于自身实际预算，不补点、不扣除编译时间。', '',
             f'以三组 SuperMimic 完成预算的最大训练用时为上限，下表取 0–{data["cutoff"]/60:.2f} 分钟内各组最后一次实际评估。各组评估时刻不同，保留时间差，不将连线插值当作同步分数。', '',
             '| 方法 | 实际评估时间（分钟） | 已用真实步数 | 验证 Return |',
             '|---|---:|---:|---:|']
    for row in data['rows']:
        lines.append(f'| {row["label"]} | {minutes(row["actual_evaluation_s"])} | {row["real_steps"]} | {number(row["validation_return"])} |')
    ppo, official = data['rows'][:2]
    delta = official['validation_return']-ppo['validation_return']
    lines += ['', f'使用同一原始训练参考的官方代码 SuperMimic，相对延长预算 PPO 的验证回报差为 {delta:+.3f}；'
              f'两次实测时刻相差 {abs(official["actual_evaluation_s"]-ppo["actual_evaluation_s"]):.1f} 秒，'
              f'对应真实环境步为 {official["real_steps"]:,} 与 {ppo["real_steps"]:,}。这是当前配置下的时间与样本利用对照，不能推广为算法的普遍差距。']
    lines += ['', f'延长 PPO 最终训练 {minutes(status["training_s"])} 分钟，完整执行 {minutes(status["end_to_end_s"])} 分钟；预算结束的 final.pt 测试回报为 {status["final_test"]["return"]:.3f}。该补充仅描述本次实测时间窗口，未据离散评估点计算同目标加速倍数，也不替代多 seed 或超参数搜索。']
    return lines


def collection_costs():
    entries = [
        ('DeepSeek 试采集', 'runs/deepseek_report_pilot_20261115'),
        ('DeepSeek 前一轮探索（中止）', 'runs/deepseek_report_20261115'),
        ('DeepSeek 正式采集', 'runs/deepseek_report_seeded_20261115'),
        ('Codex 历史正式采集', 'runs/agent32_collection_20261114'),
        ('Codex 当前规则重判', 'runs/agent32_collection_20261114_reassessed_20261115'),
    ]
    rows = []
    for label, folder in entries:
        path = ROOT/folder
        source = next((path/name for name in ('summary.json', 'collection_summary.json', 'status.json')
                       if (path/name).exists()), None)
        data = read_json(source) if source else {}
        elapsed = data.get('elapsed_s')
        time_basis = 'elapsed_s' if elapsed is not None else ''
        if elapsed is None and all(k in data for k in ('started_unix', 'finished_unix')):
            elapsed = data['finished_unix']-data['started_unix']
            time_basis = 'recorded finished_unix - started_unix'
        state = data.get('status', 'pending')
        if 'KeyboardInterrupt' in data.get('error', ''):
            state = 'interrupted'
        rows.append(dict(label=label, source=str(source.relative_to(ROOT)) if source else '',
                         status=state, accepted=data.get('accepted_count'),
                         attempts=data.get('attempts'), api_calls=data.get('api_calls'),
                         elapsed_s=elapsed, time_basis=time_basis))
    return rows


def source_order(dataset):
    protocol = read_json(dataset/'protocol.json')
    if not protocol:
        protocol = read_json(dataset/'dataset_protocol.json')
    mapping = protocol.get('mapping', {}).get('common_test', [])
    if mapping:
        order = [p['source'] for p in mapping]
        assert len(order) == 12, 'Common test must contain 12 whole clips'
        assert {s: order.count(s) for s in set(order)} == {'original': 4, 'deepseek': 4, 'codex': 4}, order
        return order, 'dataset protocol mapping.common_test'
    return ['original']*4+['deepseek']*4+['codex']*4, 'predeclared order; dataset protocol not yet available'


def gpu_startup_note(suite):
    snapshots = []
    for name, _, _ in METHODS:
        path = suite/f'{name}_gpu_before.csv'
        if path.exists():
            records = list(csv.reader(path.read_text().splitlines()))
            if records and records[0] and records[0][0].strip() == 'pid':
                snapshots.append((name, any(row and row[0].strip() for row in records[1:])))
    if not snapshots:
        return 'GPU 启动占用记录尚未齐备。'
    occupied = sum(present for _, present in snapshots)
    return (f'已记录 {len(snapshots)} 次 GPU 启动前快照，其中 {occupied} 次列出其他 CUDA 计算进程。'
            '这些快照只描述启动时刻，不代表完整训练期间的独占监测。')


def tables_and_results(runs, dataset, out):
    rows, groups = [], []
    order, order_basis = source_order(dataset)
    names = dict(original='原始参考', deepseek='DeepSeek 参考', codex='Codex 参考')
    for run in runs:
        status, m = run['status'], run['metrics'] or {}
        row = dict(method=run['name'], label=run['label'], status=status.get('status', 'pending'),
                   checkpoint='final.pt', training_s=status.get('training_s'),
                   end_to_end_s=status.get('end_to_end_s'), real_steps=status.get('real_steps'),
                   imagined_steps=status.get('imagined_steps'),
                   return_value=m.get('return'), normalized_return=m.get('normalized_return'),
                   velocity_rmse_m_s=m.get('velocity_rmse_m_s'), heading_rms_deg=m.get('heading_rms_deg'),
                   root_position_rmse_m=m.get('root_position_rmse_m'),
                   local_termination_fraction=m.get('fall_rate'),
                   peak_process_memory_mib=status.get('peak_process_memory_mib'),
                   source=str((run['directory']/'status.json').relative_to(ROOT)))
        rows.append(row)
        episode_returns = m.get('episode_returns', [])
        if episode_returns:
            assert len(episode_returns) == 12, (run['name'], len(episode_returns))
            for source, label in names.items():
                values = [v for v, s in zip(episode_returns, order) if s == source]
                groups.append(dict(method=run['name'], source=source, source_label=label, episodes=len(values),
                                   mean_return=sum(values)/len(values), min_return=min(values), max_return=max(values)))
    save_csv(out/'supermimic_test_results.csv', rows)
    save_csv(out/'supermimic_test_by_source.csv', groups,
             ['method', 'source', 'source_label', 'episodes', 'mean_return', 'min_return', 'max_return'])
    curve_rows = []
    for run in runs:
        for point in run['curve']:
            curve_rows.append(dict(method=run['name'], **{k: point.get(k) for k in (
                'real_steps', 'imagined_steps', 'training_s', 'return', 'velocity_rmse_m_s',
                'heading_rms_deg', 'root_position_rmse_m')}))
    save_csv(out/'supermimic_curve_data.csv', curve_rows,
             ['method', 'real_steps', 'imagined_steps', 'training_s', 'return',
              'velocity_rmse_m_s', 'heading_rms_deg', 'root_position_rmse_m'])
    return rows, groups, order_basis


def state_label(row):
    return {'complete':'完成', 'training':'训练中', 'initializing':'初始化',
            'failed':'中止/失败', 'pending':'待完成'}.get(row['status'], row['status'])


def motion_diagnosis(directory, suite, out):
    """Read separate, fresh rollouts without changing the training-end results."""
    if read_json(directory/'status.json').get('status') != 'complete':
        return [], []
    protocol = read_json(directory/'protocol.json')
    assert Path(protocol['suite']).resolve() == suite.resolve(), 'Motion diagnosis suite differs'
    assert protocol['suite_protocol_sha256'] == sha(suite/'protocol.json'), 'Motion diagnosis protocol differs'
    comparison = read_json(directory/'comparison.json', [])
    assert {r['group'] for r in comparison} == {m[0] for m in METHODS[:4]}, 'Four motion groups required'
    assert len(comparison) == 4, 'Repeated motion groups'
    rows = []
    source_rows = []
    statistics_files = []
    for name, label, _ in METHODS[:4]:
        result = next(r for r in comparison if r['group'] == name)
        metadata = protocol['models'][name]
        assert result['episodes'] == 12, f'{name}: motion test scope differs'
        assert metadata['checkpoint_sha256'] == sha(suite/name/'final.pt'), f'{name}: motion checkpoint differs'
        rows.append(dict(label=label, **result,
                         checkpoint_sha256=metadata['checkpoint_sha256'],
                         trace_sha256=metadata['trace_sha256'],
                         initial_state_max_abs_difference_from_first_group=metadata['initial_state_max_abs_difference_from_first_group']))
        statistics_file = directory/name/'motion_statistics.json'
        by_source = read_json(statistics_file)['by_source']
        statistics_files.append(statistics_file)
        for source, values in by_source.items():
            source_rows.append(dict(group=name, label=label, source=source, **values))
    save_csv(out/'motion_reevaluation.csv', rows)
    save_csv(out/'motion_reevaluation_by_source.csv', source_rows)
    sources = [directory/name for name in ('status.json', 'protocol.json', 'comparison.json')]+statistics_files
    return rows, sources


def motion_table(rows):
    if not rows:
        return []
    lines = ['', '**同权重运动复测（额外 rollout）**', '',
             '下表用四组原预算（各 2,097,152 步）的 final.pt，在相同初始状态重新执行全部 12 条测试轨迹，独立于上面的训练结束测试，不替换其回报或误差。完整片段的实际前进速度为世界 X 净位移 / 5.98 秒；提前终止片段使用有效时长。参考速度按同一时间段计算。横向位移取各片段最终世界 Y 位移的绝对值再平均；运动方向偏差为净位移方向相对世界 +X 的绝对夹角，与机头偏航角不同。', '',
             '| 方法 | 实际前进速度（m/s） | 参考速度（m/s） | 绝对横向位移（cm） | 运动方向偏差（°） | 完整执行 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lateral = row['mean_abs_lateral_displacement_m']
        lines.append(f'| {row["label"]} | {number(row["mean_forward_speed_from_displacement_m_s"],4)} | '
                     f'{number(row["mean_reference_forward_speed_from_displacement_m_s"],4)} | '
                     f'{number(None if lateral is None else lateral*100,2)} | '
                     f'{number(row["mean_abs_net_movement_direction_deg"],2)} | '
                     f'{row["complete_episodes"]}/{row["episodes"]} |')
    sm = rows[1:]
    speeds = [r['mean_forward_speed_from_displacement_m_s'] for r in sm]
    references = [r['mean_reference_forward_speed_from_displacement_m_s'] for r in sm]
    if all(v is not None for v in [*speeds, *references]) and max(speeds) < min(references):
        lines += ['', f'三组 SuperMimic 的平均实际速度为 {min(speeds):.3f}–{max(speeds):.3f} m/s，'
                  f'仍低于共同参考约 {references[0]:.3f} m/s。较高回报不代表已精确完成速度跟踪。'
                  '后续可提高机身位移、速度跟踪项对误差的区分度，并比较混合来源参考训练；当前没有验证这些改动的收益。']
    lines += ['', '完整执行表示没有触发本地提前终止；不另设成功率门槛。接近静止时，净位移方向对微小漂移敏感，应结合速度阅读；净位移小于 1 μm 时方向未定义。有效回合数与初态一致性见 `motion_reevaluation.csv`，分来源运动指标见 `motion_reevaluation_by_source.csv`。']
    return lines


def result_table(rows):
    lines = ['| 方法 | 状态 | 测试 Return | 速度 RMSE（m/s） | 偏航 RMS（°） | 机身 RMSE（cm） |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        root = r['root_position_rmse_m']
        lines.append(f'| {r["label"]} | {state_label(r)} | {number(r["return_value"])} | '
                     f'{number(r["velocity_rmse_m_s"],4)} | {number(r["heading_rms_deg"],2)} | '
                     f'{number(None if root is None else root*100,2)} |')
    return lines


def dynamic_conclusion(rows, extended_complete=False):
    main = rows[:4]
    completed = [r for r in main if r['status'] == 'complete']
    if len(completed) != 4:
        pending = '、'.join(r['label'] for r in main if r['status'] != 'complete')
        return f'当前四组主对照完成 {len(completed)}/4；{pending} 尚无完整结果。图中只显示已经写入日志的实测点，不能据未完成实验判断优劣。'
    best = max(main, key=lambda r: r['return_value'])
    ppo, official = main[:2]
    delta = official['return_value']-ppo['return_value']
    comp = '较高' if delta > 0 else '较低' if delta < 0 else '相同'
    time_ratio = official['training_s']/ppo['training_s']
    text = (f'本轮相同真实步预算下，{best["label"]} 的测试回报最高（{best["return_value"]:.3f}）。'
            f'官方代码 SuperMimic 的回报比 PPO {comp}，差值 {delta:+.3f}；'
            f'其训练总用时为 PPO 的 {time_ratio:.2f} 倍。')
    if time_ratio >= 1:
        text += '本轮没有体现更短的完整预算训练时间，也不足以给出达到同一目标回报的加速倍数。'
        text += '延长 PPO 预算的补充时间对照见第 6.8 节。' if extended_complete else '后续需要更密集评估及更长的 PPO 预算。'
    else:
        text += '这是本机同设置单次运行的实测耗时差，尚不足以证明跨随机种子的稳定加速。'
    return text+' Agent 两组与原始参考组的差别包含训练数据分布，不能只归因于优化算法。'


def source_conclusion(rows, groups):
    official = rows[1]
    if official['status'] != 'complete':
        return ''
    values = {(g['method'], g['source']): g['mean_return'] for g in groups}
    source_names = dict(original='原始参考', deepseek='DeepSeek 参考', codex='Codex 参考')
    findings = []
    for row in rows[2:4]:
        if row['status'] != 'complete':
            continue
        better, worse = [], []
        for source, name in source_names.items():
            delta = values[(row['method'],source)]-values[('supermimic',source)]
            if delta > 0:
                better.append(name)
            elif delta < 0:
                worse.append(name)
        if better and worse:
            average = '总体均值更高，但' if row['return_value'] > official['return_value'] else ''
            findings.append(f'{row["label"]} 相对官方代码组{average}在{"、".join(worse)}上的回报更低，在{"、".join(better)}上更高。')
    if findings:
        findings.append('这表明总体均值不能代表每种来源都改善；参考动作分布与 Isaac / MuJoCo 动力学一致性同时变化，当前对照不能分离它们或语言模型能力的影响。')
    return ''.join(findings)


def first_update_note(suite, rows):
    official = rows[1]
    if official['status'] != 'complete':
        return ''
    updates = read_lines(suite/'supermimic/updates.jsonl')
    if not updates:
        return ''
    first = updates[0]['training_s']
    total = official['training_s']
    return (f'官方代码 SuperMimic 的首条更新日志对应累计训练 {first:.1f} 秒，'
            f'占本次总训练时间的 {100*first/total:.1f}%。这段包含首次编译、数据积累和更新，'
            '不能全部解释为纯编译时间。当前短预算任务受首次启动成本影响明显；图表保留这段实测时间，'
            '不能据原论文的大规模任务加速比推断本机也会同幅度加速。')


def assemble(out, suite, dataset, runs, rows, groups, costs, motion_rows=None, extended=None):
    baseline = (out/'baselines/section_baselines.md').read_text()
    baseline = baseline.replace('这些本地诊断不等同于课程统一评测成绩。', '')
    improvement_path = out/'improvements/section_improvements.md'
    if improvement_path.exists():
        improvements = improvement_path.read_text()
    else:
        improvements = '''## 4. 问题一：行走抖动

补充实验正在整理。计划比较相邻动作差惩罚权重，联合观察实际关节加速度、动作变化、前进速度和回报；结果未完成前不填写改善比例。

## 5. 问题二：高速前进

补充实验正在整理。计划在相同条件下提高目标前进速度，记录实际前进速度、偏航和提前终止情况；不把改指令后的零样本测试误称为已完成高速策略训练。
'''
    # Keep one protocol description; ablation/results and course scope are added below.
    method = (out/'section_method_review.md').read_text().split('### 6.4')[0].strip()
    method = method.replace('## 6. 训练效率：基于世界模型的方法', '## 6. 问题三：训练时间与稳定性——SuperMimic 对照')
    method = method.replace('### 6.2 四组对照的定义',
                            '![SuperMimic 与参考生成的流程](method_flow.png)\n\n图 6：参考动作提供任务目标；世界模型通过当前策略的真实仿真转移学习。\n\n### 6.2 四组对照的定义')
    method = method.replace('各组在同一 GPU 串行执行，记录缓存设置及其他作业占用',
                            '各组在同一 GPU 串行执行，每次训练使用独立的 Inductor/Triton 编译缓存，记录其他作业占用')
    method = method.replace('去掉 Critic 时固定折扣因子', '去掉 Critic 时固定折扣因子为 0.99')
    method = method.replace('各组训练参考统一为 24 条、每条 6 秒，数据来源是预设实验变量。',
        '各组训练参考统一为 24 条、每条 6 秒，数据来源是预设实验变量。原始参考取前 6 秒；Codex 从当前规则下通过的 41 条记录中取固定顺序前 32 条，再固定打乱为 24/4/4；DeepSeek 新采集 32 条并同样划分。Codex 采用新的划分，不沿用历史 32 条实验的数据划分。')
    method = method.replace('验证和测试使用同一套分来源保留的完整轨迹，各来源等权；',
        '四组共用验证集和测试集：原始、DeepSeek、Codex 每个来源各 4 条，共 12 条，各来源等权；')
    ready = sum(r['status'] == 'complete' for r in rows)
    top = ['# Sesame 四足机器人强化学习行走实验报告', '',
           '上海交通大学强化学习课程作业 · 初版 · 2026-11-15', '',
           f'本报告按环境准备、基线、奖励设计、抖动、高速前进和训练效率六部分组织。历史 DQN/PPO/SAC 各有 3 个训练种子；当前统一设置的 SuperMimic 对照与消融完成 {ready}/7。本轮新增训练均为 seed 42，属于单种子探索。正文展示图片、曲线和表格；视频后续在附录整理。', '',
           '**评测范围：**以下回报是对应本地仿真任务的研究指标。课程正式评分需要在统一接口中重新 rollout，不能直接使用不同训练奖励的日志回报代替。', '']
    text = '\n'.join(top)+baseline+'\n\n'+improvements+'\n\n'+method+'\n\n'
    new = ['### 6.4 当前跟踪奖励与参数', '',
           '本节使用当前奖励版本 `sesame_nonfoot_no_joint_limit_20261115`。跟踪项为 `w × exp(−误差平方/尺度²)`，身体项在所选非足端连杆上取平均；每步总奖励率乘 0.02 秒。它与第 2 节 Isaac 速度行走奖励不同，仅在本节四组及消融之间共用。', '',
           '| 奖励项 | 权重 w | 尺度 | 作用 |', '|---|---:|---:|---|',
           '| 机身位置跟踪 | 0.5 | 0.3 m | 跟踪参考的整体位移 |',
           '| 机身姿态跟踪 | 0.5 | 0.4 rad | 保持参考方向与姿态 |',
           '| 非足端连杆位置 | 1.0 | 0.3 m | 学习身体与腿部协调 |',
           '| 非足端连杆姿态 | 1.0 | 0.4 rad | 限制错误的肢体构型 |',
           '| 非足端连杆线速度 | 1.0 | 1.0 m/s | 跟踪运动速度 |',
           '| 非足端连杆角速度 | 1.0 | 3.14 rad/s | 跟踪转动速度 |',
           '| 相邻动作差平方和 | −0.1 | — | 减少动作跳变 |', '',
           'Actor 输入为 30 维本体观测的 10 帧历史及 220 维参考目标，共 520 维，输出 8 维动作。默认 H=32，世界模型 512×3、Actor 512/256/128、Critic 256/128；策略想象批量 2048，世界模型批量 512、每次 10 批、8 步预测训练，replay ratio=16。无 Critic 组与其他组均用 gamma=0.99。足端跟踪奖励与关节限位惩罚不参与本轮计算；实仿和想象奖励对应一致。', '',
           'PPO 每环境采样 32 步，更新 5 个 epochs、4 个 minibatches，学习率 0.001，clip=0.2，gamma=0.99、GAE λ=0.95，熵系数 0.005、目标 KL=0.01。初始动作标准差为 0.1，是本轮固定的探索设置，未经过 PPO 超参数搜索。', '',
           '### 6.5 四组主对照结果', '',
           '![四组同设置学习曲线](supermimic_comparison.png)', '',
            '图 7：同一验证集的回报和运动误差，横轴均为训练时间（分钟）。每 262,144 步评估一次；点为实际评估时刻，连线仅辅助阅读，长首段包含首次编译等待及采样更新。实线为已完成运行，虚线表示未完成记录；单个训练 seed 不绘制跨种子误差带。6 秒正奖励上限为 30，未提前终止不等于精确跟踪。', '']
    new += result_table(rows[:4])
    new += ['', dynamic_conclusion(rows, extended is not None), '',
            '各组使用 final.pt；下表把同一 12 条保留测试参考按来源分开，避免总体均值掩盖对某一种动作来源的偏好。同一步态家族中的变体不等于跨技能泛化。', '',
            '| 方法 | 原始参考 Return | DeepSeek 参考 Return | Codex 参考 Return |', '|---|---:|---:|---:|']
    for row in rows[:4]:
        values = [next((g['mean_return'] for g in groups if g['method'] == row['method'] and g['source'] == s), None)
                  for s in ('original', 'deepseek', 'codex')]
        new.append('| '+row['label']+' | '+' | '.join(number(v) for v in values)+' |')
    interpretation = source_conclusion(rows, groups)
    if interpretation:
        new += ['', interpretation]
    new += motion_table(motion_rows or [])
    new += ['', '### 6.6 想象长度与 Critic 消融', '',
            '![想象长度和 Critic 消融](supermimic_ablation.png)', '',
            '图 8：使用原始训练参考的 H=16、32、64 及 H=32 无 Critic 消融，真实步预算、奖励、Actor 与世界模型规模及评估集相同；无 Critic 组移除价值网络。点为实际评估时刻，连线仅辅助阅读，长首段包含首次编译等待及采样更新。H 改变想象步数和计算量，不能解释为相同想象预算。', '']
    ablation_rows = [dict(rows[i], label=('H=32，有 Critic' if i == 1 else rows[i]['label'])) for i in (1, 4, 5, 6)]
    new += result_table(ablation_rows)
    if all(r['status'] == 'complete' for r in ablation_rows):
        best = max(ablation_rows, key=lambda r: r['return_value'])
        no_critic_delta = rows[6]['return_value']-rows[1]['return_value']
        new += ['', f'本轮消融中 {best["label"]} 测试回报最高（{best["return_value"]:.3f}）；去掉 Critic 的回报变化为 {no_critic_delta:+.3f}。这是固定预算、单种子的结果，不据此声称某个设计在其他速度或动作上普遍更好。']
    else:
        new += ['', '消融尚未全部完成；当前不填写最佳 H 或 Critic 的改善比例。']
    new += ['', '### 6.7 训练、完整执行与数据准备成本', '',
            '| 方法 | 训练（分钟） | 完整执行（分钟） | 真实步数 | 想象步数 | 提前终止 | 峰值显存（GiB） |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        termination = r['local_termination_fraction']
        terminated = '—' if termination is None else f'{round(termination*12)}/12'
        memory = r['peak_process_memory_mib']
        new.append(f'| {r["label"]} | {minutes(r["training_s"])} | {minutes(r["end_to_end_s"])} | '
                   f'{r["real_steps"] if r["real_steps"] is not None else "—"} | '
                   f'{r["imagined_steps"] if r["imagined_steps"] is not None else "—"} | {terminated} | '
                   f'{number(None if memory is None else memory/1024,2)} |')
    new += ['', '训练用时读取同步后的 sampling/update 计时；完整执行读取脚本主计时区间，包含初始化、评估和保存，数据准备另列。提前终止由本地任务的终止条件统计，不单独命名为跌倒率；回合回报与均方根误差来自全部指定测试片段。峰值显存由进程采样监测，可能遗漏短时尖峰。', '', first_update_note(suite, rows), '', gpu_startup_note(suite), '',
            '| 数据准备阶段 | 状态 | 合格参考 | 生成/修订次数 | API 次数 | 实测耗时（分钟） |',
            '|---|---|---:|---:|---:|---:|']
    states = dict(complete='完成', failed='中止/失败', interrupted='主动中止', validating='进行中',
                  planning='进行中', generating='进行中', pending='待完成')
    for c in costs:
        cells = [c['label'], states.get(c['status'],c['status']),
                 *(str(c[k]) if c[k] is not None else '—' for k in ('accepted','attempts','api_calls')),
                 minutes(c['elapsed_s'])]
        new.append('| '+' | '.join(cells)+' |')
    deepseek_costs = [c for c in costs if c['label'].startswith('DeepSeek')]
    if deepseek_costs and all(c['elapsed_s'] is not None for c in deepseek_costs):
        formal = next(c for c in deepseek_costs if c['label'] == 'DeepSeek 正式采集')
        new += ['', f'DeepSeek 正式采集耗时 {minutes(formal["elapsed_s"])} 分钟；计入试采和中止探索后，共 {minutes(sum(c["elapsed_s"] for c in deepseek_costs))} 分钟。']
    new += ['', 'DeepSeek 前一轮为主动中止后调整提示。无 elapsed_s 时，只用已记录的开始与结束时间之差；无结束记录则不估算。重判不属于新采集。历史教师训练和控制器调试还有额外成本，主图未计入这些研发成本。Codex 的 API 次数为 0 只表示采集脚本没有调用外部 API；对话中的程序编写与此前调试没有单独计时，不代表参考生成没有成本。', '']
    new += extended_section(extended)
    new += ['', '## 附录：复现与后续视频整理', '',
            '主实验目录：`runs/course_comparison_20261115/`；统一数据与来源映射：`datasets/course_comparison_20261115/`。每组保存 protocol、源代码快照、曲线、更新记录和 final.pt；初始 Actor 可学习参数哈希用于核对初始化，第三方源文件哈希用于核对官方版本。参数哈希相同不代表完整仿真逐位一致：零步验证得分有小幅差异，原曲线起点未校正。', '',
            '本报告新增结果表：`supermimic_test_results.csv`、`supermimic_test_by_source.csv`、`supermimic_curve_data.csv`、`data_preparation_costs.csv`。重新生成：`python3 scripts/build_course_draft.py`。', '',
            '视频暂不在正文展示。后续附录按“环境导入／DQN-PPO-SAC／抖动／高速／四组 SuperMimic”整理；历史素材索引位于 `reports/supermimic_results_20261115/`。新实验的策略视频仍需按对应 final.pt 和统一测试条件导出，不能用旧实验视频替代。', '',
            '课程要求使用统一环境的全部指定回合计算轨迹回报，不另设小组成功率门槛；正式 rollout 记录与后续真机部署仍需独立完成。[课程要求原文](../../docs/course_requirements.txt)', '']
    return text+'\n'.join(new)


CSS = '''
@font-face {font-family:ReportNoto;src:url("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc");}
@page {size:A4;margin:14mm 14mm 16mm;@bottom-center{content:counter(page) " / " counter(pages);font-family:ReportNoto;font-size:8pt;color:#64748b;}}
body{font-family:ReportNoto,"Noto Sans CJK SC",sans-serif;font-size:9pt;line-height:1.5;color:#203047;margin:0;}
h1{font-size:21pt;line-height:1.3;margin:0 0 4mm;color:#142c47;}
h2{font-size:14pt;line-height:1.35;margin:7mm 0 3mm;padding-bottom:1.5mm;border-bottom:1pt solid #d5dfec;break-after:avoid;}
h3{font-size:11pt;margin:4mm 0 2mm;break-after:avoid;}
p{margin:0 0 2.5mm;orphans:3;widows:3;}
img{display:block;max-width:100%;max-height:100mm;margin:3mm auto;object-fit:contain;}
p:has(img){break-inside:avoid;break-after:avoid;}
p:has(img)+p{font-size:8pt;color:#55677a;}
table{border-collapse:collapse;width:100%;font-size:7.5pt;line-height:1.4;table-layout:fixed;margin:3mm 0 4mm;break-inside:avoid;}
thead{display:table-header-group;}tr{break-inside:avoid;}th,td{border:.4pt solid #d4deea;padding:1.3mm;vertical-align:top;overflow-wrap:anywhere;}
th{background:#eaf1f9;}tbody tr:nth-child(even){background:#f8fafc;}
table.method-table th:first-child{width:24%;}table.method-table td:first-child{overflow-wrap:normal;}
a{color:#185ba0;text-decoration:none;}code{font-family:"DejaVu Sans Mono",ReportNoto,monospace;font-size:.8em;overflow-wrap:anywhere;}
ul,ol{padding-left:5mm;}li{margin-bottom:1.5mm;}
@media screen{body{max-width:1060px;margin:30px auto;padding:0 24px;font-size:15px;}img{max-height:700px;}table{font-size:13px;}}
'''


PDF_CODE = r'''
import sys,json,hashlib,re
from pathlib import Path
deps,out=sys.argv[1:3]
sys.path.insert(0,deps)
from weasyprint import HTML,CSS
from weasyprint.text.fonts import FontConfiguration
from pypdf import PdfReader
out=Path(out)
rendered=HTML(filename=str(out/'report.html')).render(stylesheets=[CSS(filename=str(out/'report.css'))],font_config=FontConfiguration())
overflow=[]
for i,page in enumerate(rendered.pages,1):
 for box in page._page_box.descendants():
  if type(box).__name__=='TextBox' and (box.position_x<45 or box.position_x+box.width>page.width-45):
   overflow.append(dict(page=i,text=box.text[:60],left=box.position_x,right=box.position_x+box.width))
rendered.write_pdf(out/'report.pdf')
reader=PdfReader(out/'report.pdf')
texts=[p.extract_text() or '' for p in reader.pages]
(out/'pdf_extracted.txt').write_text('\n\n'.join(texts))
metadata=dict(pages=len(texts),bytes=(out/'report.pdf').stat().st_size,text_characters_per_page=[len(t) for t in texts],horizontal_overflows=overflow,pdf_sha256=hashlib.sha256((out/'report.pdf').read_bytes()).hexdigest())
(out/'pdf_validation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
print(json.dumps(metadata,ensure_ascii=False))
if overflow:sys.exit(2)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/course_draft_20261115')
    parser.add_argument('--suite',type=Path,default=ROOT/'runs/course_comparison_20261115')
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/course_comparison_20261115')
    parser.add_argument('--motion-evaluation',type=Path,default=ROOT/'runs/course_motion_evaluation_20261115')
    parser.add_argument('--extended-ppo',type=Path,default=ROOT/'runs/course_ppo_extended_20261115')
    parser.add_argument('--no-pdf',action='store_true')
    parser.add_argument('--pdf-python',default='/home/ubuntu/miniforge3/envs/metaworld_v010/bin/python')
    args=parser.parse_args()
    out=args.output.resolve();suite=args.suite.resolve();dataset=args.dataset.resolve()
    out.mkdir(parents=True,exist_ok=True)
    runs=load_runs(suite)
    plot_curves(runs,out)
    plot_curves(runs,out,ablation=True)
    rows,groups,order_basis=tables_and_results(runs,dataset,out)
    costs=collection_costs()
    save_csv(out/'data_preparation_costs.csv',costs)
    motion_rows,motion_sources=motion_diagnosis(args.motion_evaluation.resolve(),suite,out)
    extended,extended_sources=extended_ppo_comparison(args.extended_ppo.resolve(),suite,runs,out)
    content=assemble(out,suite,dataset,runs,rows,groups,costs,motion_rows,extended)
    (out/'report.md').write_text(content)
    body=markdown.markdown(content,extensions=['tables','fenced_code'])
    body=re.sub(r'<table>(\s*<thead>\s*<tr>\s*<th>方法</th>)',r'<table class="method-table">\1',body)
    # Keep documentation text, but local Markdown links become paths for the HTML reader.
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sesame 强化学习行走实验报告</title><link rel="stylesheet" href="report.css"></head><body>'+body+'</body></html>'
    (out/'report.html').write_text(document)
    (out/'report.css').write_text(CSS)
    missing_images=[]
    for image in re.findall(r'<img[^>]+src="([^"]+)"',document):
        if not (out/html.unescape(image)).exists():missing_images.append(image)
    if missing_images:raise FileNotFoundError(missing_images)
    assert '<video' not in document
    prohibited_fields=('foot_position_rmse_m','foot_tracking_rmse_m','success_rate')
    for f in (out/'supermimic_test_results.csv',out/'supermimic_curve_data.csv'):
        assert not any(s in f.read_text() for s in prohibited_fields),f
    source_files=[Path(__file__).resolve(),out/'baselines/section_baselines.md',out/'section_method_review.md']
    source_files += [p for p in (out/'improvements/section_improvements.md',dataset/'protocol.json',dataset/'dataset_protocol.json') if p.exists()]
    source_files += [r['directory']/name for r in runs for name in ('status.json','curve.jsonl') if (r['directory']/name).exists()]
    source_files += [ROOT/c['source'] for c in costs if c['source']]
    source_files += motion_sources
    source_files += extended_sources
    source_files += sorted(suite.glob('*_gpu_before.csv'))
    manifest=dict(date='2026-11-15',completed=sum(r['status']=='complete' for r in rows),total=7,
                  run_names=[r['method'] for r in rows],test_source_order_basis=order_basis,
                  fresh_motion_reevaluation=bool(motion_rows),
                  extended_ppo_complete=extended is not None,
                  report_sha256=sha(out/'report.md'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  note='Saved records only. No simulation, training, inference, or metric reassignment.')
    (out/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if not args.no_pdf:
        subprocess.run([args.pdf_python,'-c',PDF_CODE,str(ROOT/'.runtime/pdf_export_deps'),str(out)],check=True)
    print(json.dumps(dict(output=str(out),complete=manifest['completed'],total=7),ensure_ascii=False))


if __name__=='__main__':
    main()
