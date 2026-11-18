"""Build the completed dense-evaluation report from recorded scalar results.

This script performs file/metadata operations only. It neither trains a model
nor runs inference. The previous report and experiment directories are inputs.
Figures and LaTeX are produced separately by the parameterized plotting/export
scripts after this builder succeeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = '2026-11-16'
BASE_STEPS = 2_097_152
EXTENDED_STEPS = 8_388_608
EVAL_EVERY = 32_768
METHODS = (
    ('ppo', 'PPO'),
    ('supermimic', 'SuperMimic（官方代码）'),
    ('supermimic_deepseek', 'SuperMimic（DeepSeek）'),
    ('supermimic_codex', 'SuperMimic（Codex）'),
    ('supermimic_h16', 'H=16，有 Critic'),
    ('supermimic_h64', 'H=64，有 Critic'),
    ('supermimic_no_critic', 'H=32，无 Critic'),
)
LABELS = dict(METHODS, ppo_extended='PPO（延长预算）')
TIMING_FIELDS = (
    'startup_training_s', 'post_startup_training_s', 'evaluation_s',
    'cumulative_evaluation_s', 'evaluation_trigger', 'evaluation_index',
)
CURVE_FIELDS = (
    'method', 'real_steps', 'imagined_steps', 'training_s', 'return',
    'velocity_rmse_m_s', 'heading_rms_deg', 'root_position_rmse_m',
    *TIMING_FIELDS,
)
TABLE_COLUMNS = (2, 5, 3, 3, 9, 9, 3, 4, 4, 5, 4, 6, 5, 7, 6, 4)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def number(value, digits=3) -> str:
    return '—' if value is None else f'{float(value):.{digits}f}'


def minutes(value) -> str:
    return number(float(value) / 60, 2)


def save_csv(path: Path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('Table row shape differs from header')
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']
    lines.extend('| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows)
    return '\n'.join(lines)


def replace_paragraph(text: str, prefix: str, replacement: str) -> str:
    paragraphs = text.split('\n\n')
    matches = [i for i, paragraph in enumerate(paragraphs) if paragraph.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one template paragraph starting with {prefix!r}')
    paragraphs[matches[0]] = replacement
    return '\n\n'.join(paragraphs)


def readiness(args):
    states = {}
    for name, _ in METHODS:
        path = args.suite / name / 'status.json'
        states[name] = read_json(path).get('status', 'unknown') if path.is_file() else 'missing'
    for name, folder in (('ppo_extended', args.extended_ppo), ('motion_evaluation', args.motion_evaluation)):
        path = folder / 'status.json'
        states[name] = read_json(path).get('status', 'unknown') if path.is_file() else 'missing'
    return {'ready': all(state == 'complete' for state in states.values()), 'states': states}


def checked_run(directory: Path, name: str, budget: int):
    status = read_json(directory / 'status.json')
    protocol = read_json(directory / 'protocol.json')
    curve = read_lines(directory / 'curve.jsonl')
    updates = read_lines(directory / 'updates.jsonl')
    startup = read_json(directory / 'startup.json')
    settings = protocol['args']
    if status['status'] != 'complete' or not (directory / 'final.pt').is_file():
        raise ValueError(f'{name}: completed final.pt required')
    required = dict(steps=budget, eval_every=EVAL_EVERY, eval_steps=300,
                    physics_dt=.0005, seed=42, envs=512)
    if any(settings.get(k) != v for k, v in required.items()):
        raise ValueError(f'{name}: experiment settings differ from dense-run protocol')
    if status['real_steps'] != budget:
        raise ValueError(f'{name}: incomplete training budget')
    metrics = status['final_test']
    if metrics['episodes'] != 12 or metrics['steps_per_episode'] != 300 or len(metrics['episode_returns']) != 12:
        raise ValueError(f'{name}: final-test scope differs')
    for key in ('return', 'velocity_rmse_m_s', 'heading_rms_deg', 'root_position_rmse_m', 'fall_rate'):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f'{name}: nonfinite final metric {key}')
    if not curve or curve[0]['real_steps'] != 0 or curve[-1]['real_steps'] != budget:
        raise ValueError(f'{name}: curve endpoints differ from recorded budget')
    if len(curve) != budget // EVAL_EVERY + 2:
        raise ValueError(f'{name}: expected initial, first-update, and all periodic/final evaluations')
    if any(a['real_steps'] >= b['real_steps'] or a['training_s'] >= b['training_s']
           for a, b in zip(curve, curve[1:])):
        raise ValueError(f'{name}: nonmonotonic evaluation steps/time')
    if not math.isclose(curve[-1]['training_s'], status['training_s'], rel_tol=1e-9):
        raise ValueError(f'{name}: final training clocks disagree')
    initial = curve[0]
    if initial['evaluation_trigger'] != 'initial' or initial['post_startup_training_s'] is not None:
        raise ValueError(f'{name}: initial evaluation must remain outside displayed training')
    shown = [row for row in curve if row['post_startup_training_s'] is not None]
    first = shown[0]
    if first['evaluation_trigger'] != 'first_update' or abs(first['post_startup_training_s']) > 1e-9:
        raise ValueError(f'{name}: first displayed observation must be a real first-update evaluation at zero')
    origin = float(startup['training_s'])
    if not math.isclose(origin, float(updates[0]['training_s']), abs_tol=1e-8):
        raise ValueError(f'{name}: startup boundary differs from first update')
    if not math.isclose(origin, float(status['startup_training_s']), abs_tol=1e-8):
        raise ValueError(f'{name}: startup status clock differs')
    for row in curve:
        if not math.isfinite(float(row['return'])) or not math.isfinite(float(row['evaluation_s'])):
            raise ValueError(f'{name}: nonfinite evaluation record')
        if row['evaluation_s'] < 0:
            raise ValueError(f'{name}: negative evaluation duration')
    for row in shown:
        if not math.isclose(row['training_s'] - origin, row['post_startup_training_s'], abs_tol=1e-6):
            raise ValueError(f'{name}: post-startup clock differs from recorded offset')
    if not 0 < status['training_s'] <= status['end_to_end_s']:
        raise ValueError(f'{name}: invalid inclusive/end-to-end training clocks')
    return dict(name=name, label=LABELS[name], directory=directory, status=status,
                protocol=protocol, curve=curve, shown=shown, startup=startup,
                origin=origin, metrics=metrics)


def checked_inputs(args):
    ready = readiness(args)
    if not ready['ready']:
        raise RuntimeError('Completed experiments required: ' + json.dumps(ready['states'], ensure_ascii=False))
    suite_protocol = read_json(args.suite / 'protocol.json')
    dataset = Path(suite_protocol['dataset']).resolve()
    runs = [checked_run(args.suite / name, name, BASE_STEPS) for name, _ in METHODS]
    extended = checked_run(args.extended_ppo, 'ppo_extended', EXTENDED_STEPS)
    common = runs[0]['protocol']
    for run in [*runs, extended]:
        p = run['protocol']
        for key in ('reward_version', 'official_commit', 'mjlab_commit'):
            if p[key] != common[key]:
                raise ValueError(f'{run["name"]}: shared method source/configuration differs: {key}')
        for key in ('validation_file', 'test_file'):
            if Path(p['args'][key]).resolve() != Path(common['args'][key]).resolve():
                raise ValueError(f'{run["name"]}: held-out reference split differs')
    for key, value in common['args'].items():
        if key not in ('output', 'steps') and extended['protocol']['args'].get(key) != value:
            raise ValueError(f'Extended PPO differs in {key}')
    for key in ('ppo_config', 'runner_config'):
        if extended['protocol'][key] != common[key]:
            raise ValueError(f'Extended PPO differs in {key}')
    if read_json(args.extended_ppo / 'initial_actor.json')['parameter_sha256'] != read_json(args.suite / 'ppo/initial_actor.json')['parameter_sha256']:
        raise ValueError('Extended PPO initialization differs')
    return runs, extended, dataset


def time_window(runs, extended, field):
    # The comparison uses actual observations; no time-interpolated rewards.
    models = [extended, *runs[1:4]]
    sm_end = max(run['shown'][-1][field] for run in runs[1:4])
    end = min(sm_end, extended['shown'][-1][field])
    result = []
    for run in models:
        available = [row for row in run['shown'] if row[field] <= end + 1e-8]
        if not available:
            raise ValueError(f'{run["name"]}: no observed score in common {field} window')
        point = available[-1]
        result.append(dict(method=run['name'], label=run['label'], window_end_s=end,
                           actual_evaluation_s=point[field], real_steps=point['real_steps'],
                           validation_return=point['return']))
    return dict(rows=result, cutoff=end, full_supermimic_window=sm_end <= extended['shown'][-1][field], field=field)


def result_table(rows):
    return table(['方法', '测试 Return', '速度 RMSE（m/s）', '偏航 RMS（°）', '机身 RMSE（cm）'],
                 [[r['label'], number(r['return_value']), number(r['velocity_rmse_m_s'], 4),
                   number(r['heading_rms_deg'], 2), number(r['root_position_rmse_m'] * 100, 2)] for r in rows])


def difference(value, digits=3):
    if value == 0:
        return '相同'
    return ('高 ' if value > 0 else '低 ') + number(abs(value), digits)


def source_analysis(groups):
    values = {(row['method'], row['source']): row['mean_return'] for row in groups}
    labels = dict(original='原始参考', deepseek='DeepSeek 参考', codex='Codex 参考')
    sentences = []
    for method in ('supermimic_deepseek', 'supermimic_codex'):
        deltas = [f'{label}上{values[(method, source)] - values[("supermimic", source)]:+.3f}'
                  for source, label in labels.items()]
        sentences.append(f'{LABELS[method]} 相对官方代码组的回报差分别为' + '、'.join(deltas) + '。')
    return ''.join(sentences) + '参考动作分布与 Isaac / MuJoCo 动力学一致性同时变化，该对照无法分离二者及语言模型能力的独立贡献。'


def make_report(template, runs, extended, rows, groups, motion_rows, total_window, active_window, args, dataset):
    first, official = rows[:2]
    delta = official['return_value'] - first['return_value']
    ratio = official['training_s'] / first['training_s']
    total_ppo, total_sm = total_window['rows'][:2]
    total_delta = total_sm['validation_return'] - total_ppo['validation_return']
    front = template.split('## 6.', 1)[0].rstrip()
    abstract = ('**摘要**：本文以 Sesame 小型四足机器人为对象，研究强化学习行走控制中的速度跟踪、动作平滑性与训练效率。'
                '在 Isaac 平地行走任务中比较 DQN、PPO 和 SAC，并通过动作差惩罚与目标速度对照分析运动表现；'
                '在 MuJoCo 参考轨迹跟踪任务中比较 PPO、基于官方代码适配的 SuperMimic，以及使用 DeepSeek 和 Codex 生成参考轨迹的 SuperMimic。'
                '实验结果表明，PPO 在域随机化评估下取得最高的基线回报，增大动作差惩罚系数降低了动作信号变化率，'
                '但实际关节加速度的改善依赖评估条件。'
                f'参考跟踪任务在相同真实交互步数下，官方代码 SuperMimic 的测试回报较 PPO {difference(delta)}，'
                f'包含启动开销的训练时间为 PPO 的 {ratio:.2f} 倍。'
                f'在含启动开销的 {minutes(total_window["cutoff"])} 分钟观测窗口内，'
                f'官方代码 SuperMimic 的验证回报较延长预算 PPO {difference(total_delta)}。')
    front = replace_paragraph(front, '**摘要**：', abstract)
    method = template[template.index('## 6.'):template.index('### 6.5')].rstrip()
    # Keep method, data provenance, metric definitions and reward/network tables.
    method = replace_paragraph(method, '学习曲线纵轴为验证轨迹回报',
        '学习曲线纵轴为验证轨迹回报 `Return = Σr_t`，横轴为正式训练时间。'
        '各组在首次有效参数更新后立即进行实测评估，以该时刻作为时间零点；'
        f'此后按每 {EVAL_EVERY:,} 个真实环境步的全局边界评估，训练预算终点另行确保评估。'
        f'主实验每组保留 {len(runs[0]["shown"])} 个训练后评估点，延长预算 PPO 保留 {len(extended["shown"])} 个点，'
        '相较每 262,144 步评估的设置，周期采样频率提高 8 倍。'
        '横轴排除首次更新之前的启动时段以及验证、测试和保存耗时；评估点之间的实际训练时间保留。'
        '零步初始评估保存在原始记录中，不绘入正式训练曲线。连续实线连接已有评估记录的 EMA 值，'
        '平滑半衰期为 131072 个真实交互步。验证从参考轨迹第一帧开始，执行完整 6 秒；'
        '最终测试统一采用 `final.pt`。6 秒正奖励上限为 30，对应归一化回报 `Return/30`。')
    method = replace_paragraph(method, '表中训练时间包括仿真采样',
        '表中训练时间包括仿真采样、模型与策略更新及首次编译，不含初始化、验证、测试和保存。'
        '该统计保留首次更新之前的累计用时，区别于曲线的正式训练时间。'
        '端到端运行时间与参考数据准备成本分别统计。各组在同一 GPU 上串行运行，'
        '每次训练采用独立的 Inductor/Triton 编译缓存。评估使用独立的随机数上下文，'
        '并在结束后恢复训练随机数状态，避免评估消耗训练所用的随机数流。')
    common_caption = ('横轴为扣除启动和评估耗时后的正式训练时间，首次有效更新后的实测评估位于零点。'
                      f'周期评估间隔为 {EVAL_EVERY:,} 个真实环境步。连续实线展示因果 EMA，'
                      '半衰期为 131072 步，不绘制散点或虚线。各组采用训练种子 42，无跨种子误差带。')
    parts = [front, method, '### 6.5 主对照结果',
             '![四组同设置学习曲线](supermimic_comparison.png)',
             '图 7：四种方法在共同验证集上的回报与运动误差。' + common_caption,
             result_table(rows[:4])]
    best = max(rows[:4], key=lambda row: row['return_value'])
    parts.append(f'相同真实环境步预算下，{best["label"]} 的测试回报最高，为 {best["return_value"]:.3f}。'
                 f'采用原始参考的官方代码 SuperMimic 相比 PPO 的测试回报{difference(delta)}，'
                 f'包含启动开销的训练时间为 PPO 的 {ratio:.2f} 倍。'
                 '该比较描述固定预算下的回报与计算开销，达到同一目标回报的时间比不由该结果直接确定。'
                 '延长 PPO 预算的时间对照见第 6.8 节。两个 Agent 组同时改变训练参考分布，其结果包含数据来源的影响。')
    parts.append('为分析参考来源的影响，将 `final.pt` 在 12 条保留测试参考上的结果按来源分别汇总。同一步态家族内的变体测试不代表跨技能泛化。')
    grouped = {(g['method'], g['source']): g['mean_return'] for g in groups}
    parts.append(table(['方法', '原始参考 Return', 'DeepSeek 参考 Return', 'Codex 参考 Return'],
                       [[row['label'], *(number(grouped[(row['method'], source)]) for source in ('original', 'deepseek', 'codex'))]
                        for row in rows[:4]]))
    parts.append(source_analysis(groups))
    parts.extend(['**实际运动表现**',
        '四个主对照组采用各 2,097,152 步训练预算结束时的 `final.pt`，从相同初始状态独立执行全部 12 条测试轨迹。'
        '该运动评估与训练结束测试为不同执行批次，分别报告结果。完整片段的实际前进速度定义为世界 X 净位移 / 5.98 秒，'
        '提前终止片段使用有效时长；参考速度采用相同时间段。绝对横向位移为各片段最终世界 Y 位移绝对值的平均值。'
        '运动方向偏差为净位移方向相对世界 +X 的绝对夹角，区别于机头偏航角。'])
    parts.append(table(['方法', '实际前进速度（m/s）', '参考速度（m/s）', '绝对横向位移（cm）', '运动方向偏差（°）', '完整执行'],
        [[r['label'], number(r['mean_forward_speed_from_displacement_m_s'], 4),
          number(r['mean_reference_forward_speed_from_displacement_m_s'], 4),
          number(None if r['mean_abs_lateral_displacement_m'] is None else r['mean_abs_lateral_displacement_m'] * 100, 2),
          number(r['mean_abs_net_movement_direction_deg'], 2), f'{r["complete_episodes"]}/{r["episodes"]}'] for r in motion_rows]))
    speeds = [r['mean_forward_speed_from_displacement_m_s'] for r in motion_rows[1:]]
    refs = [r['mean_reference_forward_speed_from_displacement_m_s'] for r in motion_rows[1:]]
    if all(v is not None for v in speeds + refs):
        reference_text = (f'统一参考速度为 {refs[0]:.3f} m/s。'
                          if max(refs) - min(refs) < 1e-8 else
                          f'对应参考速度为 {min(refs):.3f}–{max(refs):.3f} m/s。')
        parts.append(f'三组 SuperMimic 的实际前进速度为 {min(speeds):.3f}–{max(speeds):.3f} m/s，'
                     + reference_text +
                     ('三组实际速度均低于参考速度，仍存在前进速度跟踪不足。' if all(v < r for v, r in zip(speeds, refs)) else '') +
                     '回报、实际速度与运动方向分别衡量不同的执行特征。')
    parts.append('完整执行数统计未触发任务提前终止条件的片段，不另设成功率门槛。'
                 '接近静止时，净位移方向对微小漂移敏感，需结合实际速度解释；净位移小于 1 μm 时，方向未定义。'
                 '有效回合数与初态一致性记录于 `motion_reevaluation.csv`，分来源运动指标记录于 `motion_reevaluation_by_source.csv`。')
    ablations = [dict(rows[i], label=('H=32，有 Critic' if i == 1 else rows[i]['label'])) for i in (1, 4, 5, 6)]
    parts.extend(['### 6.6 想象长度与 Critic 消融',
        '![想象长度和 Critic 消融](supermimic_ablation.png)',
        '图 8：原始训练参考下 H=16、32、64 及 H=32 无 Critic 的消融。真实步预算、奖励和评估集一致；'
        '不同 H 对应不同的想象步数和计算量。' + common_caption,
        result_table(ablations)])
    best_h = max(ablations, key=lambda row: row['return_value'])
    critic_delta = rows[6]['return_value'] - rows[1]['return_value']
    parts.append(f'{best_h["label"]} 的测试回报最高，为 {best_h["return_value"]:.3f}；'
                 f'移除 Critic 后，回报变化为 {critic_delta:+.3f}。该消融结论的适用范围为所测参考动作、训练预算和训练种子。')
    parts.extend(['### 6.7 计算与数据准备成本', table(
        ['方法', '训练（分钟）', '端到端运行（分钟）', '真实步数', '想象步数', '提前终止', '采样显存峰值（GiB）'],
        [[r['label'], minutes(r['training_s']), minutes(r['end_to_end_s']), r['real_steps'], r['imagined_steps'],
          f'{round(r["local_termination_fraction"] * 12)}/12', number(r['peak_process_memory_mib'] / 1024, 2)] for r in rows]),
        '表中训练时间采用同步后的采样与更新计时，并保留启动阶段；端到端运行时间包含初始化、评估和保存。'
        '增加评估频率会增加端到端运行时间，其耗时未计入学习曲线的正式训练时间。'
        '提前终止数依据任务终止条件计算，与跌倒率不同。显存统计为进程离散采样观测到的最大值，时间分辨率由监测间隔限定。'])
    t0 = runs[1]['origin']
    active = runs[1]['status']['post_startup_training_s']
    parts.append(f'官方代码 SuperMimic 的首次有效更新发生在累计训练 {t0:.1f} 秒，'
                 f'占含启动训练时间的 {100 * t0 / rows[1]["training_s"]:.1f}%；其后正式训练累计 {active:.1f} 秒。'
                 '启动区间包含首次编译、数据积累及更新，未将其全部解释为编译时间。'
                 '完整计时记录保留在 `rerun_timing_summary.csv`，原论文大规模任务的加速比不直接适用于本实验。')
    snapshots = [args.suite / f'{name}_gpu_before.csv' for name, _ in METHODS]
    available = [p for p in snapshots if p.is_file()]
    occupied = sum(bool(list(csv.DictReader(p.read_text().splitlines()))) for p in available)
    parts.append(f'{len(available)} 次训练启动前 GPU 快照中，{occupied} 次列出了其他 CUDA 计算进程。该记录覆盖启动时刻，不构成整个训练期间的独占监测。')
    # Reference generation was not rerun: retain the already measured cost table and its scope.
    old_cost_section = template[template.index('| 数据准备阶段 |'):template.index('### 6.8')].strip()
    parts.append('训练继续使用已划分的参考数据；下列参考准备成本对应原始采集阶段，不计为本次重复训练的新采集开销。')
    parts.append(old_cost_section)
    parts.extend(['### 6.8 延长 PPO 预算的训练时间对照',
        '为比较相近训练时间内的策略质量，独立训练真实环境步预算为 8,388,608 的 PPO。'
        '除步数预算外，训练种子 42、数据、网络、超参数、评估间隔和独立编译缓存规则均与原预算 PPO 保持一致。',
        '![延长 PPO 预算的训练时间对照](ppo_extended_time_comparison.png)',
        '图 9：延长预算 PPO 与三组 SuperMimic 的验证回报。' + common_caption +
        '各曲线止于各自训练预算；下表的总时间对照使用包含启动开销的原始训练计时。'])
    scope = ('三组 SuperMimic 的最大含启动训练时间' if total_window['full_supermimic_window'] else
             '三组 SuperMimic 最大含启动训练时间与延长 PPO 已记录训练时长的较小值')
    parts.append(f'以{scope}为窗口上限，在 0–{minutes(total_window["cutoff"])} 分钟内取各组最后一次实测验证结果。'
                 '评估时刻分别保留，不通过曲线插值构造同步分数；该表的时间口径与图中扣除启动阶段的横坐标分别报告。')
    parts.append(table(['方法', '实际评估时间（分钟）', '已用真实步数', '验证 Return'],
        [[r['label'], minutes(r['actual_evaluation_s']), r['real_steps'], number(r['validation_return'])] for r in total_window['rows']]))
    parts.append(f'使用相同原始训练参考的官方代码 SuperMimic，验证回报较延长预算 PPO {difference(total_delta)}，'
                 f'两次实测时刻相差 {abs(total_sm["actual_evaluation_s"] - total_ppo["actual_evaluation_s"]):.1f} 秒，'
                 f'对应真实环境步数分别为 {total_sm["real_steps"]:,} 和 {total_ppo["real_steps"]:,}。'
                 '该结果描述给定配置及观测窗口内的回报差异与累计样本量。')
    active_ppo, active_sm = active_window['rows'][:2]
    active_delta = active_sm['validation_return'] - active_ppo['validation_return']
    parts.append(f'按图中的正式训练时间口径，在 0–{minutes(active_window["cutoff"])} 分钟窗口内，'
                 f'官方代码 SuperMimic 与延长预算 PPO 最后一次实测验证回报分别为 {active_sm["validation_return"]:.3f} 和 {active_ppo["validation_return"]:.3f}，'
                 f'差值为 {active_delta:+.3f}，两次评估时间相差 {abs(active_sm["actual_evaluation_s"] - active_ppo["actual_evaluation_s"]):.1f} 秒。'
                 '该窗口的逐方法数据记录于 `ppo_extended_post_startup_time_points.csv`。')
    es = extended['status']
    parts.append(f'延长预算 PPO 包含启动开销的最终训练时间为 {minutes(es["training_s"])} 分钟，'
                 f'端到端运行时间为 {minutes(es["end_to_end_s"])} 分钟，最终测试回报为 {es["final_test"]["return"]:.3f}。'
                 '所有比较采用单个训练种子和固定超参数；密集时间采样提高曲线分辨率，不增加独立训练重复次数，'
                 '也不构成跨种子稳定性或显著性证据。')
    parts.extend(['## 附录：实验可复现性',
        f'主实验的协议、源代码快照、原始曲线、更新记录及最终权重保存在 `{relative(args.suite)}/`。'
        '延长预算 PPO、运动复测、参考数据和结果表的路径与校验值见 `report_manifest.json`。'
        '第 1–5 节沿用原实验记录，第 6 节对应独立重新训练与运动复测。'
        '网络初始化和官方代码版本分别通过参数与源文件哈希核验。'
        '[课程要求原文](../../docs/course_requirements.txt)'])
    return '\n\n'.join(parts).strip() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT / 'runs/course_comparison_dense_20261116')
    parser.add_argument('--extended-ppo', type=Path, default=ROOT / 'runs/course_ppo_extended_dense_20261116')
    parser.add_argument('--motion-evaluation', type=Path, default=ROOT / 'runs/course_motion_evaluation_dense_20261116')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/course_rerun_20261116')
    parser.add_argument('--template', type=Path, default=ROOT / 'reports/course_draft_20261115')
    parser.add_argument('--check-config', action='store_true', help='Read completion/configuration metadata only; never write report files')
    args = parser.parse_args()
    for key in ('suite', 'extended_ppo', 'motion_evaluation', 'output', 'template'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output == args.template or args.output.is_relative_to(args.template):
        raise ValueError('Output must be independent of the template report')
    if not (args.template / 'report.md').is_file():
        raise FileNotFoundError(args.template / 'report.md')
    ready = readiness(args)
    if args.check_config and not ready['ready']:
        print(json.dumps(dict(**ready, output=relative(args.output), report_written=False), ensure_ascii=False))
        return
    runs, extended, dataset = checked_inputs(args)
    import build_course_draft as legacy
    legacy_runs = legacy.load_runs(args.suite)
    # Validate the fresh motion records before creating any report artifacts.
    motion_protocol = read_json(args.motion_evaluation / 'protocol.json')
    if Path(motion_protocol['suite']).resolve() != args.suite or motion_protocol['suite_protocol_sha256'] != digest(args.suite / 'protocol.json'):
        raise ValueError('Motion evaluation belongs to a different training suite')
    for run in runs[:4]:
        model = motion_protocol['models'][run['name']]
        if model['checkpoint_sha256'] != digest(run['directory'] / 'final.pt'):
            raise ValueError(f'{run["name"]}: motion checkpoint differs')
    if args.check_config:
        print(json.dumps(dict(**ready, config_valid=True, output=relative(args.output), report_written=False), ensure_ascii=False))
        return
    existing = args.output / 'report_manifest.json'
    if (args.output / 'report.md').exists() and (not existing.exists() or read_json(existing).get('generator') != 'scripts/build_course_rerun_report.py'):
        raise FileExistsError('Refusing to replace an unrelated report')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, groups, order_basis = legacy.tables_and_results(legacy_runs, dataset, args.output)
    motion_rows, motion_sources = legacy.motion_diagnosis(args.motion_evaluation, args.suite, args.output)
    if len(motion_rows) != 4:
        raise ValueError('Four complete fresh motion evaluations required')
    costs = legacy.collection_costs()
    cost_labels = {'DeepSeek 前一轮探索（中止）': 'DeepSeek 探索采集（中止）',
                   'Codex 当前规则重判': 'Codex 统一验收规则复核'}
    costs = [dict(row, label=cost_labels.get(row['label'], row['label'])) for row in costs]
    legacy.save_csv(args.output / 'data_preparation_costs.csv', costs)
    total_window = time_window(runs, extended, 'training_s')
    active_window = time_window(runs, extended, 'post_startup_training_s')
    save_csv(args.output / 'ppo_extended_time_points.csv', total_window['rows'])
    save_csv(args.output / 'ppo_extended_post_startup_time_points.csv', active_window['rows'])
    curve_rows = [dict(method=run['name'], **point) for run in runs for point in run['curve']]
    save_csv(args.output / 'supermimic_curve_data.csv', curve_rows, CURVE_FIELDS)
    extended_rows = [dict(method=run['name'], validation_return=point['return'], **point)
                     for run in [extended, *runs[1:4]] for point in run['curve']]
    save_csv(args.output / 'ppo_extended_curve_data.csv', extended_rows,
             ['method', 'real_steps', 'training_s', 'validation_return', *TIMING_FIELDS])
    es = extended['status']
    save_csv(args.output / 'ppo_extended_final_result.csv', [dict(method='ppo_extended', real_steps=es['real_steps'],
        training_s=es['training_s'], post_startup_training_s=es['post_startup_training_s'], end_to_end_s=es['end_to_end_s'],
        test_return=es['final_test']['return'], checkpoint_sha256=digest(args.extended_ppo / 'final.pt'))])
    timings = [dict(method=run['name'], label=run['label'], startup_training_s=run['origin'],
                   training_s=run['status']['training_s'], post_startup_training_s=run['status']['post_startup_training_s'],
                   cumulative_evaluation_s=run['status']['cumulative_evaluation_s'], checkpoint_s=run['status']['checkpoint_s'],
                   end_to_end_s=run['status']['end_to_end_s'], evaluations=len(run['curve']), displayed_evaluations=len(run['shown']))
               for run in [*runs, extended]]
    save_csv(args.output / 'rerun_timing_summary.csv', timings)
    template = (args.template / 'report.md').read_text()
    content = make_report(template, runs, extended, rows, groups, motion_rows, total_window, active_window, args, dataset)
    tables = re.findall(r'(?:^\|.*\|\n)+', content, re.MULTILINE)
    columns = tuple(len(block.splitlines()[0].strip().strip('|').split('|')) for block in tables)
    if columns != TABLE_COLUMNS:
        raise ValueError(f'LaTeX table layout differs: {columns}')
    figures = re.findall(r'^图 (\d+)：', content, re.MULTILINE)
    if figures != [str(i) for i in range(1, 10)]:
        raise ValueError(f'Expected figures 1–9: {figures}')
    for folder in ('baselines', 'improvements'):
        shutil.copytree(args.template / folder, args.output / folder, dirs_exist_ok=True)
    for name in ('method_flow.png', 'method_flow.pdf', 'report.css'):
        shutil.copy2(args.template / name, args.output / name)
    (args.output / 'report.md').write_text(content)
    (args.output / 'section_method_review.md').write_text(content[content.index('## 6.'):content.index('### 6.4')].strip() + '\n')
    source_files = [Path(__file__), Path(legacy.__file__), args.template / 'report.md',
                    args.suite / 'protocol.json', dataset / 'protocol.json']
    source_files += [run['directory'] / name for run in [*runs, extended]
                     for name in ('protocol.json', 'status.json', 'curve.jsonl', 'updates.jsonl', 'startup.json', 'initial_actor.json')]
    source_files += motion_sources
    source_files += [ROOT / c['source'] for c in costs if c.get('source')]
    source_files += [p for p in args.template.glob('baselines/*') if p.is_file()]
    source_files += [p for p in args.template.glob('improvements/*') if p.is_file()]
    source_files += sorted(args.suite.glob('*_gpu_before.csv'))
    manifest = dict(date=REPORT_DATE, generator='scripts/build_course_rerun_report.py', completed=8, total=8,
        run_names=[run['name'] for run in [*runs, extended]], suite=relative(args.suite),
        extended_ppo=relative(args.extended_ppo), motion_evaluation=relative(args.motion_evaluation),
        inherited_sections='1–5 from ' + relative(args.template), test_source_order_basis=order_basis,
        fresh_motion_reevaluation=True, extended_ppo_complete=True, report_sha256=digest(args.output / 'report.md'),
        source_sha256={relative(p): digest(p) for p in sorted(set(source_files))},
        evaluation_interval_real_steps=EVAL_EVERY, evaluations=timings,
        total_time_window=total_window, post_startup_time_window=active_window,
        table_count=16, figure_count=9,
        note='Completed recorded reruns only. Curves start at a measured first-update evaluation; raw initial evaluations retained. Rendering and LaTeX export are separate steps.')
    (args.output / 'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), complete=8, tables=16, figures=9,
                          motion_groups=len(motion_rows), report_written=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
