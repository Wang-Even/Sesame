"""Index saved experiments only: no simulation, inference, or training."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/supermimic_results_20261115'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'curves').mkdir(exist_ok=True)


def read(path):
    return json.loads((ROOT / path).read_text())


def relative(path):
    return Path(os.path.relpath(ROOT / path, OUT)).as_posix()


def link(label, path):
    return f'[{label}]({relative(path)})'


SUITES = [
    ('small', '早期小模型（修正版）', 'runs/supermimic_comparison_corrected_20261110'),
    ('compat', '官方算法＋公开 mjlab 兼容版', 'runs/official_comparison_20261111'),
    ('research', '官方研究版 mjlab，2 ms', 'runs/research_comparison_20261112'),
    ('original', '原始参考，0.5 ms', 'runs/agent_reference_comparison_20261113'),
    ('agent32', 'Agent 32 条参考，0.5 ms', 'runs/supermimic_agent32_seed42_20261114'),
]
rows, curves = [], {}
for group, title, base in SUITES:
    if group == 'agent32':
        files = [ROOT / base / 'status.json']
    else:
        files = sorted((ROOT / base).glob('*/result.json' if group == 'small' else '*/status.json'))
    for file in files:
        status = json.loads(file.read_text())
        assert status['status'] in ('complete', 'completed'), file
        folder = file.parent.relative_to(ROOT)
        legacy = group == 'small'
        metrics = status['test' if legacy else 'final_test']
        method = status['algorithm' if legacy else 'method']
        curve_path = folder / ('evaluations.jsonl' if legacy else 'curve.jsonl')
        points = [json.loads(line) for line in (ROOT / curve_path).read_text().splitlines() if line]
        if legacy:
            points = [p for p in points if p.get('label') == 'validation']
        key = f'{group}__{folder.name}'
        protocol = read(folder / 'protocol.json') if (ROOT / folder / 'protocol.json').exists() else {}
        row = dict(
            experiment=key, group=title, method=method, seed=42,
            variant=folder.name, status=status['status'],
            checkpoint='best.pt' if legacy else 'final.pt',
            normalized_return=metrics.get('normalized_return', ''),
            return_value=metrics.get('return', metrics.get('mean_return')),
            heading_rms_deg=metrics['heading_rms_deg'],
            velocity_rmse_m_s=metrics['velocity_rmse_m_s'],
            joint_position_rmse_rad=metrics['joint_position_rmse_rad'],
            root_position_rmse_m=metrics['root_position_rmse_m'],
            local_termination_fraction=metrics['fall_rate'],
            test_episodes=metrics['episodes'],
            training_s=status['training_seconds' if legacy else 'training_s'],
            end_to_end_s=status.get('end_to_end_s', status.get('wall_seconds_before_shutdown')),
            real_steps=status['cumulative_real_steps' if legacy else 'real_steps'],
            imagined_steps=status['imagined_steps'],
            peak_process_memory_mib=status['peak_process_gpu_mib' if legacy else 'peak_process_memory_mib'],
            reward_version=protocol.get('reward_version', 'historical_reward_see_protocol'),
            source=str(file.relative_to(ROOT)), run=str(folder), curve_source=str(curve_path),
            curve_png=f'reports/supermimic_results_20261115/curves/{key}.png',
        )
        rows.append(row)
        curves[key] = [dict(real_steps=p.get('real_steps', p.get('cumulative_real_steps')),
                            training_s=p.get('training_s', p.get('training_seconds')),
                            value=p.get('normalized_return', p.get('mean_return')))
                       for p in points]

counts = {m: sum(r['method'] == m for r in rows) for m in ('supermimic', 'ppo')}
assert counts == {'supermimic': 28, 'ppo': 5}, counts
with (OUT / 'experiments.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
colors = {'supermimic': '#0b56ac', 'ppo': '#c45c27'}
for row in rows:
    points = curves[row['experiment']]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), layout='constrained')
    for ax, field, scale, xlabel in [
        (axes[0], 'real_steps', 1e6, 'Real environment steps (million)'),
        (axes[1], 'training_s', 3600, 'Training time (h)')]:
        ax.plot([p[field] / scale for p in points], [p['value'] for p in points],
                color=colors[row['method']], linewidth=2, marker='.', markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Validation return' if row['normalized_return'] == '' else 'Normalized validation return')
        ax.grid(alpha=.15)
    fig.suptitle(row['experiment'].replace('__', ' / '), fontsize=10)
    fig.savefig(ROOT / row['curve_png'], dpi=150)
    fig.savefig((ROOT / row['curve_png']).with_suffix('.pdf'))
    plt.close(fig)

overview = [
    ('(a) Research mjlab / uniform / 2 ms', ['research__supermimic_seed42', 'research__ppo_seed42']),
    ('(b) Research mjlab / return sampling / 2 ms', ['research__supermimic_return_seed42', 'research__ppo_return_seed42']),
    ('(c) Original references / 0.5 ms', ['original__supermimic_original_seed42', 'original__ppo_original_seed42']),
    ('(d) Agent references: 24 train, 4 val, 4 test', ['agent32__supermimic_agent32_seed42_20261114']),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for ax, (title, keys) in zip(axes.flat, overview):
    for key in keys:
        row = next(r for r in rows if r['experiment'] == key)
        points = curves[key]
        ax.plot([p['training_s'] / 3600 for p in points], [p['value'] for p in points],
                label='SuperMimic' if row['method'] == 'supermimic' else 'PPO',
                color=colors[row['method']], lw=2.3, marker='.', markersize=5)
    ax.set(title=title, xlabel='Training time (h)', ylabel='Normalized validation return', ylim=(0, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
axes[1, 1].text(.97, .1, 'No paired PPO run on this dataset', transform=axes[1, 1].transAxes,
                ha='right', fontsize=9, color='#555555')
fig.savefig(OUT / 'learning_curves_overview.png', dpi=180)
fig.savefig(OUT / 'learning_curves_overview.pdf')
plt.close(fig)

# Videos retain their original paths. Each one is tied to its recorded protocol.
video_specs = [
    ('原始参考：SuperMimic / PPO 策略推理', 'runs/evaluation_videos_20261114/videos/policy_comparison_all8.mp4',
     'runs/evaluation_videos_20261114/protocol.json',
     '0.5 ms 原始参考对照的 final.pt；全部 8 条测试，72 秒。SuperMimic 平均有效前进速度 0.0957 m/s，PPO −0.0012 m/s。',
     'runs/agent_reference_comparison_20261113/learning_curves.png'),
    ('32 条参考训练：策略推理（原规则）', 'runs/supermimic_agent32_seed42_20261114/evaluation_video_20261115/best_policy_all4.mp4',
     'runs/supermimic_agent32_seed42_20261114/evaluation_video_20261115/protocol.json',
     'best.pt；全部 4 条测试，24 秒。这次录像复测回报与训练结束时测试分别保留。',
     'runs/supermimic_agent32_seed42_20261114/validation_curves.png'),
    ('32 条参考训练：同一权重按当前规则复测', 'runs/supermimic_agent32_seed42_20261114/evaluation_no_foot_no_limit_20261115/best_policy_all4.mp4',
     'runs/supermimic_agent32_seed42_20261114/evaluation_no_foot_no_limit_20261115/protocol.json',
     '仍是旧奖励训练的 best.pt；新奖励归一化回报 0.8136、偏航 RMS 4.46°、当前本地检查 4/4。没有重新训练，不能解释为学习提升。', None),
    ('Agent 早期 12 条候选：PD 执行', 'runs/evaluation_videos_20261114/videos/agent_candidates_all12.mp4',
     'runs/evaluation_videos_20261114/protocol.json',
     '全部 12 条，正常与第 1 次随机化执行并排；这是参考采集诊断，不是训练后策略。',
     'runs/agent_reference_comparison_20261113/candidate_validation.png'),
    ('MimicAgent 初试：参考与执行', 'runs/mimicagent_sesame_20261114/reference_vs_execution.mp4',
     'runs/mimicagent_sesame_20261114/video.json', '12 次生成/修订，3 次通过运动学检查，按当时条件入库 0 条。视频只展示协议所选单条。', None),
    ('DeepSeek 扩大采集：参考与执行', 'runs/mimicagent_scale30_20261114/reference_vs_execution.mp4',
     'runs/mimicagent_scale30_20261114/video.json', '30 个候选、111 个生成/修订版本、128 次仿真，按当时条件入库 0 条。视频是单条示例。',
     'runs/mimicagent_scale30_20261114/collection_results.png'),
    ('对话 Agent 30 条：参考与执行', 'runs/codex_agent30_20261114/reference_vs_execution.mp4',
     'runs/codex_agent30_20261114/video.json', '30 个动作程序、120 次仿真，按当时条件入库 0 条；当时速度采样均值有偏差，不能套用为当前规则结果。视频是单条示例。',
     'runs/codex_agent30_20261114/comparison.png'),
    ('32 条入库采集：反馈控制执行', 'runs/agent32_collection_20261114/reference_vs_execution.mp4',
     'runs/agent32_collection_20261114/video.json', '45 个候选、180 次仿真，原规则入库 32 条。视频是固定顺序第一条的参考与反馈＋PD 执行，不是训练后策略。',
     'runs/agent32_collection_20261114/collection_results.png'),
]
videos = []
for title, path, protocol, note, curve in video_specs:
    file = ROOT / path
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'format=duration:stream=width,height,nb_frames', '-of', 'json', str(file)], text=True))
    videos.append(dict(title=title, path=path, protocol=protocol, note=note, curve=curve,
                       duration_s=float(probe['format']['duration']), **probe['streams'][0]))

latest = 'runs/supermimic_agent32_seed42_20261114'
reeval = read(f'{latest}/evaluation_no_foot_no_limit_20261115/evaluation.json')
episode_stats = read(f'{latest}/evaluation_no_foot_no_limit_20261115/protocol.json')['episode_statistics']
with (OUT / 'video_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(videos[0]))
    writer.writeheader()
    writer.writerows(videos)

md = ['# SuperMimic 实验、视频与曲线汇总', '', '更新：2026-11-15。', '',
      '**共 28 次 SuperMimic 完整训练、5 次 PPO 对照，共 33 次，全部为 seed 42。** 这是不同配置的运行数，不是 28 个独立随机种子。', '',
      '最近增加：32 条合格参考的数据集（24 训练／4 验证／4 测试）、1 次 SuperMimic 完整训练，以及删除足端误差与关节限位检查/惩罚后的旧权重复测。按新规则重判已有采集记录后有 41/45 条通过；没有新增采集或按新奖励重新训练。', '',
      '计数排除功能检查、未完成运行和已发现参考末尾奖励错误的旧 SuperMimic；修正版复用的 PPO 只计一次。', '',
      '## 训练结果', '',
      '表中研究版使用 final.pt 的独立测试；早期小模型使用验证集选择的 best.pt，且奖励定义不同。归一化回报不等于成功率。旧本地检查不是课程评分标准，新汇总不列已删除的足端误差指标。', '',
      '| 实验 | 完整运行数 SM / PPO | 主对照测试回报 SM / PPO | 偏航 RMS（°）SM / PPO | 训练时间（分钟）SM / PPO |',
      '|---|---:|---:|---:|---:|']
main_names = {
    'small': ('supermimic_seed42', 'ppo_seed42'),
    'compat': ('supermimic_seed42', 'ppo_seed42'),
    'research': ('supermimic_seed42', 'ppo_seed42'),
    'original': ('supermimic_original_seed42', 'ppo_original_seed42'),
    'agent32': ('supermimic_agent32_seed42_20261114', None),
}
for group, title, _ in SUITES:
    subset = [r for r in rows if r['experiment'].startswith(group+'__')]
    main = [next((r for r in subset if r['variant'] == name), None) for name in main_names[group]]
    def pair(field, digits):
        return ' / '.join('—' if r is None else f'{r[field]:.{digits}f}' for r in main)
    count = ' / '.join(str(sum(r['method'] == method for r in subset)) for method in ('supermimic', 'ppo'))
    returns = pair('return_value' if group == 'small' else 'normalized_return', 4)
    if group == 'small': returns += '（原始回报）'
    times = ' / '.join('—' if r is None else f'{r["training_s"]/60:.2f}' for r in main)
    md.append(f'| {title} | {count} | {returns} | {pair("heading_rms_deg",2)} | {times} |')
md += ['', '研究版 return 采样额外对照：SuperMimic / PPO 测试归一化回报 0.6308 / 0.4565，偏航 RMS 6.84° / 58.25°。两轮消融覆盖参考数量 6/12/24、想象长度 8/16/32/64/128，以及是否使用 Critic；每项结果见完整表。', '',
       '这几组奖励、数据、仿真步长与计时负载存在差别，不能跨组直接排名或计算加速比。研究版 2 ms 和原始参考 0.5 ms 的配对实验中，SuperMimic 回报较高，但训练总用时仍长于 PPO。32 条参考版本没有同数据 PPO 对照。', '',
       '训练时间沿用日志，包含采样和更新，排除评估及环境初始化；各轮编译预热不同，同卡还有其他作业。采集时间另计：最新正式采集 8.45 分钟、训练 4.83 分钟、含初始化/评估的训练进程 6.39 分钟。', '',
       '## 主要学习曲线', '',
       '曲线来自验证集原始记录，没有平滑、跨 seed 误差带或测试集选点。各小图单独比较；最新数据集只画实际完成的 SuperMimic。所有训练曲线仍对应当时的旧奖励。', '',
       '![主要实验学习曲线](learning_curves_overview.png)', '',
       '[曲线 PDF](learning_curves_overview.pdf) · [逐实验指标 CSV](experiments.csv) · [视频索引 CSV](video_manifest.csv)', '',
       '## 重点视频', '']
for video in videos:
    md += [f'### {video["title"]}', '', video['note'], '',
           link(f'视频（{video["duration_s"]:.0f} 秒）', video['path'])+' · '+link('视频协议', video['protocol'])]
    if video['curve']:
        md[-1] += ' · '+link('对应训练曲线' if '策略推理' in video['title'] else '对应实验图', video['curve'])
    md.append('')
md += ['采集图及旧报告保留当时的判定口径，可能包含如今已取消的诊断列；它们属于历史记录。当前汇总及新增曲线不以这些列判定结果。', '',
       '## 旧权重按新规则复测', '',
       '这次视频对应 best.pt（训练到 1,048,576 步时保存）。4 条片段均完整执行，当前本地检查 4/4；不是新训练，也不是课程统一环境成绩。', '',
       '| 测试片段 | 有效前进速度（m/s） | 净运动方向（°，相对 +X） | 末端横向位移（cm） |',
       '|---|---:|---:|---:|']
for i, s in enumerate(episode_stats):
    md.append(f'| {i+1} | {s["forward_speed_from_displacement_m_s"]:.4f} | {s["net_movement_direction_deg"]:+.2f} | {s["lateral_displacement_m"]*100:+.2f} |')
md += ['', '相对参考的机身偏航 RMS 为 4.46°；净运动方向与机头朝向是两个量。速度按重置前 0–5.98 秒的 X 位移/时间计算。新奖励归一化回报 0.8136 不与旧奖励回报直接比较。', '',
       '## 每次训练的指标、曲线和权重', '',
       '每行有按环境步数和训练时间绘制的曲线；表内回报均为独立测试结果。未录制的消融实验没有视频，未用别组视频代替。', '',
       '| 实验目录 | 测试回报 | 偏航 RMS（°） | 训练（分钟） | 曲线 | 权重 | 原记录 |',
       '|---|---:|---:|---:|---|---|---|']
for row in rows:
    value = row['normalized_return'] if row['normalized_return'] != '' else row['return_value']
    md.append(f'| {row["experiment"]} | {value:.4f}{"（原始）" if row["normalized_return"] == "" else ""} | '
              f'{row["heading_rms_deg"]:.2f} | {row["training_s"]/60:.2f} | '
              f'{link("PNG",row["curve_png"])} / {link("PDF",str(Path(row["curve_png"]).with_suffix(".pdf")))} | '
              f'{link(row["checkpoint"], str(Path(row["run"])/row["checkpoint"]))} | {link("JSON",row["source"])} |')
md += ['', '## 已有消融图与缺项', '',
       '- '+link('公开 mjlab 兼容版：六幅消融图', 'runs/official_comparison_20261111/figures/six_panels.pdf'),
       '- '+link('官方研究版：六幅消融图', 'runs/research_comparison_20261112/figures/six_panels.pdf'),
       '- 尚缺：当前新奖励完整训练；32 条参考的同条件 PPO；多 seed 重复；消融配置各自的策略录像。',
       '- 41 条重判参考尚未建立新训练划分或用于新一轮训练。',
       '- '+link('已作废旧 SuperMimic 的原因', 'runs/supermimic_comparison_20261110/known_issue.json')+'：想象轨迹到达参考末尾被错误扣除摔倒惩罚；已由修正版取代。', '',
       '本页由保存的 JSON/JSONL 和视频元数据生成；不运行仿真，不改动权重和历史指标。重新整理：`python3 scripts/build_supermimic_results_index.py`。', '']
text = '\n'.join(md)
(OUT / 'RESULTS.md').write_text(text)
body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
for video in videos:
    href = relative(video['path'])
    body = body.replace(f'<a href="{href}">',
        f'<video controls preload="none" src="{html.escape(href)}"></video><br><a href="{href}">')
style = '''body{font-family:system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 24px;color:#172638;line-height:1.7}h1,h2,h3{line-height:1.35}h2{margin-top:42px;border-bottom:1px solid #ddd;padding-bottom:10px}a{color:#125caa}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px}td,th{border:1px solid #d8dee5;padding:8px 12px;white-space:nowrap}th{background:#eef3f8}img,video{display:block;max-width:100%;max-height:720px;margin:16px 0}video{width:900px;background:#111}code{font-size:.85em}'''
(OUT / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SuperMimic 实验结果</title><style>'+style+'</style><body>'+body+'</body></html>')
manifest = dict(date='2026-11-15', counts=counts, experiments=rows, videos=videos,
                re_evaluation=reeval, curve_values=curves,
                source_sha256={r['source']: hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest() for r in rows})
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(dict(output=str(OUT), counts=counts, videos=len(videos), curves=len(rows)), ensure_ascii=False))
