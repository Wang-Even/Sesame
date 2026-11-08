#!/usr/bin/env python3
"""Analyze saved Dyna-PPO metadata and figures, without importing a model or simulator."""
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

GROUPS = ('baseline', 'full_dr', 'dyna_ppo')
LABELS = {'baseline': '固定参数 PPO', 'full_dr': '完整域随机化 PPO', 'dyna_ppo': 'Dyna-PPO'}
COLORS = {'baseline': '#2563EB', 'full_dr': '#E07A1F', 'dyna_ppo': '#7C3AED'}
CONDITIONS = {'nominal': '标称物理', 'randomized': '随机物理'}
METRICS = {
    'velocity_x_rmse_m_s': '瞬时前向速度 RMSE（m/s）',
    'velocity_xy_tracking_rmse_m_s': '瞬时平面速度 RMSE（m/s）',
    'distance_x_m': '世界 x 位移（m）',
    'abs_final_heading_yaw_deg': '末端绝对航向角（°）',
    'abs_lateral_displacement_m': '末端绝对横向位移（m）',
    'average_speed_abs_error_m_s': '回合平均速度绝对误差（m/s）',
    'return': '本地评测回报',
    'fall_rate': '记录跌倒比例',
    'duration_s': '回合时长（s）',
    'applied_target_rate_rms_rad_s': '施加目标角速度 RMS（rad/s）',
    'applied_target_acceleration_rms_rad_s2': '施加目标角加速度 RMS（rad/s²）',
    'actual_joint_acceleration_rms_rad_s2': '实际关节角加速度 RMS，50 Hz（rad/s²）',
    'raw_action_clipped_fraction': '原始动作裁剪比例',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, inputs, lines=False):
    raw = Path(path).read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if lines else json.loads(raw)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    values = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite saved metric')
    return {'mean': statistics.mean(values) if values else None,
            'sd': statistics.stdev(values) if len(values)>1 else None, 'count': len(values)}


def interpret_results(groups, complete):
    """Generate descriptive findings from existing aggregates, without altering them."""
    if not complete:
        return {'paragraphs':['评测尚未完整，暂不给出正式策略优劣结论。'],
                'descriptive_near_tie_relative_threshold':.01}
    primary='velocity_x_rmse_m_s'
    random=groups['randomized'];nominal=groups['nominal']
    dyna,reference,baseline=(random[g][primary] for g in ('dyna_ppo','full_dr','baseline'))
    delta=dyna['mean']-reference['mean']
    relative=100*delta/reference['mean']
    def fmt(point,digits=6):
        return f"{point['mean']:.{digits}f}"+(f" ± {point['sd']:.{digits}f}" if point['sd'] is not None else '（单种子）')
    heading=('本轮主要指标未优于完整域随机化 PPO。' if delta>=0 else
             '本轮 Dyna-PPO 的主要指标均值低于完整域随机化 PPO。')
    lines=[f"**{heading}** 随机物理条件下，Dyna-PPO 的瞬时前向速度 RMSE 为 {fmt(dyna)} m/s，"
           f"完整 DR 为 {fmt(reference)} m/s；Dyna 均值{'升高' if relative>=0 else '降低'} {abs(relative):.2f}%（误差越低越好）。"]
    seed_deltas={};descriptions=[]
    for seed in sorted(dyna['seed_means'],key=int):
        value=dyna['seed_means'][seed];other=reference['seed_means'][seed]
        difference=value-other;percent=100*difference/other;seed_deltas[seed]=difference
        label='近乎持平' if abs(percent)<1 else ('更差' if difference>0 else '更好')
        descriptions.append(f'种子 {seed} {label}（差值 {difference:+.6f} m/s）')
    nominal_dyna,nominal_reference=nominal['dyna_ppo'][primary],nominal['full_dr'][primary]
    nominal_percent=100*(nominal_dyna['mean']/nominal_reference['mean']-1)
    nominal_label='均值接近' if abs(nominal_percent)<1 else ('Dyna均值更高' if nominal_percent>0 else 'Dyna均值更低')
    lines.append('随机条件逐种子比较：'+'；'.join(descriptions)+'。'
                 f'标称条件{nominal_label}：{fmt(nominal_dyna)} 对 {fmt(nominal_reference)} m/s（相对差 {nominal_percent:+.2f}%）。'
                 '“接近/持平”仅描述数值，不代表通过等效性检验；这些小样本结果不构成统计显著性结论。')
    baseline_percent=100*(baseline['mean']-dyna['mean'])/baseline['mean']
    lines.append(f"相对原固定参数基线，随机条件 RMSE {'下降' if baseline_percent>=0 else '上升'}约 {abs(baseline_percent):.2f}%。"
                 'Dyna 方案本身保留了域随机化，因此这项差异不能全部归功于 Dyna；额外想象更新的作用应与完整 DR 对照比较。')
    smooth=[];smooth_stats={}
    for metric,label in (('applied_target_rate_rms_rad_s','目标角速度 RMS'),
                         ('applied_target_acceleration_rms_rad_s2','目标角加速度 RMS'),
                         ('actual_joint_acceleration_rms_rad_s2','实际关节角加速度 RMS')):
        current,old=random['dyna_ppo'][metric],random['full_dr'][metric]
        keys=set(current['seed_means']) & set(old['seed_means'])
        valid=[s for s in keys if current['seed_means'][s] is not None and old['seed_means'][s] is not None]
        lowered=sum(current['seed_means'][s]<old['seed_means'][s] for s in valid)
        change=100*(current['mean']/old['mean']-1)
        smooth.append(f"{label}均值{'下降' if change<0 else '上升'} {abs(change):.2f}%（{lowered}/{len(valid)} 个种子降低）")
        smooth_stats[metric]={'dyna_minus_full_dr_relative_percent':change,
                              'seeds_with_lower_value':lowered,'valid_seed_pairs':len(valid)}
    clip_dyna=random['dyna_ppo']['raw_action_clipped_fraction']['mean']
    clip_reference=random['full_dr']['raw_action_clipped_fraction']['mean']
    mixed=any(v['seeds_with_lower_value']!=v['valid_seed_pairs'] for v in smooth_stats.values())
    lines.append('随机条件的平滑性诊断：'+'；'.join(smooth)+'。'
                 +('部分均值下降，但改善并非跨种子一致。' if mixed else '这三项差分指标在各训练种子上均降低。')
                 +f"原始动作裁剪比例从 {clip_reference:.2%} {'升至' if clip_dyna>clip_reference else '降至'} {clip_dyna:.2%}；"
                 '这些辅助指标应与主跟踪误差一同解读。')
    heading_dyna,heading_dr=(random[g]['abs_final_heading_yaw_deg'] for g in ('dyna_ppo','full_dr'))
    lateral_dyna,lateral_dr=(random[g]['abs_lateral_displacement_m'] for g in ('dyna_ppo','full_dr'))
    lines.append(f"随机条件下，Dyna 的末端绝对航向角为 {fmt(heading_dyna,2)}°、绝对侧移为 {fmt(lateral_dyna,4)} m，"
                 f"完整 DR 分别为 {fmt(heading_dr,2)}°、{fmt(lateral_dr,4)} m。")
    acceleration=random['dyna_ppo']['actual_joint_acceleration_rms_rad_s2']['seed_means']
    old_acceleration=random['full_dr']['actual_joint_acceleration_rms_rad_s2']['seed_means']
    acceleration_deltas={seed:acceleration[seed]-old_acceleration[seed] for seed in acceleration
                         if acceleration[seed] is not None and old_acceleration.get(seed) is not None}
    if acceleration_deltas:
        largest_reduction=min(acceleration_deltas,key=acceleration_deltas.get)
        reductions=sum(-value for value in acceleration_deltas.values() if value<0)
        if acceleration_deltas[largest_reduction]<0 and -acceleration_deltas[largest_reduction]>.5*reductions:
            lines[-1]+=(f" 实际关节角加速度均值下降主要来自种子 {largest_reduction}，"
                         f"该种子的末端绝对航向角为 {heading_dyna['seed_means'][largest_reduction]:.2f}°；"
                         '平滑指标的均值改善伴随任务跟踪和方向表现的取舍，不能仅概括为“更少抖动”。')
    return {'paragraphs':lines,'randomized_primary_dyna_minus_full_dr':delta,
            'randomized_primary_dyna_minus_full_dr_relative_percent':relative,
            'randomized_primary_reduction_vs_baseline_percent':baseline_percent,
            'randomized_primary_paired_seed_deltas':seed_deltas,
            'nominal_primary_dyna_minus_full_dr_relative_percent':nominal_percent,
            'randomized_smoothness_comparison':smooth_stats,
            'descriptive_near_tie_relative_threshold':.01}


def flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix+'.'+str(key) if prefix else str(key)
            if isinstance(item, dict):
                result.update(flatten(item, name))
            elif number(item):
                result[name] = item
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def configure_plotting():
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'pdf.fonttype': 3,
                         'figure.facecolor': 'white', 'axes.grid': True,
                         'grid.alpha': .2, 'axes.axisbelow': True})


def savefig(fig, out, name, artifacts):
    for extension in ('png', 'pdf'):
        path = out/f'{name}.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        artifacts[str(path.name)] = digest(path)
    plt.close(fig)


def train_value(row, metric):
    if metric == 'distance_x_m':
        return row.get('episode_metrics', {}).get('Episode/distance_x_m')
    if metric == 'value_function':
        return row.get('losses', {}).get('value_function')
    return row.get(metric)


def trailing(rows, metric, iteration, window=10):
    values = [train_value(row, metric) for row in rows if iteration-window<row['iteration']<=iteration]
    values = [value for value in values if number(value)]
    return statistics.mean(values) if values else None


def training_plot(training, out, artifacts):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    specs = [('mean_episode_return', '真实环境回合回报'), ('distance_x_m', '真实环境回合前向位移（m）'),
             ('value_function', '真实 rollout PPO 价值损失'), ('elapsed_recorded_s', '累计 rollout + 更新计时（s）')]
    for ax, (metric, title) in zip(axes.flat, specs):
        for group in GROUPS:
            runs = [rows for (variant, _), rows in training.items() if variant==group and rows]
            if not runs:
                continue
            iterations = sorted(set.intersection(*[{r['iteration'] for r in rows if number(train_value(r, metric))} for rows in runs]))
            xs, means, sds = [], [], []
            for iteration in iterations:
                values = [(next(row[metric] for row in rows if row['iteration']==iteration)
                           if metric=='elapsed_recorded_s' else trailing(rows, metric, iteration)) for rows in runs]
                point = stats(values)
                row = next(r for r in runs[0] if r['iteration']==iteration)
                xs.append(row['environment_transitions']/1e6); means.append(point['mean']); sds.append(point['sd'])
            ax.plot(xs, means, color=COLORS[group], label=LABELS[group], linewidth=1.9,
                    marker='o' if metric=='distance_x_m' else None, markersize=3)
            if len(runs)>1:
                ax.fill_between(xs, [m-s for m,s in zip(means,sds)], [m+s for m,s in zip(means,sds)],
                                color=COLORS[group], alpha=.14)
        ax.set(title=title, xlabel='累计真实环境步数（百万）')
        if metric=='value_function':
            ax.set_yscale('log')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Dyna-PPO：真实数据预算相同时的训练过程', fontsize=17)
    fig.text(.5,.01,'回报等曲线：每种子过去10更新有效值均值；阴影仅在多训练种子时表示样本SD。位移仅使用回合结束日志。\n'
             '训练物理分布不同；Dyna另有世界模型与想象更新，真实步数相同不等于总计算量相同。计时不含启动和评测。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.95]); savefig(fig,out,'learning_curves',artifacts)


def evaluation_plot(groups, out, artifacts):
    fig, axes = plt.subplots(2,3,figsize=(14,8.5))
    selected = ['velocity_x_rmse_m_s','distance_x_m','abs_final_heading_yaw_deg',
                'abs_lateral_displacement_m','return','raw_action_clipped_fraction']
    for ax, metric in zip(axes.flat, selected):
        for gi, group in enumerate(GROUPS):
            for ci, condition in enumerate(CONDITIONS):
                point = groups.get(condition,{}).get(group,{}).get(metric)
                if not point or point['mean'] is None:
                    continue
                x = ci+(gi-1)*.24
                ax.bar(x,point['mean'],width=.21,color=COLORS[group],alpha=.78,
                       yerr=point['sd'],capsize=3,label=LABELS[group] if ci==0 else None)
                seed_values=[v for v in point['seed_means'].values() if v is not None]
                offsets=[(i-(len(seed_values)-1)/2)*.045 for i in range(len(seed_values))]
                ax.scatter([x+d for d in offsets],seed_values,s=20,color='#172C40',zorder=4)
        ax.set_title(METRICS[metric],fontsize=11)
        ax.set_xticks([0,1],list(CONDITIONS.values()))
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.suptitle('共同物理条件下的最终策略评测',fontsize=17)
    fig.text(.5,.01,'点为各训练种子的回合均值；误差线为独立训练种子之间的样本SD，单种子时不画误差线。\n'
             '3042/4042 为旧研究已观察过的固定共同场景，此轮为可比性复评；仅最终检查点，不按正式成绩调参。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.9]);savefig(fig,out,'evaluation_comparison',artifacts)


def diagnostics_plot(diagnostics, out, artifacts):
    # Keep original field names on each axis; metrics never receive inferred units
    # or a label such as "validation" without such evidence in the saved schema.
    keys=sorted({key for rows in diagnostics.values() for row in rows for key in row
                 if key not in ('iteration','environment_transitions','seed','variant') and number(row[key])})
    priorities=(('train_dynamics_loss',),('train_reward_loss',),
                ('validation_to_persistence_mse_ratio',),('validation_reward_mae',),
                ('imagined_acceptance_fraction',),('imagined_value_loss',),
                ('imagined_optimizer_steps',),('wm','second'))
    chosen=[]
    for tokens in priorities:
        candidates=[key for key in keys if all(token in key.lower() for token in tokens) and key not in chosen]
        if candidates:
            chosen.append(min(candidates,key=lambda key:(len(key),key)))
    if not chosen:
        chosen=[key for key in keys if any(token in key.lower() for token in ('wm','model','imag'))]
    chosen=chosen[:8]
    if not chosen:
        return {'plotted_fields':[], 'note':'没有可用世界模型/想象数值日志；未生成诊断曲线。'}
    count=(len(chosen)+1)//2
    fig,axes=plt.subplots(count,2,figsize=(13,3.1*count+1),squeeze=False)
    styles=('-', '--', ':', '-.')
    for ax,key in zip(axes.flat,chosen):
        for index,(seed,rows) in enumerate(sorted(diagnostics.items())):
            points=[]
            for i,row in enumerate(rows):
                value=row.get(key)
                if not number(value):continue
                # The algorithm logs zero placeholders before imagination is
                # attempted. They are not measured zero prediction/policy loss.
                if 'imagined_acceptance_fraction' in key and row.get('losses.imagined_candidates',0)<=0:
                    value=math.nan
                if 'imagined_value_loss' in key and row.get('losses.imagined_optimizer_steps',0)<=0:
                    value=math.nan
                points.append((row.get('iteration',i+1),value))
            if points:
                ax.plot([x for x,y in points],[y for x,y in points],label=f'训练种子 {seed}',
                        linestyle=styles[index%len(styles)],linewidth=1.4)
        ax.set(title=key,xlabel='真实 rollout 更新次数')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    axes.flat[0].legend(frameon=False)
    fig.suptitle('世界模型与想象更新：保存日志中的诊断值',fontsize=16)
    fig.text(.5,.015,'保留原始日志字段名称；训练损失不自动等于留出误差，内部验证不替代共同环境评测。\n'
             '无想象候选/更新时，通过率和想象损失的日志占位零值不绘制；优化器步数为真实计数。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.95]);savefig(fig,out,'world_model_diagnostics',artifacts)
    return {'plotted_fields':chosen,'all_numeric_fields':keys,
            'note':'各训练种子分别展示；无想象候选/优化器更新时，对应通过率/损失的零占位不绘制。模型训练损失不等于泛化性能。'}


def imagination_budget_plot(diagnostics, out, artifacts):
    specs=[('losses.imagined_candidates','累计生成的一步想象转移',True),
           ('losses.imagined_accepted','累计通过门控的想象转移',True),
           ('losses.imagined_optimizer_steps_total','累计想象优化器更新次数',False),
           ('losses.world_model_optimizer_steps_total','累计世界模型联合优化器步数',False)]
    if not any(rows for rows in diagnostics.values()):return False
    fig,axes=plt.subplots(2,2,figsize=(12.8,8))
    for ax,(field,title,cumulative) in zip(axes.flat,specs):
        plotted=False
        for index,(seed,rows) in enumerate(sorted(diagnostics.items())):
            if not rows or not all(number(row.get(field)) for row in rows):continue
            values=[];total=0.
            for row in rows:
                total=total+row[field] if cumulative else row[field]
                values.append(total)
            ax.plot([row['iteration'] for row in rows],values,label=f'训练种子 {seed}',
                    linestyle=('-', '--', ':')[index%3],linewidth=1.8)
            plotted=True
        ax.set(title=title,xlabel='真实 rollout 更新次数')
        if not plotted:ax.text(.5,.5,'该字段未完整记录',ha='center',transform=ax.transAxes)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Dyna 额外计算预算：与真实环境步数分别记账',fontsize=16)
    fig.text(.5,.01,'门控通过数量不等于实际优化使用数量：最小样本门槛或 KL 停止可能跳过。\n'
             '世界模型一次联合优化器更新包含三个成员；按成员计的更新次数为图中值的三倍。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.95]);savefig(fig,out,'imagination_budget',artifacts)
    return True


def episode_value(row,key):
    if key=='fall_rate':return float(row['fell'])
    value=row.get(key,row.get('study_metrics',{}).get(key))
    return None if value is None else float(value)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study-dir',type=Path,default=ROOT/'runs/dyna_ppo_20261103')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--allow-incomplete',action='store_true',help='Explicit provisional report; never label missing runs/cases completed')
    args=ap.parse_args(); study=args.study_dir.resolve(); out=(args.output_dir or study/'analysis').resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain under the Sesame workspace')
    old=ROOT/'runs/robustness_ablation_20261103'
    if out==old or old in out.parents:raise ValueError('Never write into the old frozen study')
    inputs={};protocol=load(study/'protocol.json',inputs);phash=inputs[str(study/'protocol.json')]
    seeds=protocol['training']['seeds'];entries=protocol['checkpoints']
    expected={(g,s) for g in GROUPS for s in seeds}
    if {(e['variant'],e['train_seed']) for e in entries}!=expected or len(entries)!=len(expected):
        raise ValueError('Protocol must declare exactly one checkpoint for every group/training seed')
    missing=[];training={};diagnostics={};computations=[];case_rows=[];initials={}
    settings=protocol['evaluation'];n_episodes=len(settings['seeds'])*settings['episodes_per_seed']
    expected_ids={(s,i) for s in settings['seeds'] for i in range(settings['episodes_per_seed'])}
    for entry in entries:
        run=Path(entry['run_dir']);source=run/'training.jsonl'
        result=load(run/'result.json',inputs) if (run/'result.json').exists() else {}
        if result.get('status')!='completed':missing.append(f'{entry["id"]}: training status={result.get("status","missing")}')
        if source.exists():
            rows=load(source,inputs,lines=True)
            if any(row['iteration']!=i+1 for i,row in enumerate(rows)):raise ValueError(f'Noncontiguous training log: {source}')
            elapsed=0.
            for row in rows:
                if row['environment_transitions']!=row['iteration']*protocol['training']['num_envs']*protocol['training']['steps_per_env']:
                    raise ValueError('Real transition accounting mismatch')
                elapsed+=float(row.get('collection_seconds',0))+float(row.get('update_seconds',0))
                row['elapsed_recorded_s']=elapsed
            training[(entry['variant'],entry['train_seed'])]=rows
            if len(rows)!=protocol['training']['iterations']:missing.append(f'{entry["id"]}: {len(rows)} training updates')
            account={'variant':entry['variant'],'train_seed':entry['train_seed'],
                     'real_transitions':rows[-1]['environment_transitions'] if rows else 0,
                     'recorded_rollout_plus_update_seconds':elapsed,
                     'collection_seconds_sum':sum(float(r.get('collection_seconds',0)) for r in rows),
                     'update_seconds_sum':sum(float(r.get('update_seconds',0)) for r in rows)}
        else:
            rows=[];missing.append(f'{entry["id"]}: training.jsonl missing');account={'variant':entry['variant'],'train_seed':entry['train_seed']}
        if entry['variant']=='dyna_ppo':
            config=run/'dyna_config.json'
            if config.exists():account['dyna_config']=load(config,inputs)
            else:missing.append(f'{entry["id"]}: dyna_config.json missing')
            account['saved_dyna_result_counters']={k:v for k,v in flatten(result).items()
                if any(t in k.lower() for t in ('dyna','imag','world_model','wm_'))}
            account.update({'dyna.'+key:value for key,value in flatten(result.get('dyna_training_summary',{})).items()})
            for field in ('wm_fit_seconds','imagined_update_seconds'):
                measured=[row.get('losses',{}).get(field) for row in rows]
                valid=[value for value in measured if number(value)]
                account[field+'_sum']=sum(valid) if valid else None
                account[field+'_logged_updates']=len(valid)
            if result.get('status')=='completed':
                counters=result.get('dyna_training_summary',{})
                for field,expected_value in (('real_updates',len(rows)),('real_transitions_seen',rows[-1]['environment_transitions'] if rows else 0)):
                    if counters.get(field)!=expected_value:raise ValueError(f'Dyna result/log counter mismatch: {field}')
                checks={'imagined_candidates':'imagined_generated','imagined_accepted':'imagined_accepted',
                        'imagined_optimizer_steps':'imagined_optimizer_steps'}
                for field,total_field in checks.items():
                    values=[row.get('losses',{}).get(field) for row in rows]
                    if all(number(value) for value in values) and sum(values)!=counters.get(total_field):
                        raise ValueError(f'Dyna per-update/total counter mismatch: {field}')
                if counters.get('imagined_accepted',0)>counters.get('imagined_generated',0):
                    raise ValueError('More accepted than generated imagined transitions')
            dr=[]
            for row in rows:
                diagnostic={'iteration':row['iteration'],'environment_transitions':row['environment_transitions']}
                diagnostic.update({k:v for k,v in flatten(row).items() if any(t in k.lower() for t in ('dyna','imag','world_model','wm_'))})
                if len(diagnostic)>2:dr.append(diagnostic)
            for filename in ('dyna_diagnostics.jsonl','world_model_diagnostics.jsonl','world_model.jsonl'):
                if (run/filename).exists():
                    extra=load(run/filename,inputs,lines=True)
                    by_iteration={row['iteration']:row for row in dr}
                    for i,row in enumerate(extra):
                        iteration=row.get('iteration',i+1)
                        by_iteration.setdefault(iteration,{'iteration':iteration}).update(flatten(row))
                    dr=[by_iteration[k] for k in sorted(by_iteration)]
            diagnostics[entry['train_seed']]=dr
        computations.append(account)
        for condition in settings['conditions']:
            path=study/'evaluation/cases'/entry['id']/f'{condition}.json'
            if not path.exists():missing.append(f'{entry["id"]}/{condition}: evaluation missing');continue
            case=load(path,inputs)
            if case.get('status')!='completed' or not case.get('pairing_verified') or not case.get('all_actions_observations_rewards_finite'):
                raise ValueError(f'Invalid evaluation case: {path}')
            if case['protocol_sha256']!=phash or case['evaluation_environment_variant']!='full_dr':
                raise ValueError('Evaluation must come from this protocol and common full-DR environment')
            if case['policy_id']!=entry['id'] or case['variant']!=entry['variant'] or case['train_seed']!=entry['train_seed']:
                raise ValueError('Evaluation identity mismatch')
            if result.get('checkpoint_sha256')!=case['checkpoint_sha256']:raise ValueError('Evaluation/training checkpoint mismatch')
            episodes=case['episodes'];ids=[(r['evaluation_seed'],r['env_id']) for r in episodes]
            if len(ids)!=n_episodes or set(ids)!=expected_ids:raise ValueError('Expected first complete paired episodes exactly once')
            for initial in case['initial_conditions']:
                key=(condition,initial['evaluation_seed'])
                if key in initials and initials[key]!=initial:raise ValueError('Initial-state/domain pairing mismatch')
                initials[key]=initial
            row={'policy_id':entry['id'],'variant':entry['variant'],'train_seed':entry['train_seed'],'condition':condition,'episodes':len(episodes)}
            for metric in METRICS:
                point=stats(episode_value(e,metric) for e in episodes)
                row[metric]=point['mean'];row[metric+'_valid_episodes']=point['count']
            case_rows.append(row)
    if missing and not args.allow_incomplete:raise RuntimeError('Incomplete inputs: '+'; '.join(missing))
    groups={};group_rows=[];differences=[]
    for condition in settings['conditions']:
        groups[condition]={}
        for group in GROUPS:
            selected=[r for r in case_rows if r['variant']==group and r['condition']==condition]
            groups[condition][group]={}
            for metric in METRICS:
                point=stats(r[metric] for r in selected)
                point['seed_means']={str(r['train_seed']):r[metric] for r in selected}
                groups[condition][group][metric]=point
                group_rows.append({'condition':condition,'variant':group,'metric':metric,**{k:v for k,v in point.items() if k!='seed_means'}})
        for reference in ('baseline','full_dr'):
            for metric in METRICS:
                paired={}
                for seed in seeds:
                    a=groups[condition][reference][metric]['seed_means'].get(str(seed));b=groups[condition]['dyna_ppo'][metric]['seed_means'].get(str(seed))
                    if a is not None and b is not None:paired[str(seed)]=b-a
                differences.append({'condition':condition,'candidate':'dyna_ppo','reference':reference,'metric':metric,
                                    **stats(paired.values()),'paired_seed_differences':paired})
    out.mkdir(parents=True,exist_ok=True);configure_plotting();artifacts={}
    if training:training_plot(training,out,artifacts)
    if case_rows:evaluation_plot(groups,out,artifacts)
    diagnostics_info=diagnostics_plot(diagnostics,out,artifacts)
    diagnostics_info['imagination_budget_plot_available']=imagination_budget_plot(diagnostics,out,artifacts)
    write_csv(out/'per_training_seed.csv',case_rows);write_csv(out/'group_summary.csv',group_rows)
    write_csv(out/'paired_seed_differences.csv',[{**{k:v for k,v in r.items() if k!='paired_seed_differences'},
              **{f'delta_seed{s}':v for s,v in r['paired_seed_differences'].items()}} for r in differences])
    write_csv(out/'world_model_diagnostics.csv',[{'train_seed':seed,**row} for seed,rows in diagnostics.items() for row in rows])
    write_csv(out/'compute_accounting.csv',[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in computations])
    summary={'status':'provisional' if missing else 'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
             'protocol_sha256':phash,'analysis_source_sha256':digest(__file__),'inputs_sha256':inputs,
             'training_seeds':seeds,'cases_completed':len(case_rows),'cases_expected':len(entries)*len(settings['conditions']),
             'episodes_per_case':n_episodes,'total_evaluation_episodes':sum(r['episodes'] for r in case_rows),
             'aggregation':'Each policy/condition averages paired episodes; independent unit is training seed. SD is sample SD across available seed means; n=1 yields null SD.',
             'evaluation_scope':'Seeds 3042/4042 were not used for training but their results were observed in the prior DR study; this is a fixed-scenario comparability re-evaluation, not a new blind test set.',
             'configuration_selection':'Dyna configuration is selected using training-internal held-out diagnostics and smoke checks, then frozen before formal evaluation; no tuning based on formal results.',
             'per_seed':case_rows,'groups':groups,'paired_seed_differences':differences,
             'compute_accounting':computations,'world_model_diagnostics':diagnostics_info,
             'missing_inputs':missing,'artifacts_sha256':artifacts,'limitations':protocol.get('limitations',[])}
    summary['descriptive_findings']=interpret_results(groups,not missing)
    save_json(out/'summary.json',summary)
    opening=[item for paragraph in summary['descriptive_findings']['paragraphs'] for item in (paragraph,'')]
    lines=['# Dyna-PPO 训练与共同条件评测','',*opening,f'状态：**{summary["status"]}**。训练种子：{seeds}；已完成 {len(case_rows)}/{summary["cases_expected"]} 组策略/条件评测。',
           '',f'每个策略/条件先平均 {n_episodes} 个配对回合，再按独立训练种子汇总。均值后的 ± 为种子间样本标准差，不是置信区间。',
           '评测种子 3042/4042 未用于训练，但对应场景的成绩已在旧 DR 研究中观察过。本轮沿用这些固定共同场景做可比性复评，不是全新盲测或独立的新测试集。',
           'Dyna 配置只依据训练内部留出诊断与 smoke 检查确认，在正式评测前冻结；不根据本轮正式成绩调参。',
           '单训练种子不能估计训练随机性或声称统计显著性；即使有三个种子，也不据此自动宣称统计显著性。','',
           '| 条件 | 策略 | 瞬时前向 RMSE (m/s) | 前向位移 (m) | 绝对航向 (°) | 跌倒比例 | 有效训练种子 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    def fmt(point):
        if point['mean'] is None:return '待完成'
        return f"{point['mean']:.5f}"+(f" ± {point['sd']:.5f}" if point['sd'] is not None else '（单种子）')
    for condition in settings['conditions']:
        for group in GROUPS:
            g=groups[condition][group]
            lines.append('| '+' | '.join([CONDITIONS[condition],LABELS[group],*[fmt(g[k]) for k in ('velocity_x_rmse_m_s','distance_x_m','abs_final_heading_yaw_deg','fall_rate')],str(g['velocity_x_rmse_m_s']['count'])])+' |')
    lines += ['', '## 主要指标的配对种子差值', '',
              '差值为 Dyna-PPO 减去参照策略的瞬时前向速度 RMSE；负值表示误差降低。', '',
              '| 条件 | 参照 | 配对差值均值 ± 样本SD (m/s) | 每个训练种子的差值 |',
              '| --- | --- | --- | --- |']
    for row in differences:
        if row['metric']=='velocity_x_rmse_m_s':
            values='；'.join(f'{seed}: {delta:+.6f}' for seed,delta in row['paired_seed_differences'].items()) or '待完成'
            lines.append('| '+' | '.join([CONDITIONS[row['condition']],LABELS[row['reference']],fmt(row),values])+' |')
    lines += ['', '## 真实数据与额外计算', '',
              f'每个策略的计划真实数据预算为 {protocol["training"]["transitions_per_run"]:,} 环境步。Dyna 额外训练世界模型并生成想象转移、执行辅助更新。真实环境步数相同不等于相同总计算预算；不能把该对照解释为相同算力下的优势。',
              '计时表累计日志中的 collection_seconds + update_seconds，排除启动与正式评测；世界模型/想象详细字段保存在 summary.json 与 world_model_diagnostics.csv。未记录的计时、预测误差或想象数量不会补造。不同运行的计时受共享 GPU 负载、缓存和运行时状态影响，不用这些非同时测量证明计算效率优势。',
              'model_optimizer_steps 是三成员联合优化器的步数；model_member_optimizer_steps 为其三倍。imagined_generated/accepted 分别为生成与通过筛选的一步转移，accepted 不保证已被优化器使用（小批量门槛或 KL 停止可能跳过）。',
              '训练回报取真实环境已完成回合的 RSL-RL 缓冲均值，再作 10 更新尾随平均。训练物理分布不同；应以共同条件评测比较策略性能。',
              '', '[真实训练曲线](learning_curves.png) · [共同条件对照](evaluation_comparison.png)',
              '[世界模型诊断](world_model_diagnostics.png)' if diagnostics_info['plotted_fields'] else '世界模型数值日志尚不可用，未生成模型诊断曲线。',
              '[想象数据与模型更新预算](imagination_budget.png)' if diagnostics_info['imagination_budget_plot_available'] else '想象预算日志尚不可用。',
              '[逐训练种子指标](per_training_seed.csv) · [配对种子差值](paired_seed_differences.csv) · [计算计时](compute_accounting.csv)',
              '', '## 解释范围', '',
              '仅评测冻结的 final.pt；标称与随机条件共用已验证的初始状态和实际物理参数。3042/4042 场景已在旧研究中观察，本轮复评用于可比性，不作为新盲测。随机条件仍在训练域范围内，不是分布外或真实机器人验证。',
              '奖励未为航向重新设计；速度改善不自动表示消除侧行。平滑性指标以 50 Hz 控制频率统计，不代表物理峰值、机械功率或抖动频率。',
              '世界模型训练损失不是独立泛化证明；是否有留出预测指标以原始日志字段和 dyna_config.json 为准。']
    lines += ['', '## Dyna 实际计算记录', '',
              '| 训练种子 | 真实环境步数 | 想象优化器更新次数 | rollout + 更新日志计时 (s) |',
              '| --- | --- | --- | --- |']
    for account in computations:
        if account['variant']=='dyna_ppo':
            steps=account.get('dyna.imagined_optimizer_steps','未记录')
            elapsed=account.get('recorded_rollout_plus_update_seconds')
            lines.append('| '+' | '.join([str(account['train_seed']),str(account.get('real_transitions','未记录')),
                         str(steps),f'{elapsed:.2f}' if elapsed is not None else '未记录'])+' |')
    if missing:lines+=['','## 未完成输入','',*['- '+item for item in missing]]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'analysis_dir':str(out),'cases':len(case_rows),'training_seeds':seeds},ensure_ascii=False))


if __name__=='__main__':main()
