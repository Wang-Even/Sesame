#!/usr/bin/env python3
"""Read saved CUDA experiment records and prepare report sections 4 and 5."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/course_draft_20261115/improvements')
    args=parser.parse_args();study=args.study_dir.resolve();out=args.output_dir.resolve()
    result=json.loads((study/'evaluation/result.json').read_text())
    if result['status']!='completed':raise RuntimeError('Evaluation incomplete')
    rows=[]
    for filename in result['cases']:
        case=json.loads(Path(filename).read_text());m=case['metrics'];extra=case['additional_metrics']
        rows.append(dict(model=case['variant'],condition=case['condition'],target_speed_m_s=case['target_speed_m_s'],
            episodes=m['episodes'],common_reward_return=m['return']['mean'],
            training_action_rate_reward_return=extra['return_under_training_action_rate_coefficient']['mean'],
            mean_forward_speed_m_s=m['avg_speed_m_s']['mean'],sampled_mean_vx_m_s=extra['sampled_mean_vx_m_s']['mean'],
            velocity_rmse_m_s=m['velocity_x_rmse_m_s']['mean'],heading_rms_deg=math.degrees(m['heading_rms_rad']['mean']),
            forward_distance_m=m['distance_x_m']['mean'],lateral_displacement_m=m['abs_lateral_displacement_m']['mean'],
            normalized_action_rate_rms_per_s=extra['normalized_action_rate_rms_per_s']['mean'],
            target_rate_rms_rad_s=m['applied_target_rate_rms_rad_s']['mean'],
            target_acceleration_rms_rad_s2=m['applied_target_acceleration_rms_rad_s2']['mean'],
            actual_joint_acceleration_rms_rad_s2=m['actual_joint_acceleration_rms_rad_s2']['mean'],
            fall_rate=m['fall_rate'],true_termination_rate=case['true_termination_rate'],
            source_case=filename,checkpoint=case['checkpoint'],checkpoint_sha256=case['checkpoint_sha256']))
    if len(rows)!=8 or sum(r['episodes'] for r in rows)!=1024:raise RuntimeError('Unexpected evaluation coverage')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'motion_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'baseline':'#c65c27','smooth':'#245eaf','fast':'#245eaf'}
    labels={'baseline':'PPO baseline','smooth':'PPO stronger action penalty','fast':'PPO trained at 0.20 m/s'}
    for speed,models,stem,metrics in [
        (.10,['baseline','smooth'],'smoothness_comparison',[
            ('target_acceleration_rms_rad_s2','Target acceleration RMS (rad/s²)'),
            ('actual_joint_acceleration_rms_rad_s2','Joint acceleration RMS (rad/s²)'),
            ('velocity_rmse_m_s','Forward velocity RMSE (m/s)'),('heading_rms_deg','Heading RMS (deg)')]),
        (.20,['baseline','fast'],'speed_comparison',[
            ('mean_forward_speed_m_s','Effective forward speed (m/s)'),('velocity_rmse_m_s','Forward velocity RMSE (m/s)'),
            ('heading_rms_deg','Heading RMS (deg)'),('true_termination_rate','Early termination fraction')])]:
        fig,axes=plt.subplots(2,2,figsize=(10,6.4),layout='constrained')
        for panel,(ax,(metric,title)) in enumerate(zip(axes.flat,metrics)):
            for j,model in enumerate(models):
                selected=[next(r for r in rows if r['model']==model and r['condition']==condition and r['target_speed_m_s']==speed) for condition in ['nominal','randomized']]
                xs=[x+(j-.5)*.32 for x in [0,1]]
                values=[r[metric] for r in selected]
                ax.bar(xs,values,width=.30,color=colors[model],label=labels[model])
                if metric=='true_termination_rate':
                    ax.set_ylim(0,1)
                    for x,y in zip(xs,values):ax.text(x,y+.02,f'{y:.1%}',ha='center',fontsize=9)
            ax.set_xticks([0,1],['Nominal','Randomized']);ax.set_ylabel(title);ax.set_title(f'({"abcd"[panel]})',loc='left');ax.grid(axis='y',alpha=.15)
            if metric=='mean_forward_speed_m_s':ax.axhline(speed,color='#555',ls='--',lw=1,label='Command 0.20 m/s')
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.savefig(out/f'{stem}.png',dpi=180);fig.savefig(out/f'{stem}.pdf');plt.close(fig)
    training=[]
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for ax,model in zip(axes,['baseline','smooth','fast']):
        folder=study/'training'/f'{model}_seed42';d=json.loads((folder/'result.json').read_text())
        training.append(dict(model=model,training_s=d['training_stage_seconds'],loop_wall_s=d['training_loop_wall_seconds'],
                             program_wall_s=d['wall_seconds_before_shutdown'],initial_network_sha256=d['initial_network_sha256'],
                             initial_condition_sha256=d['initial_condition_sha256']))
        elapsed=0.;points=[]
        for line in (folder/'training.jsonl').read_text().splitlines():
            r=json.loads(line);elapsed+=r['collection_seconds']+r['update_seconds']
            if r['mean_episode_return'] is not None:points.append((elapsed/60,r['mean_episode_return']))
        ax.plot([p[0] for p in points],[p[1] for p in points],color=colors[model],lw=1.5)
        ax.set(title=model,xlabel='Logged training time (min)',ylabel='Own training reward return');ax.grid(alpha=.15)
    fig.savefig(out/'motion_training_curves.png',dpi=180);fig.savefig(out/'motion_training_curves.pdf');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(study=str(study),training_seed_count=1,training=training,results=rows,
        caveats=['All models use the same baseline action-rate coefficient 0.08 for evaluation return.',
                 'The smoothness training reward uses coefficient 0.16 and is displayed separately.',
                 'Speed comparison uses target 0.20 for both policies, one trained at 0.10 and one at 0.20.',
                 'Episode counts are not independent training seeds. No invented success thresholds.']),indent=2))
    md=['## 4. 问题一：行走抖动与平滑性','',
        '相邻动作差惩罚有助于限制动作突变，但不保证实际关节运动足够平滑。本轮从头训练两个普通 PPO：基线系数为 0.08，改进组为 0.16；目标速度均为 0.10 m/s，其他配置相同。两组初始网络、状态与域参数哈希一致，训练 seed 均为 42，每组 204.8 万步。', '',
        '评估时两组统一使用原系数 0.08 计算回报，同时保存改进系数下的回报。每个模型、每个条件用测试 seed 35042、36042 共 128 回合；本轮是单训练种子的探索，不据此给出稳定性显著改善结论。','',
        '![动作平滑性与跟踪误差](improvements/smoothness_comparison.png)','',
        '图 4：平滑性改进对照。(a) 执行目标角加速度；(b) 实际关节加速度；(c) 前向速度误差；(d) 航向误差。均为共同测试结果。','',
        '| 条件 | 方法 | 共同奖励回报 | 目标加速度 RMS（rad/s²） | 实际关节加速度 RMS（rad/s²） | 前进速度（m/s） | 速度 RMSE（m/s） | 航向 RMS（°） | 跌倒数 / 回合数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['target_speed_m_s']!=.10:continue
        name='基线' if row['model']=='baseline' else '动作差惩罚加倍'
        md.append(f'| {"标准" if row["condition"]=="nominal" else "随机"}参数 | {name} | {row["common_reward_return"]:.3f} | {row["target_acceleration_rms_rad_s2"]:.2f} | {row["actual_joint_acceleration_rms_rad_s2"]:.2f} | {row["mean_forward_speed_m_s"]:.4f} | {row["velocity_rmse_m_s"]:.4f} | {row["heading_rms_deg"]:.2f} | {round(row["fall_rate"]*row["episodes"])}/{row["episodes"]} |')
    md+=['','动作变化率为相邻归一化动作差的 RMS 除以 0.02 秒，单位是归一化动作每秒；它不是百分比“抖动率”。实际关节加速度使用 50 Hz 关节速度差分 Δq̇ / 0.02 秒，单位 rad/s²；这不代表 500 Hz 物理子步下的真实加速度峰值。执行目标角加速度同样由控制步差分计算。各项排除回合重置跳变，实际速度差分首步使用已知重置速度。需同时观察速度、航向和跌倒；不能把机器人少动或站着不动带来的低加速度当作行走改善。归一化动作变化率、执行目标变化率及两种奖励回报另存 CSV。','',
         '## 5. 问题二：高速前进时的速度与方向保持','',
         '将目标速度从 0.10 提高到 0.20 m/s，保持相同奖励形式、动作差系数 0.08、域随机化及训练预算，从头训练高速 PPO。共同测试目标为 0.20 m/s，对照是原 0.10 m/s 训练策略直接接收新指令。这里只测试原任务两倍速度，不能据此描述为机器人的最大速度。','',
         '![高速前进对照](improvements/speed_comparison.png)','',
         '图 5：0.20 m/s 共同指令下的高速训练与直接迁移对照。(a) 有效前进速度，虚线为目标；(b) 速度误差；(c) 航向误差；(d) 提前终止比例。','',
         '| 条件 | 方法 | 共同奖励回报 | 前进速度（m/s） | 速度 RMSE（m/s） | 航向 RMS（°） | 侧移（cm） | 跌倒数 / 回合数 | 提前终止数 / 回合数 |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['target_speed_m_s']!=.20:continue
        name='低速策略直接迁移' if row['model']=='baseline' else '0.20 m/s 训练'
        md.append(f'| {"标准" if row["condition"]=="nominal" else "随机"}参数 | {name} | {row["common_reward_return"]:.3f} | {row["mean_forward_speed_m_s"]:.4f} | {row["velocity_rmse_m_s"]:.4f} | {row["heading_rms_deg"]:.2f} | {row["lateral_displacement_m"]*100:.2f} | {round(row["fall_rate"]*row["episodes"])}/{row["episodes"]} | {round(row["true_termination_rate"]*row["episodes"])}/{row["episodes"]} |')
    md+=['','前进速度为 X 方向净位移除以回合时长；逐控制步 vx 均值另存 CSV。提前终止包括跌倒、越界或无效状态，与正常 10 秒结束区分。全部测试轨迹纳入统计。','',
         '三组各自的[训练曲线另存 PDF](improvements/motion_training_curves.pdf)。平滑组修改了奖励系数，高速组修改了任务速度，因此不从三条训练回报曲线推断性能高低；性能对照以上述共同测试为准。','',
         '| 新训练 | 采样与更新时间（分钟） | 完整训练进程（分钟） |','|---|---:|---:|']
    for r in training:md.append(f'| {r["model"]} | {r["training_s"]/60:.2f} | {r["program_wall_s"]/60:.2f} |')
    md+=['',f'结果目录：`{study.relative_to(ROOT)}`。三组串行训练，均使用 final.pt；本轮测试共 1024 回合。实验仍只包含一个训练种子，若要论证重复训练稳定性，需补充其他 seed。','']
    (out/'section_improvements.md').write_text('\n'.join(md))
    print(json.dumps(dict(output=str(out),cases=len(rows),episodes=1024)))


if __name__=='__main__':main()
