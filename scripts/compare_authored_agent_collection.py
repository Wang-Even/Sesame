"""Compare completed collections using saved scalar reports, without simulation."""
import argparse
import json
from pathlib import Path
from summarize_mimicagent_collection import summarize


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--authored',type=Path,required=True)
    a=p.parse_args()
    runs=[a.previous.resolve(),a.authored.resolve()]
    protocols=[read(r/'protocol.json') for r in runs]
    if protocols[0]['config']!=protocols[1]['config']:
        raise ValueError('Different execution/acceptance configuration')
    for name in ['sesame_trajectory_agent/validation.py','sesame_trajectory_agent/mimic_motion.py',
                 'sesame_research/robot.py','sesame_research/task.py']:
        if protocols[0]['source_sha256'][name]!=protocols[1]['source_sha256'][name]:
            raise ValueError('Different simulator/validator source: '+name)
    summaries=[];records=[]
    for run in runs:
        if read(run/'status.json')['status']!='complete':raise ValueError('Collection incomplete')
        s=summarize(run)
        reports=[read(f) for f in sorted(run.glob('candidates/*/attempt_*/report.json'))]
        executed=[r for r in reports if 'nominal' in r]
        s['nominal_pass_count']=sum(r['nominal']['passed'] for r in executed)
        s['nominal_no_fall_count']=sum(not r['nominal_metrics']['fell'] for r in executed)
        s['nominal_failure_counts']={}
        for r in executed:
            for reason in r['nominal']['reasons']:
                s['nominal_failure_counts'][reason]=s['nominal_failure_counts'].get(reason,0)+1
        best=min(executed,key=lambda r:(r['nominal_metrics']['fell'],r['nominal_metrics']['mean_speed_error_m_s']))
        s['diagnostic_selection']={'criterion':'lowest nominal mean-speed error among non-fallen executions, selected after collection',
            'id':best['id'],'metrics':best['nominal_metrics'],'accepted':best['accepted'],
            'nominal_reasons':best['nominal']['reasons']}
        summaries.append(s);records.append(executed)
    data={'config_and_validator_match':True,'collections':summaries,
        'limitations':['Different program budgets (111 vs 30); not a blind model comparison',
            'Current chat agent has access to previous failures and saved GPU geometry probes',
            'Authored parameter choices are manual and adaptive; candidates are correlated',
            'Admission includes all unchanged local filters and three reused randomized trials; not held-out success probability',
            'Recorded forward velocity is a 50 Hz sample mean; displacement divided by time can differ',
            'Contact foot-center velocity is a slip proxy, not contact-point sliding velocity']}
    out=runs[1]
    (out/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors=['#ce632c','#155aaa'];labels=['DeepSeek workflow','Current chat agent']
    for i,(s,rows) in enumerate(zip(summaries,records)):
        denominator=s['unique_generated_motion_arrays']
        values=[100*s['unique_executed_motion_arrays']/denominator,100*s['accepted_count']/denominator]
        x=[-.18+i*.36,1-.18+i*.36]
        bars=axs[0].bar(x,values,width=.33,color=colors[i],label=labels[i])
        for bar,n in zip(bars,[s['unique_executed_motion_arrays'],s['accepted_count']]):
            axs[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{denominator}',ha='center',fontsize=9)
        axs[1].scatter([r['nominal_metrics']['mean_forward_speed_m_s'] for r in rows],
            [r['nominal_metrics']['mean_speed_error_m_s'] for r in rows],s=27,color=colors[i],alpha=.8,label=labels[i])
    axs[0].set(xticks=[0,1],xticklabels=['Kinematics passed','All checks passed'],ylabel='Fraction of unique references (%)',ylim=(0,112),title='(a) Collection outcomes')
    axs[0].legend(frameon=False,loc='upper right')
    axs[1].axvline(.1,color='gray',linestyle='--',linewidth=1)
    axs[1].axhline(protocols[0]['config']['gates']['mean_speed_error_max_m_s'],color='gray',linestyle=':',linewidth=1)
    axs[1].set(xlabel='Recorded mean forward velocity (m/s)',ylabel='Mean forward speed error (m/s)',title='(b) All nominal executions')
    axs[1].legend(frameon=False,loc='upper right');axs[1].grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(out/('comparison.'+ext),dpi=180)
    plt.close(fig)
    old,new=summaries
    lines=['# 当前对话 Agent 生成轨迹的实验','',
        '由当前 Codex 对话模型规划、编写并根据仿真反馈修订；没有调用 DeepSeek API，也没有采样训练策略的动作作为生成结果。',
        '', '| 项目 | 上一轮 DeepSeek | 本轮对话 Agent |','|---|---:|---:|']
    fields=[('代码版本','attempts'),('去重后参考轨迹','unique_generated_motion_arrays'),
        ('运动学通过并实际执行','unique_executed_motion_arrays'),('物理仿真总次数','total_physical_trials'),
        ('正常仿真未摔倒','nominal_no_fall_count'),('包含随机化的跌倒次数','trials_with_local_fall'),
        ('正常仿真通过全部动态条件','nominal_pass_count'),('全部检查通过并入库','accepted_count'),
        ('API 调用次数','api_calls')]
    lines += [f'| {label} | {old[key]} | {new[key]} |' for label,key in fields]
    n=new['unique_generated_motion_arrays'];count=new['accepted_count']
    lines += ['',f'本轮入库通过率：**{count}/{n} = {count/n:.1%}**。分母包含所有成功生成的去重参考，不只统计进入仿真的样本。',
        f"正常仿真未摔倒：{new['nominal_no_fall_count']}/{new['nominal_executions']}；未摔倒不等于通过全部检查。",
        '', '检验保持不变：6 秒、50 Hz、目标 +x 方向 0.1 m/s；运动学通过后执行 1 次正常仿真和 3 次随机化仿真。生成阶段没有裁剪或自动修正。阈值是原有本地实验筛选条件，不是课程或论文规定。',
        '', '生成思路：根据 CUDA 正运动学调整髋关节中立角，比较步幅与步频、抬腿曲线、支撑时序及左右步幅差；每次修订都计入 30 个版本的预算。详细参数、反馈来源、代码和原始执行轨迹均已保存。',
        '', '这不是两个模型的公平排名：预算不同，我已看到上一轮失败记录，且候选之间存在自适应修订关系。两轮执行配置及验证器源文件哈希一致。',
        '', '代表视频按“未摔倒的正常仿真中平均速度误差最小”事后选择，用于诊断，不能代表平均表现或独立测试结果。',
        '', f"选中 `{new['diagnostic_selection']['id']}`，是否通过全部检验：{new['diagnostic_selection']['accepted']}。",
        '', '[对比图](comparison.png) · [每条指标](collection_attempts.csv) · [完整汇总](comparison.json) · [参考与实际执行视频](reference_vs_execution.mp4)',
        '', '复现已经提交的程序（输出目录必须不存在）：','', '```bash',
        '.venv_supermimic_research/bin/python scripts/run_mimicagent.py \\\n  --authored-dir runs/codex_agent30_inputs_20261114 \\\n  --output runs/codex_agent30_replay \\\n  --candidates 30 --repairs 0','```',
        '', '该命令重放保存的生成程序并重新仿真，不调用模型。`plan.json`、`design.json`、`motion.py` 保存提案；`READY` 表示一个输入版本完整。']
    selection=new['diagnostic_selection'];m=selection['metrics']
    lines += ['',f"本轮运行耗时 {new['elapsed_s']/60:.2f} 分钟（含等待人工提案），进程峰值显存 {new['peak_process_memory_mib']} MiB。此时间不能用作两个模型的生成效率排名。",
        '',f"代表轨迹：偏航 RMS {m['heading_rms_deg']:.2f}°；未通过的正常仿真指标为平面速度 RMSE {m['velocity_rmse_m_s']:.3f} > 0.05 m/s、平均前进速度误差 {m['mean_speed_error_m_s']:.3f} > 0.025 m/s、接触足端速度 {m['contact_foot_speed_rms_m_s']:.3f} > 0.05 m/s。"]
    if (out/'video.json').exists():
        v=read(out/'video.json')['execution']
        lines += ['',f"视频中的实际前进位移为 {v['forward_displacement_m']:.3f} m，6 秒位移/时间为 {v['forward_speed_from_displacement_m_s']:.3f} m/s，侧向净位移 {100*v['lateral_displacement_m']:.2f} cm；原验收的 50 Hz 速度采样均值为 {v['mean_vx_m_s']:.3f} m/s。二者存在差异，后续应增加物理子步采样核查，不能直接替换旧指标来改判通过。"]
    lines += ['', '本轮没有合格训练数据，未启动新的训练。下一步值得检查控制器的目标插值与接触切换，以及速度采样和足端中心速度对滚动接触的影响；增加提案数量本身尚未解决这些问题。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparison':str(out/'comparison.json'),'selected':new['diagnostic_selection']},ensure_ascii=False))


if __name__=='__main__':main()
