"""Reproduce the original experimental filter and its matched controls.

The filter is a local research choice, not a course requirement or a proof that
rejected references cannot be learned. Keep historical run records unchanged.

All model computation is delegated to the CUDA training driver. NumPy here only
packs previously saved arrays; plotting summarizes recorded scalar metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv_supermimic_research/bin/python'


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def pack_augmented(original,accepted,destination):
    import numpy as np
    manifest=read(accepted/'manifest.json')
    if manifest['accepted_count']<=0:raise ValueError('The original experimental filter exported no references; this does not establish that the candidates are unusable')
    extra=accepted/'train'
    names=read(original/'joint_names.json')
    if read(extra/'joint_names.json')!=names:raise ValueError('Joint order mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    fields=['root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel']
    frames=[]
    for field in fields:
        chunks=[np.load(p/(field+'.npy'),allow_pickle=False) for p in (original,extra)]
        if not frames:frames=[len(x) for x in chunks]
        np.save(destination/(field+'.npy'),np.concatenate(chunks),allow_pickle=False)
    first=np.load(original/'motion_ends.npy',allow_pickle=False)
    second=np.load(extra/'motion_ends.npy',allow_pickle=False)
    np.save(destination/'motion_ends.npy',np.concatenate([first,second+frames[0]]),allow_pickle=False)
    motion_names=read(original/'motion_names.json')+['agent_'+s for s in read(extra/'motion_names.json')]
    write(destination/'joint_names.json',names);write(destination/'motion_names.json',motion_names)
    meta=read(original/'meta.json')
    meta.update(clips=len(motion_names),frames=sum(frames),duration_h=(sum(frames)-len(motion_names))/50/3600,
        source='Original training references plus validated Agent nominal executions',
        original_source=str(original),agent_manifest=str(accepted/'manifest.json'),
        agent_manifest_sha256=hashlib.sha256((accepted/'manifest.json').read_bytes()).hexdigest(),split='train')
    write(destination/'meta.json',meta)


def summarize(root,search,stage):
    from collections import Counter
    import csv
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[('supermimic_original_seed42','SuperMimic: original'),
            ('ppo_original_seed42','PPO: original'),
            ('supermimic_agent_augmented_seed42','SuperMimic: Agent augmented')]
    rows=[];curves={}
    for group,label in groups:
        path=root/group
        s=read(path/'status.json') if (path/'status.json').exists() else {'status':'not_started'}
        row={'group':group,'status':s['status']}
        if s['status']=='complete':
            row.update(s['final_test'],training_s=s['training_s'],end_to_end_s=s['end_to_end_s'],
                       peak_process_memory_mib=s['peak_process_memory_mib'])
            # Correct derived labels without rewriting historical measurements.
            row['custom_threshold_pass_rate']=row.pop('success_rate')
            row['local_termination_rate']=row.pop('fall_rate')
        rows.append(row)
        if (path/'curve.jsonl').exists():
            curves[label]=[json.loads(line) for line in (path/'curve.jsonl').read_text().splitlines() if line.strip()]
    search_state=read(search/'status.json')
    experiment=read(root/'protocol.json')
    search_history=[]
    for name in experiment.get('agent_search_history',[str(search)]):
        directory=Path(name)
        if not directory.is_absolute():directory=ROOT/directory
        if not (directory/'status.json').exists():continue
        stage_state=read(directory/'status.json')
        duration=stage_state.get('elapsed_s')
        if duration is None and stage_state.get('finished_unix'):
            duration=stage_state['finished_unix']-stage_state['started_unix']
        tokens={}
        for f in directory.glob('responses/*_api.json'):
            for k,v in read(f).get('usage',{}).items():
                if type(v) is int:tokens[k]=tokens.get(k,0)+v
        search_history.append({'directory':str(directory),'status':stage_state['status'],
            'completed_candidates':stage_state.get('completed_candidates',0),
            'accepted_count':stage_state.get('accepted_count',0),'process_elapsed_s':duration,
            'started_unix':stage_state['started_unix'],'finished_unix':stage_state.get('finished_unix'),
            'api_usage':tokens})
    completed_searches=[x for x in search_history if x['finished_unix'] is not None]
    search_timing={'process_elapsed_sum_s':sum(x['process_elapsed_s'] for x in completed_searches),
        'wall_clock_span_s':max(x['finished_unix'] for x in completed_searches)-min(x['started_unix'] for x in search_history)
                             if completed_searches else None,
        'note':'Includes failed/superseded searches. Process durations overlap; their sum is not wall-clock elapsed or exclusive GPU compute. Standalone timestep diagnostics are listed separately.'}
    search_reports=[read(p) for p in sorted(search.glob('candidates/*/report.json'))]
    candidates=[]
    for r in search_reports:
        candidates.append({'id':r['id'],'accepted':r['accepted'],'stage':r['stage'],
                           'reasons':r.get('reasons',[]),'nominal_metrics':r.get('nominal_metrics',{})})
    measured=[r for r in search_reports if r.get('nominal')]
    failures=Counter(reason for r in measured for reason in r['nominal']['reasons'])
    closest=min(measured,key=lambda r:r['nominal_metrics']['velocity_rmse_m_s']) if measured else None
    search_diagnosis={'nominal_trials':len(measured),
        'randomized_trials':sum(len(r.get('randomized',[])) for r in search_reports),
        'nominal_finite_and_complete':sum(r['nominal_metrics']['finite'] and r['nominal_metrics']['completed'] for r in measured),
        'randomized_finite_and_complete':sum(t['metrics']['finite'] and t['metrics']['completed'] for r in search_reports for t in r.get('randomized',[])),
        'nominal_falls':sum(r['nominal_metrics']['fell'] for r in measured),
        'nominal_failure_counts':dict(failures),
        'lowest_velocity_rmse_candidate':closest['id'] if closest else None,
        'lowest_velocity_rmse_candidate_metrics':closest['nominal_metrics'] if closest else None}
    write(root/'results.json',{'stage':stage,'training_seeds':[42],'groups':rows,
                              'metric_scope':{'course_score':None,'course_evaluation_performed':False,
                                  'return':'Local motion-tracking reward, not the course scoring reward',
                                  'custom_threshold_pass_rate':'Former success_rate; assistant-selected diagnostic thresholds, not a course or paper success standard',
                                  'local_termination_rate':'Former fall_rate; local task termination fraction, not a course fall definition',
                                  'agent_filter':'Original assistant-selected experimental filter; rejection does not establish training unusability'},
                              'agent_search':search_state,'candidate_results':candidates,
                              'search_history':search_history,'search_timing':search_timing,
                              'search_diagnosis':search_diagnosis})
    keys=['group','status','normalized_return','custom_threshold_pass_rate','local_termination_rate','velocity_rmse_m_s',
          'heading_rms_deg','joint_position_rmse_rad','root_position_rmse_m',
          'training_s','end_to_end_s','peak_process_memory_mib']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
    for row in rows:writer.writerow({k:row.get(k) for k in keys})
    (root/'comparison.csv').write_text(stream.getvalue())
    panels=[('real_steps','normalized_return','Environment steps','Normalized return'),
            ('wall_clock_h','normalized_return','Wall-clock training time (h)','Normalized return'),
            ('real_steps','velocity_rmse_m_s','Environment steps','Velocity RMSE (m/s)'),
            ('real_steps','heading_rms_deg','Environment steps','Heading RMS (deg)')]
    plt.rcParams.update({'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(x,y,xlabel,ylabel) in zip(axes.flat,panels):
        for label,curve in curves.items():
            ax.plot([r[x] for r in curve],[r[y] for r in curve],label=label,lw=2)
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
        if y=='normalized_return':ax.set_ylim(-.02,1.02)
        if curves:ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Sesame · local research evaluation · seed 42 · held-out validation')
    fig.savefig(root/'learning_curves.png',dpi=200);fig.savefig(root/'learning_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    recorded=[r for r in search_reports if r.get('nominal_metrics')]
    gate=read(search/'protocol.json')['config']['gates']
    checks=[('velocity_rmse_m_s','Velocity RMSE (m/s)',gate['velocity_rmse_max_m_s']),
            ('contact_foot_speed_rms_m_s','Contact foot speed RMS (m/s)',gate['contact_foot_speed_rms_max_m_s']),
            ('heading_rms_deg','Heading RMS (deg)',gate['heading_rms_max_deg'])]
    for ax,(metric,label,limit) in zip(axes,checks):
        valid=[(i+1,r['nominal_metrics'][metric]) for i,r in enumerate(recorded) if r['nominal_metrics'].get(metric) is not None]
        if valid:ax.plot([x for x,y in valid],[y for x,y in valid],'o-',lw=1.5,label='Executed candidate')
        ax.axhline(limit,color='#c75924',ls='--',label='Original experimental threshold')
        ax.set(xlabel='Executed candidate index',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Agent reference diagnostics · nominal trials · not course requirements')
    fig.savefig(root/'candidate_validation.png',dpi=200);fig.savefig(root/'candidate_validation.pdf');plt.close(fig)
    stage_text={'complete':'训练及对照已完成',
        'agent_training_blocked_no_accepted_references':'原始数据对照训练完成；Agent 增强分组未训练（原自定筛选未导出轨迹）',
        'running':'进行中',
        'ppo_training_agent_blocked_by_reference_validation':'PPO 对照训练中；Agent 轨迹未通过原自定筛选'}
    lines=['# Agent 参考轨迹重训对照','',f'状态：{stage_text.get(stage,stage)}。训练 seed=42，单种子试验。','',
        '2026-11-13 更正：课程 PDF 第 2、4 节要求按统一环境的 rollout 轨迹回报评分，明确不另设小组自定义评分标准或成功率达标线。本实验的速度、偏航、滑脚及位置误差阈值由助手自行设置，既不是课程要求，也不是 SuperMimic 论文的成功标准。此前由“0/12 通过”推断“轨迹不能用于训练”、由“成功率 0%”判断课程任务未完成，均缺乏依据。','',
        '以下是本地参考动作跟踪实验；尚未在课程统一评测环境中运行，不能提供正式课程评分或达标结论。训练参考数量也不是课程提交门槛。','',
        '各训练组使用相同的官方 SuperMimic/mjlab 源码、0.5 ms 物理步长、50 Hz 控制频率、512 个并行环境、2,097,152 步和独立测试参考。测试集为 8 条参考，每条 9 秒。原有 2 ms 实验不作为本轮配对对照。','',
        '按用户确认，仅删除生成阶段的人为限幅、关节角投影与强制平滑；结果检验、机器人物理关节限位和电机模型保留。平滑仍可由 Agent 显式选择，默认关闭。','',
        f"Agent 搜索完成 {search_state.get('completed_candidates',0)} 个候选，其中 {search_state.get('accepted_count',0)} 个通过原实验自定筛选。旧流程据此决定是否导出数据，这是本轮方法设计，不是训练数据必须满足的通用条件。",'',
        '| 方法 | 本地归一化回报 | 本地提前终止率 | 速度误差 m/s | 偏航 RMS ° | 训练秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        display={'supermimic_original_seed42':'SuperMimic（原始参考）','ppo_original_seed42':'PPO（原始参考）',
                 'supermimic_agent_augmented_seed42':'SuperMimic（Agent 增强参考）'}[row['group']]
        if row['status']=='complete':
            lines.append(f"| {display} | {row['normalized_return']:.4f} | {row['local_termination_rate']:.1%} | {row['velocity_rmse_m_s']:.4f} | {row['heading_rms_deg']:.2f} | {row['training_s']:.1f} |")
        else:
            status_text={'not_started':'未训练','training':'训练中','initializing':'初始化'}.get(row['status'],row['status'])
            lines.append(f"| {display} | {status_text} | — | — | — | — |")
    if search_state.get('accepted_count',0)==0:
        lines+=['','Agent 分组未启动：原自定筛选没有导出轨迹。当前只能说尚未测量 Agent 数据对训练的影响，不能断言这些轨迹没有训练价值，也不能声称 Agent 已经改善了训练效果。']
        if closest:
            m=closest['nominal_metrics']
            lines+=['',f"最终搜索共执行 {search_diagnosis['nominal_trials']} 次无扰动和 {search_diagnosis['randomized_trials']} 次扰动测试，其中分别有 {search_diagnosis['nominal_finite_and_complete']} 和 {search_diagnosis['randomized_finite_and_complete']} 次数值有效且完整执行。无扰动测试中按本地判定记录摔倒 {search_diagnosis['nominal_falls']} 次。这些执行事实与自定质量筛选是否通过分开报告。",'',
                f"按速度 RMSE 最小选出的候选为 `{closest['id']}`：速度 RMSE={m['velocity_rmse_m_s']:.4f} m/s，接触足端速度 RMS={m['contact_foot_speed_rms_m_s']:.4f} m/s。这两项仅作诊断。完整记录见 results.json。",'',
                '筛选的作用应通过“使用 Agent 参考”及“是否启用自定筛选”的对照和消融检验，同时对原始数据采用一致口径；不能只过滤 Agent 数据后就断言其无效。相关训练尚未进行。']
    lines+=['','曲线来自验证集，上表来自 final.pt 的独立测试集。旧日志的 success_rate 表示全程不提前终止且同时满足助手自定的速度 RMSE≤0.05 m/s、偏航 RMS≤10°、关节 RMSE≤0.25 rad、足端 RMSE≤0.02 m、机身位置 RMSE≤0.10 m；它不是课程成功率。新汇总将该字段明确标为 custom_threshold_pass_rate，保留原数值用于追溯，不再把它画作成功率。旧 fall_rate 实际统计本地任务提前终止，汇总改名为 local_termination_rate。原始日志和权重未改写。',
        '',f"最终一轮 Agent 数据生成和过滤耗时：{search_state['elapsed_s']:.1f} 秒。全部已结束搜索阶段（含失败/被替换阶段）的进程用时之和为 {search_timing['process_elapsed_sum_s']:.1f} 秒；搜索墙钟跨度为 {search_timing['wall_clock_span_s']:.1f} 秒。部分搜索与训练重叠，不能把进程用时相加当作真实墙钟时间或独占 GPU 计算量。",'',
        '训练秒数计入采样、更新与首次编译，排除环境初始化和评估；包含初始化及评估的总时长、进程峰值显存见 comparison.csv。数据准备开销另列，不可省略。同卡还有其他作业，不能把耗时差直接当作算法加速。逐阶段记录及 API token 用量见 results.json 的 search_history。','',
        '0.5 ms 减轻了诊断中观察到的 2 ms 数值振荡，但 0.25 ms 对照仍有接触/航向差异，尚未完成严格的步长收敛证明。原实验配置保留用于复查；本次更正没有重新分类或重新训练模型。','',
        '原始日志、配置、权重和曲线均保留；没有改动已有 validation/test 数据，也没有改动两个官方源码目录。','',
        '文件：[训练曲线 PDF](learning_curves.pdf) · [候选检验 PDF](candidate_validation.pdf) · [结果表 CSV](comparison.csv) · [完整记录](results.json)。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--search',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();search=a.search.resolve()
    state={'status':'waiting_for_original_supermimic','pid':os.getpid(),'started_unix':time.time()}
    write(root/'status.json',state)
    env=os.environ.copy();env.update(XDG_CACHE_HOME=str(ROOT/'.runtime/research_cache'),WANDB_MODE='disabled',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4')
    def wait_for(path):
        while True:
            if path.exists():
                s=read(path)
                if s['status']=='complete':return s
                if s['status']=='failed':raise RuntimeError(f'Child failed; inspect {path}')
            time.sleep(5)
    def train(method,name,motion):
        output=root/name
        if (output/'status.json').exists() and read(output/'status.json')['status']=='complete':return
        if (output/'protocol.json').exists():raise RuntimeError(f'Review incomplete run: {output}')
        cmd=[str(PY),'-u','scripts/train_research_sesame.py','--method',method,'--output',str(output),
             '--envs','512','--steps','2097152','--batch','2048','--width','512','--horizon','32',
             '--sampling-mode','uniform','--physics-dt','0.0005','--motion-file',str(motion)]
        state.update(status='training',current=name,command=cmd)
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;write(root/'status.json',state)
            if child.wait():raise RuntimeError(f'Training failed: {name}')
    try:
        wait_for(root/'supermimic_original_seed42/status.json')
        original=ROOT/'datasets/supermimic_research_sesame_20261112/train'
        train('ppo','ppo_original_seed42',original)
        state.update(status='waiting_for_agent_references');write(root/'status.json',state)
        result=wait_for(search/'status.json')
        if result['accepted_count']:
            dataset=root/'agent_augmented_train'
            pack_augmented(original,search/'accepted',dataset)
            train('supermimic','supermimic_agent_augmented_seed42',dataset)
            state['status']='complete'
        else:state['status']='agent_training_blocked_no_accepted_references'
        summarize(root,search,state['status'])
        state['finished_unix']=time.time();write(root/'status.json',state)
    except BaseException as e:
        state.update(status='failed',error=str(e),finished_unix=time.time());write(root/'status.json',state)
        raise


if __name__=='__main__':main()
