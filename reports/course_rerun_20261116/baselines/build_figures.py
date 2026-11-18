"""Read historical metadata and draw report figures; no model computation."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/home/ywwang/sesame')
OUT=Path(__file__).resolve().parent
ANALYSIS=ROOT/'runs/offpolicy_heading_20261107/analysis'
curve_rows=list(csv.DictReader((ANALYSIS/'training_curve_data.csv').open()))
group_rows=list(csv.DictReader((ANALYSIS/'group_summary.csv').open()))
methods=['dqn','ppo','sac']
colors={'dqn':'#777777','ppo':'#c95725','sac':'#1259a6'}
times={}
sources=[ANALYSIS/'training_curve_data.csv',ANALYSIS/'group_summary.csv']
for method in methods:
    for seed in (42,43,44):
        folder=ROOT/(f'runs/ppo_heading_20261105/training/heading_seed{seed}' if method=='ppo' else f'runs/offpolicy_heading_20261107/training/{method}_seed{seed}')
        file=folder/'training.jsonl'
        sources.append(file)
        cumulative=0.
        for line in file.read_text().splitlines():
            row=json.loads(line)
            if method=='ppo':
                cumulative+=row['collection_seconds']+row['update_seconds']
            else:
                cumulative=row['training_wall_seconds']
            times[(method,seed,row['environment_transitions'])]=cumulative

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
metrics=[('mean_episode_return','Training return'),('velocity_x_rmse_m_s','Forward velocity RMSE (m/s)'),
         ('heading_rms_rad','Heading RMS (deg)')]
fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
for col,(metric,title) in enumerate(metrics):
    for method in methods:
        by_step={}
        for seed in (42,43,44):
            rows=[r for r in curve_rows if r['metric']==metric and r['variant']==method and int(r['train_seed'])==seed]
            steps=[int(r['environment_transitions']) for r in rows]
            vals=[float(r['value']) for r in rows]
            elapsed=[times[(method,seed,s)] for s in steps]
            axes[0,col].plot(np.array(steps)/1e6,vals,color=colors[method],lw=.5,alpha=.18)
            axes[1,col].plot(np.array(elapsed)/60,vals,color=colors[method],lw=.8,alpha=.42,
                             label=method.upper() if seed==42 else None)
            for step,value in zip(steps,vals):by_step.setdefault(step,[]).append(value)
        shared=sorted(s for s,v in by_step.items() if len(v)==3)
        means=[statistics.mean(by_step[s]) for s in shared]
        sd=[statistics.stdev(by_step[s]) for s in shared]
        axes[0,col].plot(np.array(shared)/1e6,means,color=colors[method],label=method.upper(),lw=1.8)
        axes[0,col].fill_between(np.array(shared)/1e6,np.array(means)-sd,np.array(means)+sd,color=colors[method],alpha=.12)
    for r in (0,1):
        axes[r,col].set_ylabel(title)
        axes[r,col].grid(alpha=.15)
    axes[0,col].set_xlabel('Environment transitions (million)')
    axes[1,col].set_xlabel('Logged training time (min)')
axes[0,0].legend(frameon=False)
axes[1,0].legend(frameon=False)
fig.savefig(OUT/'baseline_learning_curves.png',dpi=180)
fig.savefig(OUT/'baseline_learning_curves.pdf')
plt.close(fig)

lookup={(r['condition'],r['variant'],r['metric']):r for r in group_rows}
fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
for ax,(metric,title) in zip(axes,[('return','Test return'),('velocity_x_rmse_m_s','Forward velocity RMSE (m/s)'),('heading_rms_deg','Heading RMS (deg)')]):
    for index,method in enumerate(methods):
        xs=np.array([0,1])+(index-1)*.23
        rows=[lookup[c,method,metric] for c in ['nominal','randomized']]
        ax.bar(xs,[float(r['mean']) for r in rows],width=.21,color=colors[method],label=method.upper(),
               yerr=[float(r['sd_across_training_seeds']) for r in rows],capsize=3)
    ax.set_xticks([0,1],['Nominal','Randomized'])
    ax.set_ylabel(title)
    ax.grid(axis='y',alpha=.15)
axes[0].legend(frameon=False)
fig.savefig(OUT/'baseline_test_metrics.png',dpi=180)
fig.savefig(OUT/'baseline_test_metrics.pdf')
plt.close(fig)

selected=[]
for c in ['nominal','randomized']:
    for method in methods:
        row={'condition':c,'method':method,'training_seeds':'42,43,44','test_episodes':384}
        for metric in ['return','velocity_x_rmse_m_s','heading_rms_deg','distance_x_m','fall_rate']:
            source=lookup[c,method,metric]
            row[metric+'_mean']=float(source['mean'])
            row[metric+'_sd']=float(source['sd_across_training_seeds'])
        selected.append(row)
with (OUT/'baseline_test_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
with (OUT/'baseline_curve_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(curve_rows[0])+['logged_training_s']);writer.writeheader()
    for row in curve_rows:
        if row['metric'] not in [m[0] for m in metrics]:continue
        writer.writerow(dict(row,logged_training_s=times[row['variant'],int(row['train_seed']),int(row['environment_transitions'])]))
manifest={'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          'plot_notes':['No model, simulator or GPU work was run for this metadata-only report.',
                        'Training curves retain original log windows: PPO 100 episodes; SAC/DQN 256.',
                        'Top panels: mean and sample SD at common steps across 3 seeds; faint individual curves.',
                        'Bottom panels: individual seed time curves, no interpolated mean.',
                        'PPO time sums collection/update; SAC/DQN use recorded loop wall time.',
                        'heading_rms_rad in archived plot CSV is already multiplied by display_scale=180/pi.'],
          'scene_image':{'path':'isaac_scene.png','source':'runs/offpolicy_heading_20261107/videos/ppo_seed42/offpolicy_ppo_seed42_seed25042.mp4','time_s':2.,'kind':'Frame from recorded actual Isaac RTX policy execution, not an import-tool screenshot.'},
          'go2_reference_url':'https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py',
          'go2_reference_accessed':'2026-11-15'}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps({'rows':len(selected),'output':str(OUT)}))
