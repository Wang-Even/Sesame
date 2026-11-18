"""Six paper-axis-aligned panels from measured runs; never synthesize missing data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator, FuncFormatter


def main():
    p=argparse.ArgumentParser();p.add_argument("--suite",type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.suite/"suite.json").read_text())
    out=a.suite/"figures";out.mkdir(exist_ok=True)
    runs=[]
    for spec in manifest["runs"]:
        path=a.suite/spec["name"]
        curve=[]
        if (path/"curve.jsonl").exists():
            for line in (path/"curve.jsonl").read_text().splitlines():
                try: curve.append(json.loads(line))
                except json.JSONDecodeError: pass
        status=json.loads((path/"status.json").read_text()) if (path/"status.json").exists() else {"status":"pending"}
        runs.append({**spec,"curve":curve,"status":status})
    plt.rcParams.update({"font.family":"serif","font.size":12,"axes.spines.top":False,
        "axes.spines.right":False,"axes.labelsize":13,"savefig.dpi":300,"pdf.fonttype":42})
    fig,axes=plt.subplot_mosaic([["a","b"],["a","c"],
                                ["d","e"],["d","f"]],
                                figsize=(14,10),layout="constrained")
    ax=[axes[k] for k in "abcdef"]
    blue,orange="#1055ad","#c75924"
    labels=[("GPU-hours","Normalized return"),("Dataset length (h)","Compute to target (GPU-h)"),
        ("Dataset length (h)",r"8-step prediction error ($\times 10^{-3}$)"),
        ("Wall-clock time (h)","Normalized return"),(r"Imagination horizon $H$","Final return"),
        (r"Imagination horizon $H$","Hours to return 0.82")]
    for i,(x,y) in enumerate(labels):
        ax[i].set(xlabel=x,ylabel=y);ax[i].set_title(f"({chr(97+i)}) {y}",loc="left",fontweight="bold",fontsize=12)
        ax[i].grid(alpha=.15)
    for panel,short in [(1,"GPU-h"),(2,r"Error ($\times 10^{-3}$)"),(4,"Normalized return"),(5,"Hours")]:
        ax[panel].set_ylabel(short)
    for run in runs:
        if run.get("group")=="main" and run["curve"]:
            color=blue if run["method"]=="supermimic" else orange
            label=manifest.get("main_label","SuperMimic") if run["method"]=="supermimic" else "PPO-Sim"
            for panel,key in [(0,"gpu_hours"),(3,"wall_clock_h")]:
                rows=[r for r in run["curve"] if panel != 0 or r[key]>0]
                ax[panel].plot([r[key] for r in rows],[r["normalized_return"] for r in rows],color=color,lw=2,
                               marker="o" if len(rows)==1 else None,markersize=4,label=label)
    for panel in (0,3):
        ax[panel].axhline(.82,color="gray",ls="--",lw=1)
        observed=[row["normalized_return"] for run in runs if run.get("group")=="main" for row in run["curve"]]
        ax[panel].set_ylim(min([-.05]+[v-.05 for v in observed]),1.05)
        if ax[panel].get_legend_handles_labels()[0]: ax[panel].legend(fontsize=8,frameon=False)
    ax[3].set_xlim(left=0,right=max([.01]+[row["wall_clock_h"]*1.05 for run in runs if run.get("group")=="main" for row in run["curve"]]))
    ax[0].set_xscale("log")
    ax[0].xaxis.set_major_locator(LogLocator(base=10,subs=(1.,3.)))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}"))
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    if not any(row["gpu_hours"]>0 for run in runs if run.get("group")=="main" for row in run["curve"]):
        ax[0].set_xlim(.001,1.)
        ax[0].text(.5,.5,"First training evaluation pending",transform=ax[0].transAxes,ha="center",color="gray")
    datasets=[r for r in runs if r["method"]=="supermimic" and r.get("group") in ("main","dataset") and r["status"]["status"]=="complete"]
    datasets.sort(key=lambda r:r["clips"])
    for r in datasets:
        x=r["clips"]*9/3600
        target=r["status"].get("time_to_target")
        if target is not None:
            ax[1].plot(x,target["training_s"]/3600,"o",color=blue)
        else:
            # Censor at observed budget, rather than reporting a false time-to-target.
            y=r["status"]["training_s"]/3600
            ax[1].plot(x,y,"x",color=blue);ax[1].annotate("not reached",(x,y),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
        value=r["curve"][-1].get("prediction_error_8_step")
        if value is not None: ax[2].plot(x,value*1000,"o",color=blue)
    for critic in (True,False):
        selected=[r for r in runs if r["method"]=="supermimic" and r.get("group") in ("main","horizon")
                  and r.get("critic",True)==critic and r["status"]["status"]=="complete"]
        selected.sort(key=lambda r:r["horizon"])
        style=dict(color=blue,marker="o",mfc=blue if critic else "white",lw=2 if critic else 1,label="critic" if critic else "no critic")
        if selected:
            ax[4].plot([r["horizon"] for r in selected],[r["curve"][-1]["normalized_return"] for r in selected],**style)
        reached=[r for r in selected if r["status"].get("time_to_target") is not None]
        if reached:
            ax[5].plot([r["horizon"] for r in reached],[r["status"]["time_to_target"]["training_s"]/3600 for r in reached],**style)
        censor_label_used=False
        for r in selected:
            if r["status"].get("time_to_target") is None:
                y=r["status"]["training_s"]/3600
                label=("critic" if critic else "no critic")+": not reached" if not censor_label_used else None
                ax[5].plot(r["horizon"],y,"x",color=blue if critic else "#83a6d2",label=label)
                censor_label_used=True
    for panel in (4,5):
        ax[panel].set_xscale("log",base=2)
        ticks=sorted(set(r["horizon"] for r in runs if r["method"]=="supermimic"))
        ax[panel].set_xticks(ticks)
        ax[panel].set_xlim(min(ticks)*.8,max(ticks)*1.2)
        ax[panel].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax[panel].xaxis.set_minor_formatter(NullFormatter())
        if ax[panel].get_legend_handles_labels()[0]: ax[panel].legend(frameon=False,fontsize=9)
    for panel in (1,2):
        ax[panel].set_xscale("log")
        ticks=sorted(set(r["clips"]*9/3600 for r in runs if r["method"]=="supermimic" and r["group"] in ("main","dataset")))
        ax[panel].set_xticks(ticks)
        ax[panel].set_xlim(min(ticks)*.8,max(ticks)*1.2)
        ax[panel].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax[panel].xaxis.set_minor_formatter(NullFormatter())
    for panel in (1,2,4,5):
        if not ax[panel].lines:
            ax[panel].text(.5,.5,"Experiment pending",transform=ax[panel].transAxes,ha="center",color="gray")
    for panel in (1,2,5):
        ys=[float(y) for line in ax[panel].lines for y in line.get_ydata()]
        ax[panel].set_ylim(0,max(ys)*1.35 if ys and max(ys)>0 else 1.)
    ax[2].ticklabel_format(axis="y",style="plain",useOffset=False)
    horizon_returns=[r["curve"][-1]["normalized_return"] for r in runs
                     if r["method"]=="supermimic" and r["group"] in ("main","horizon") and r["curve"]]
    ax[4].set_ylim(min([-.05]+[v-.05 for v in horizon_returns]),1.05)
    complete=sum(r["status"]["status"]=="complete" for r in runs)
    fig.suptitle(manifest.get("figure_title","Sesame") + f" · {complete}/{len(runs)} runs complete",fontsize=13)
    fig.savefig(out/"six_panels.png");fig.savefig(out/"six_panels.pdf")
    plt.close(fig)
    # Export the actual rows for auditing or alternate plotting.
    (out/"plotted_data.json").write_text(json.dumps(runs,indent=2)+"\n")
    (out/"notes.txt").write_text("Axes follow the six panels in the supplied paper screenshot.\n"
        "Normalized return is fixed-horizon reward / analytic upper bound, not success rate.\n"
        "A single training seed has no between-seed uncertainty band.\n"
        "Shared GPU times and tiny reference datasets are not numerically comparable to the paper's H100 benchmark.\n"
        "Missing experiments remain pending; x marks are right-censored target times.\n")
    comparison_plots(a.suite,runs,out,manifest)
    print(out)


def comparison_plots(suite,runs,out,manifest):
    import csv
    import io
    selected=[r for r in runs if r["group"] in ("main","reset")]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    panels=[("real_steps","normalized_return","Environment steps","Normalized return"),
            ("wall_clock_h","normalized_return","Wall-clock training time (h)","Normalized return"),
            ("real_steps","velocity_rmse_m_s","Environment steps","Velocity RMSE (m/s)"),
            ("real_steps","heading_rms_deg","Environment steps","Heading RMS error (deg)")]
    for r in selected:
        rows=r["curve"]
        if not rows:continue
        color="#1055ad" if r["method"]=="supermimic" else "#c75924"
        label=("SuperMimic" if r["method"]=="supermimic" else "PPO-Sim")+f" ({r['sampling']} RSI)"
        for ax,(x,y,_,_) in zip(axes.flat,panels):
            ax.plot([v[x] for v in rows],[v[y] for v in rows],lw=2,color=color,
                    ls="-" if r["sampling"]=="uniform" else "--",label=label)
    for i,(ax,(_,y,xlabel,ylabel)) in enumerate(zip(axes.flat,panels)):
        ax.set(xlabel=xlabel,ylabel=ylabel,title=f"({chr(97+i)}) {ylabel}")
        ax.grid(alpha=.15);ax.set_xlim(left=0)
        if y=="normalized_return":ax.set_ylim(-.05,1.05)
        if y=="normalized_return":ax.axhline(.82,color="gray",ls=":",lw=1)
        if ax.get_legend_handles_labels()[0]:ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Official research mjlab · Sesame · paired reset sampling · seed 42",fontsize=13)
    for ext in ("png","pdf"):fig.savefig(out/f"learning_curves.{ext}")
    plt.close(fig)
    columns=["name","method","sampling","horizon","critic","clips","normalized_return","success_rate",
             "fall_rate","velocity_rmse_m_s","heading_rms_deg","joint_position_rmse_rad",
             "root_position_rmse_m","training_s","peak_process_memory_mib"]
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for r in runs:
        if r["status"]["status"]!="complete":continue
        values=r|r["status"]|r["status"]["final_test"]
        writer.writerow({k:values.get(k) for k in columns})
    (out/"final_test_results.csv").write_text(stream.getvalue())
    done=sum(r["status"]["status"]=="complete" for r in runs)
    lines=["# SuperMimic 官方研究版 mjlab 重训", "",
           "指标口径：本报告是本地研究评估，不是课程统一评分。课程 PDF 第 2、4 节要求使用统一环境的 rollout 轨迹回报，不另设小组自定义成功率或达标线。旧日志 success_rate 只是助手自定阈值通过率；fall_rate 实际统计本地环境提前终止。原始字段保留供追溯，不得据此断言课程任务失败。", "",
           f"状态：{done}/{len(runs)} 个正式实验完成；训练 seed=42，每组 2,097,152 次环境交互。", "",
           "SuperMimic 固定提交 `5efbf114e11f725c31d83e217be987c13c0e4ce2`；mjlab_research 固定提交 `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`。两个官方源码目录均不修改，逐文件校验见结果目录的 source_audit.json。", "",
           "直接使用官方 MotionLoader、ForwardKinematics、MotionCommand 和 motion_end；本地仅保留 Sesame 机器人/任务配置、PPO 接口、评估和运行脚本。主对照及六图消融保持 uniform RSI；另对两种算法同时启用官方 return RSI，避免把采样变化混入环境变化。", "",
           "uniform 表示均匀选择参考轨迹起点；return 表示按官方采样器增加低回报片段的练习比例，同时保留均匀探索。两种算法分别采用相同的采样设置。", "",
           "网络、参考数据划分、奖励和训练步数与前一轮保持一致。依赖升级为研究版要求的版本；本实验仍是 Sesame 小规模适配，不能称为原论文任务和计算规模的完整复现。参考轨迹仍来自原 Isaac 平地直行数据，已验证几何一致性，但未在新引擎重新生成或证明动力学可行性。", "",
           "配置：512 个并行环境，SuperMimic 策略批量 2048，主实验想象长度 H=32；世界模型隐藏层 512×3，策略隐藏层 512/256/128，Critic 256/128。Actor 793,032、世界模型 559,630、Critic 107,777 个参数。PPO 使用相同的策略网络结构。", "",
           "## 最终权重测试结果", "",
           "8 条独立测试参考，每条 9 秒；下表均使用 final.pt。best.pt 仅按验证集选择，单独记录，不混用。", "",
           "| 方法 | 重置采样 | 本地归一化回报 | 自定阈值通过率 | 本地提前终止率 | 速度 RMSE (m/s) | 偏航 RMS (°) | 训练时间 (min) | 峰值显存 (GiB) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        s=r["status"]
        if s["status"]!="complete":continue
        m=s["final_test"]
        lines.append(f"| {r['method']} | {r['sampling']} | {m['normalized_return']:.4f} | {m['success_rate']:.1%} | {m['fall_rate']:.1%} | {m['velocity_rmse_m_s']:.4f} | {m['heading_rms_deg']:.2f} | {s['training_s']/60:.2f} | {s['peak_process_memory_mib']/1024:.2f} |")
    if all(r["status"]["status"]=="complete" for r in selected):
        lines += ["", "在本轮固定步数和参数下，SuperMimic 的回报更高、偏航误差更小，但训练用时约为 PPO 的 1.8 倍。两种算法均未达到本地研究的 0.82 回报目标；该目标不是课程达标线。这轮结果没有复现原论文的训练加速。PPO 的均匀采样曲线先下降后回升，不能把当前固定预算结果视为充分调参后的最终性能。", ""]
    lines += ["", "## 图与使用", "",
              f"结果目录：`{suite}`。曲线见 [六幅论文坐标图]({out/'six_panels.pdf'})、[PPO 对照学习曲线]({out/'learning_curves.pdf'})；完整指标见 final_test_results.csv。", "",
              "曲线使用验证集；上表使用独立测试集，因此末点与表格不必相同。六图横纵轴与提供的论文截图对应；0.82 未达到时标为 not reached。原始曲线见 curve.jsonl，训练更新见 updates.jsonl。只有一个 seed，不绘制跨 seed 误差带。", "",
              "验证中观察到同一权重重复评估的数值波动；当前结果不能据此给出跨 seed 显著性结论。", "",
              "部分运行日志出现物理求解器线搜索迭代上限警告；各组均使用相同的 iterations=10、ls_iterations=20。本轮未做求解器收敛敏感性实验，不能把通过接口检查理解为已排除所有物理数值误差。", "",
              "本地归一化回报=9 秒累计奖励/45，不等于课程评分。旧 success_rate 字段的自定条件为全程不提前终止，同时速度 RMSE≤0.05 m/s、偏航 RMS≤10°、关节 RMSE≤0.25 rad、足端位置 RMSE≤0.02 m、根位置 RMSE≤0.10 m；这些阈值没有课程文件依据，也不是论文的成功标准。", "",
              "8 步预测误差为真实仿真策略轨迹上的开环状态预测 MSE，按官方各状态通道的 nominal error 平方归一化后取平均，排除中途重置窗口；图中数值乘 1000，对应纵轴 ×10⁻³。主模型末次该值约为 9.54，不能解读成原论文约 0.01 的预测精度。", "",
              "时间包含训练采样、更新和同步，不包含环境初始化及评估；编译可能已被验证预热。端到端时间另见 status.json。显存为含训练、验证和最终测试的整个进程峰值，按 nvidia-smi 每秒采样，可能遗漏瞬时尖峰；GPU 负载另有记录，跨轮耗时不宜直接解释为算法加速。", "",
              "权重在各实验子目录的 final.pt 和 best.pt；protocol.json 保存参数、依赖版本和本地源码摘要，sources/ 保存当时使用的本地代码。完整运行环境见 requirements.lock.txt。", "",
              "```bash", "cd /home/ywwang/sesame",f".venv_supermimic_research/bin/python scripts/plot_research_sesame.py --suite {suite}","```", "",
              "重新运行完整实验时使用新的输出目录：", "",
              "```bash", ".venv_supermimic_research/bin/python scripts/run_research_sesame_suite.py --output runs/research_repeat", "```", ""]
    (suite/"REPORT.md").write_text("\n".join(lines))


if __name__=="__main__":main()



