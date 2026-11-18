"""Six paper-axis-aligned panels from measured runs; never synthesize missing data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter


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
            label="SuperMimic (official core, Sesame adapter)" if run["method"]=="supermimic" else "PPO-Sim"
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
        for r in selected:
            if r["status"].get("time_to_target") is None:
                y=r["status"]["training_s"]/3600
                ax[5].plot(r["horizon"],y,"x",color=blue)
                ax[5].annotate("not reached",(r["horizon"],y),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
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
    complete=sum(r["status"]["status"]=="complete" for r in runs)
    fig.suptitle(f"Sesame quadruped · RTX 4090 D (shared) · seed 42 · {complete}/{len(runs)} runs complete",fontsize=13)
    fig.savefig(out/"six_panels.png");fig.savefig(out/"six_panels.pdf")
    plt.close(fig)
    # Export the actual rows for auditing or alternate plotting.
    (out/"plotted_data.json").write_text(json.dumps(runs,indent=2)+"\n")
    (out/"notes.txt").write_text("Axes follow the six panels in the supplied paper screenshot.\n"
        "Normalized return is fixed-horizon reward / analytic upper bound, not success rate.\n"
        "A single training seed has no between-seed uncertainty band.\n"
        "Shared GPU times and tiny reference datasets are not numerically comparable to the paper's H100 benchmark.\n"
        "Missing experiments remain pending; x marks are right-censored target times.\n")
    print(out)


if __name__=="__main__":main()
