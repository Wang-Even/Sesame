"""Summarize recorded collection reports; no model/physics computation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def summarize(run):
    status=read(run/"status.json")
    protocol=read(run/"protocol.json")
    paths=sorted(run.glob("candidates/candidate_*/attempt_*/report.json"))
    reports=[read(p) for p in paths]
    executed=[r for r in reports if "nominal" in r]
    trials=[t for r in executed for t in [r["nominal"],*r.get("randomized",[])]]
    accepted=[r for r in reports if r.get("accepted") is True]
    failures=Counter(reason for r in reports for reason in r.get("reasons",[]) if not reason.startswith("randomized_"))
    manifest=read(run/"accepted/manifest.json") if (run/"accepted/manifest.json").exists() else None
    if status["status"]=="complete":
        if manifest is None or len(accepted)!=manifest["accepted_count"] or len(accepted)!=status["accepted_count"]:
            raise ValueError("Accepted counts disagree")
        if not accepted and (run/"accepted/train").exists():raise ValueError("Empty accepted set unexpectedly has training data")
        for entry in manifest["entries"]:
            if sha(Path(entry["trace"]))!=entry["trace_sha256"] or sha(Path(entry["validation"]))!=entry["validation_sha256"]:
                raise ValueError("An admitted trace or validation report changed")
    result={"run":str(run),"status":status["status"],
        "candidate_budget":protocol["candidates"],"completed_candidates":status.get("completed_candidates"),
        "attempts":len(reports),"generated_reference_files":len(list(run.glob("candidates/candidate_*/attempt_*/candidate_reference.npz"))),
        "unique_generated_motion_arrays":len({r["trajectory_sha256"] for r in reports if r.get("trajectory_sha256")}),
        "kinematic_pass_versions":sum(r.get("kinematics",{}).get("passed") is True for r in reports),
        "nominal_executions":len(executed),
        "unique_executed_motion_arrays":len({r["trajectory_sha256"] for r in executed}),
        "total_physical_trials":len(trials),"complete_physical_trials":sum(t["metrics"]["completed"] for t in trials),
        "trials_with_local_fall":sum(t["metrics"]["fell"] for t in trials),
        "accepted_count":len(accepted),"stage_counts":dict(Counter(r["stage"] for r in reports)),
        "nominal_or_generation_failure_counts":dict(failures.most_common()),
        "api_calls":status["api_calls"],"api_errors":status.get("api_errors",0),"candidate_errors":status.get("candidate_errors",0),
        "elapsed_s":status.get("elapsed_s"),"peak_process_memory_mib":status.get("peak_process_memory_mib"),
        "gates":protocol["config"]["gates"],"protocol_sha256":sha(run/"protocol.json"),
        "note":"Adaptive, correlated candidates, not independent trials. Kinematic validity alone is not successful walking. Existing filter thresholds are experimental, not course requirements."}
    metrics=("mean_forward_speed_m_s","mean_speed_error_m_s","velocity_rmse_m_s","heading_rms_deg",
             "heading_max_deg","root_tracking_rmse_m","contact_foot_speed_rms_m_s","fell","completed")
    with (run/"collection_attempts.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=["id","stage","accepted","trajectory_sha256",*metrics,"reasons"])
        writer.writeheader()
        for r in reports:
            row={k:r.get(k) for k in ("id","stage","accepted","trajectory_sha256")}
            row.update({k:r.get("nominal_metrics",{}).get(k) for k in metrics})
            row["reasons"]="; ".join(r.get("reasons",[]));writer.writerow(row)
    if executed:
        result["nominal_forward_speed_range_m_s"]=[min(r["nominal_metrics"]["mean_forward_speed_m_s"] for r in executed),
                                                   max(r["nominal_metrics"]["mean_forward_speed_m_s"] for r in executed)]
    write(run/"collection_summary.json",result)
    return result


def plot(run):
    """Standard scientific plotting of already recorded scalar measurements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports=[read(p) for p in sorted(run.glob("candidates/candidate_*/attempt_*/report.json"))]
    executed=[r for r in reports if "nominal" in r]
    config=read(run/"protocol.json")["config"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),layout="constrained")
    kin=[];success=[];nk=ns=0
    for r in reports:
        nk+=r.get("kinematics",{}).get("passed") is True;ns+=r.get("accepted") is True
        kin.append(nk);success.append(ns)
    x=list(range(1,len(reports)+1))
    axs[0].plot(x,kin,color="#1765ac",label="Kinematic checks passed")
    axs[0].plot(x,success,color="#d36a21",label="Admitted training references",linewidth=2)
    axs[0].set(xlabel="Code versions evaluated (including repairs)",ylabel="Cumulative count",title="(a) Collection outcomes")
    axs[0].legend(frameon=False);axs[0].grid(alpha=.2)
    target=config["target_speed_m_s"]
    axs[1].axhline(target,color="#1765ac",linestyle="--",label=f"Target: {target:g} m/s")
    rejected=[(i+1,r["nominal_metrics"]["mean_forward_speed_m_s"]) for i,r in enumerate(executed) if not r["accepted"]]
    passed=[(i+1,r["nominal_metrics"]["mean_forward_speed_m_s"]) for i,r in enumerate(executed) if r["accepted"]]
    for data,color,marker,label in [(rejected,"#d36a21","x","Failed admission"),(passed,"#1765ac","o","Passed all checks")]:
        if data:axs[1].scatter([v[0] for v in data],[v[1] for v in data],color=color,marker=marker,label=label)
    axs[1].set(xlabel="Nominal simulation execution",ylabel="Measured mean forward velocity (m/s)",title="(b) Actual forward motion")
    axs[1].legend(frameon=False);axs[1].grid(alpha=.2)
    for extension in ("png","pdf"):fig.savefig(run/("collection_results."+extension),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--plot",action="store_true");a=p.parse_args()
    print(json.dumps(summarize(a.run),ensure_ascii=False,indent=2))
    if a.plot:plot(a.run)


if __name__=="__main__":main()
