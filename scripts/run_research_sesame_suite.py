"""Gated sequential rerun with official research mjlab; leaves old runs intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/".venv_supermimic_research/bin/python"
DATA=ROOT/"datasets/supermimic_research_sesame_20261112"


def write(path,data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--main-only",action="store_true")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    gates=a.output/"validation";gates.mkdir(exist_ok=True)
    audits=[]
    for name in ["supermimic_official_5efbf114","mjlab_research_dd51a9fc"]:
        provenance=json.loads((ROOT/"third_party"/(name+".provenance.json")).read_text())
        changed=[rel for rel,sha in provenance["files_sha256"].items()
                 if hashlib.sha256((ROOT/"third_party"/name/rel).read_bytes()).hexdigest()!=sha]
        if changed:raise RuntimeError(f"Official source changed: {name}: {changed}")
        audits.append({"source":name,"commit":provenance["commit"],"files_checked":len(provenance["files_sha256"]),"changed":changed})
    write(a.output/"source_audit.json",audits)
    runs=[dict(name="supermimic_seed42",method="supermimic",group="main",clips=24,horizon=32,critic=True,sampling="uniform"),
          dict(name="ppo_seed42",method="ppo",group="main",clips=24,horizon=32,critic=True,sampling="uniform")]
    # Pair both algorithms under the official return-based reset distribution.
    runs += [dict(name=f"{method}_return_seed42",method=method,group="reset",clips=24,horizon=32,critic=True,sampling="return")
             for method in ("supermimic","ppo")]
    if not a.main_only:
        runs += [dict(name=f"dataset_{n}_seed42",method="supermimic",group="dataset",clips=n,horizon=32,critic=True,sampling="uniform") for n in (6,12)]
        runs += [dict(name=f"h{h}_{'critic' if c else 'no_critic'}_seed42",method="supermimic",group="horizon",clips=24,horizon=h,critic=c,sampling="uniform")
                 for h in (8,16,32,64,128) for c in (True,False) if (h,c)!=(32,True)]
    manifest={"official_commit":"5efbf114e11f725c31d83e217be987c13c0e4ce2",
              "mjlab_commit":"dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd","seed":42,
              "steps_per_run":2097152,"envs":512,"batch":2048,"width":512,"runs":runs,
              "source_note":"Native official research mjlab; Sesame robot/task configuration; no motion/FK replacements",
              "main_label":"SuperMimic (official research mjlab)",
              "figure_title":"Sesame quadruped · official mjlab research · RTX 4090 D · seed 42",
              "reset_control":"Main and six-panel sweep: uniform RSI. Additional paired return-RSI experiments.",
              "time_note":"Warm compile caches; sampling+updates excludes setup and evaluation; GPU may be shared.",
              "gpu_budget":"14 GiB whole-process guard; 50% Torch allocator cap"}
    if (a.output/"suite.json").exists() and json.loads((a.output/"suite.json").read_text())!=manifest:
        raise RuntimeError("Refusing to change an existing experiment protocol")
    write(a.output/"suite.json",manifest)
    write(a.output/"input_data_manifest.json",json.loads((DATA/"manifest.json").read_text()))
    env=os.environ.copy()
    env.update(WANDB_MODE="disabled",XDG_CACHE_HOME=str(ROOT/".runtime/research_cache"),
               TORCHINDUCTOR_COMPILE_THREADS="4",OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1")
    state={"status":"validating","pid":os.getpid(),"started_unix":time.time(),"completed":[]}
    write(a.output/"status.json",state)

    def run_child(cmd,logfile,label):
        state.update(current=label,command=cmd)
        with logfile.open("w") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid;write(a.output/"status.json",state)
            rc=child.wait()
        if rc:raise RuntimeError(f"{label} failed with exit {rc}; see {logfile}")

    def smoke(h):
        output=gates/f"native_h{h}"
        result=output/"result.json"
        if result.exists() and json.loads(result.read_text())["status"]=="passed":return
        cmd=[str(PY),"-u",str(ROOT/"scripts/smoke_research_sesame.py"),"--envs","512","--batch","2048",
             "--width","512","--horizon",str(h),"--wm-batches","10","--wm-batch","512","--output",str(output)]
        run_child(cmd,gates/f"native_h{h}.log",f"native validation H={h}")

    try:
        smoke(32)
        for method in ("supermimic","ppo"):
            output=gates/f"{method}_driver"
            status=output/"status.json"
            if status.exists() and json.loads(status.read_text()).get("status")=="complete":continue
            if (output/"protocol.json").exists():raise RuntimeError(f"Review incomplete driver gate: {output}")
            cmd=[str(PY),"-u",str(ROOT/"scripts/train_research_sesame.py"),"--method",method,
                 "--output",str(output),"--envs","512","--steps","16384","--batch","2048",
                 "--sampling-mode","return"]
            run_child(cmd,gates/f"{method}_driver.log",f"{method} full driver validation")
            result=json.loads(status.read_text())
            if result.get("reset_sampler",{}).get("observed_samples",0)<=0:
                raise RuntimeError("Official return sampler did not update")
        state["status"]="training"
        for spec in runs:
            if spec["horizon"] in (64,128):smoke(spec["horizon"])
            output=a.output/spec["name"]
            status=output/"status.json"
            if status.exists() and json.loads(status.read_text()).get("status")=="complete":
                state["completed"].append(spec["name"]);continue
            if (output/"protocol.json").exists():raise RuntimeError(f"Review incomplete training run: {output}")
            cmd=[str(PY),"-u",str(ROOT/"scripts/train_research_sesame.py"),"--method",spec["method"],
                 "--output",str(output),"--envs","512","--steps","2097152","--batch","2048","--width","512",
                 "--horizon",str(spec["horizon"]),"--sampling-mode",spec["sampling"],
                 "--motion-file",str(DATA/("train" if spec["clips"]==24 else f"train_{spec['clips']}"))]
            if not spec["critic"]:cmd.append("--no-critic")
            run_child(cmd,a.output/(spec["name"]+".log"),spec["name"])
            state["completed"].append(spec["name"]);write(a.output/"status.json",state)
            subprocess.run([str(PY),str(ROOT/"scripts/plot_research_sesame.py"),"--suite",str(a.output)],cwd=ROOT,env=env,check=True)
        state.update(status="complete",finished_unix=time.time())
        write(a.output/"status.json",state)
    except BaseException as e:
        state.update(status="failed",error=str(e),finished_unix=time.time());write(a.output/"status.json",state)
        raise


if __name__=="__main__":main()
