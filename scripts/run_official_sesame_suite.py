"""Sequential, resumable experiment queue. Never starts an unvalidated config."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/".venv_supermimic_official/bin/python"


def write(path,data):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--main-only",action="store_true");a=p.parse_args()
    gates=[ROOT/"runs/official_compat_20261111/memory_512_2048_h32/result.json",
           ROOT/"runs/official_compat_20261111/dc_interface/result.json"]
    for gate in gates:
        if not gate.exists() or json.loads(gate.read_text())["status"]!="passed":
            raise RuntimeError(f"Compatibility / memory gate has not passed: {gate}")
    driver_gate=ROOT/"runs/official_compat_20261111/driver_smoke/status.json"
    if not driver_gate.exists() or json.loads(driver_gate.read_text())["status"]!="complete":
        raise RuntimeError("Formal training/evaluation driver has not passed its smoke run")
    provenance=json.loads((ROOT/"third_party/supermimic_official_5efbf114.provenance.json").read_text())
    for rel,sha in provenance["files_sha256"].items():
        path=ROOT/"third_party/supermimic_official_5efbf114"/rel
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f"Official source changed: {rel}")
    a.output.mkdir(parents=True,exist_ok=True)
    runs=[dict(name="supermimic_seed42",method="supermimic",group="main",clips=24,horizon=32,critic=True),
          dict(name="ppo_seed42",method="ppo",group="main",clips=24,horizon=32,critic=True)]
    if not a.main_only:
        runs += [dict(name=f"dataset_{n}_seed42",method="supermimic",group="dataset",clips=n,horizon=32,critic=True) for n in (6,12)]
        runs += [dict(name=f"h{h}_{'critic' if c else 'no_critic'}_seed42",method="supermimic",group="horizon",clips=24,horizon=h,critic=c)
                 for h in (8,16,32,64,128) for c in (True,False) if (h,c)!=(32,True)]
    manifest={"official_commit":provenance["commit"],"seed":42,"steps_per_run":2097152,
              "envs":512,"batch":2048,"width":512,"runs":runs,
              "horizon_scope":"8,16,32,64,128; larger horizons require their own memory probe",
              "gpu_budget":"14 GiB whole-process guard; 12 GiB Torch allocator cap",
              "source_note":"Official algorithm with public-mjlab Sesame adapter; private research dependency missing"}
    if (a.output/"suite.json").exists() and json.loads((a.output/"suite.json").read_text())!=manifest:
        raise RuntimeError("Existing suite has a different protocol")
    write(a.output/"suite.json",manifest)
    # Subsets preserve complete clips; selection fixed before formal outcomes.
    import numpy as np
    data_dir=ROOT/"datasets/supermimic_official_sesame_20261111"
    with np.load(data_dir/"train.npz",allow_pickle=False) as f:
        original={k:f[k] for k in f.files}
    for n in (6,12):
        count=int(original["motion_ends"][n-1]); total=int(original["motion_ends"][-1])
        subset={k:(v[:n] if k=="motion_ends" else v[:count] if v.ndim and len(v)==total else v) for k,v in original.items()}
        file=data_dir/f"train_{n}.npz"
        if not file.exists():np.savez_compressed(file,**subset)
    env=os.environ.copy()
    env.update(WANDB_MODE="disabled",XDG_CACHE_HOME=str(ROOT/".runtime/official_cache"),
               TORCHINDUCTOR_COMPILE_THREADS="4",OMP_NUM_THREADS="4")
    state={"status":"running","pid":os.getpid(),"started_unix":time.time(),"completed":[]}
    for spec in runs:
        if spec["horizon"] in (64,128):
            probe=ROOT/f"runs/official_compat_20261111/memory_512_2048_h{spec['horizon']}"
            result=probe/"result.json"
            if not result.exists():
                probe_cmd=[str(PY),"-u",str(ROOT/"scripts/smoke_official_sesame.py"),"--envs","512",
                    "--batch","2048","--horizon",str(spec["horizon"]),"--width","512",
                    "--wm-batches","10","--wm-batch","512","--output",str(probe)]
                with (a.output/f"memory_h{spec['horizon']}.log").open("w") as log:
                    rc=subprocess.run(probe_cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if rc:
                    state.update(status="failed",error=f"Memory probe failed for H={spec['horizon']}; no training launched")
                    write(a.output/"status.json",state);raise RuntimeError(state["error"])
            if json.loads(result.read_text())["status"]!="passed":raise RuntimeError("Memory gate failed")
        output=a.output/spec["name"]
        existing=json.loads((output/"status.json").read_text()) if (output/"status.json").exists() else {}
        if existing.get("status")=="complete":
            state["completed"].append(spec["name"]);continue
        if (output/"protocol.json").exists():
            state.update(status="failed",error=f"Incomplete prior run requires explicit review: {output}")
            write(a.output/"status.json",state);raise RuntimeError(state["error"])
        cmd=[str(PY),"-u",str(ROOT/"scripts/train_official_sesame.py"),"--method",spec["method"],
            "--output",str(output),"--envs","512","--steps","2097152","--batch","2048", "--width","512",
            "--horizon",str(spec["horizon"]),"--motion-file",str(data_dir/("train.npz" if spec["clips"]==24 else f"train_{spec['clips']}.npz"))]
        if not spec["critic"]:cmd.append("--no-critic")
        state.update(current=spec["name"],command=cmd)
        write(a.output/"status.json",state)
        with (a.output/(spec["name"]+".log")).open("w") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid;write(a.output/"status.json",state)
            rc=child.wait()
        plot=[str(PY),str(ROOT/"scripts/plot_official_sesame.py"),"--suite",str(a.output)]
        subprocess.run(plot,cwd=ROOT,env=env,check=True)
        if rc:
            state.update(status="failed",exit_code=rc);write(a.output/"status.json",state)
            raise RuntimeError(f"Training failed for {spec['name']}; queue stopped")
        state["completed"].append(spec["name"])
    state.update(status="complete",finished_unix=time.time())
    write(a.output/"status.json",state)


if __name__=="__main__":main()
