"""Lossless array repacking for native mjlab_research MotionLoader (no simulation)."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    src=ROOT/"datasets/supermimic_official_sesame_20261111"
    dst=ROOT/"datasets/supermimic_research_sesame_20261112"
    dst.mkdir(exist_ok=True)
    manifest={"source":str(src),"format":"official mjlab_research packed root/joint trajectories","splits":{}}
    for split in ["train","validation","test","train_6","train_12"]:
        p=src/(split+".npz")
        with np.load(p,allow_pickle=False) as f:d={k:f[k] for k in f.files}
        assert int(d["fps"])==50
        out=dst/split;out.mkdir(exist_ok=True)
        arrays={"root_pos_w":d["body_pos_w"][:,0],"root_quat_w":d["body_quat_w"][:,0],
                "root_lin_vel_w":d["body_lin_vel_w"][:,0],"root_ang_vel_w":d["body_ang_vel_w"][:,0],
                "joint_pos":d["joint_pos"],"joint_vel":d["joint_vel"],"motion_ends":d["motion_ends"]}
        for k,v in arrays.items():
            path=out/(k+".npy")
            if path.exists():
                if not np.array_equal(np.load(path),v):raise RuntimeError(f"Refusing to change existing data: {path}")
            else:np.save(path,v,allow_pickle=False)
        names=[f"{split}_clip_{i:03d}" for i in range(len(d["motion_ends"]))]
        (out/"joint_names.json").write_text(json.dumps(d["joint_names"].tolist())+"\n")
        (out/"motion_names.json").write_text(json.dumps(names)+"\n")
        meta={"body_names":d["body_names"].tolist(),"joint_names":d["joint_names"].tolist(),
              "clips":len(names),"frames":len(d["joint_pos"]),"fps":50,"duration_h":len(names)*9/3600,
              "source":str(p),"source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
              "conversion":"Copy root-link and joint arrays exactly; FK is recomputed by official MotionLoader on CUDA."}
        (out/"meta.json").write_text(json.dumps(meta,indent=2)+"\n")
        manifest["splits"][split]={"meta":meta,"files_sha256":{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in out.iterdir() if q.is_file()}}
    (dst/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(dst)


if __name__=="__main__":main()
