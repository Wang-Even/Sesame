"""Convert existing references on CUDA; retain source lineage and split boundaries."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/supermimic_official_5efbf114/src")]

import json
import hashlib
import numpy as np
import torch
from mjlab.entity import Entity
from sesame_official.robot import robot_cfg
from sesame_official.kinematics import SesameFK, SOURCE_JOINTS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for kinematic conversion")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    # XML compilation / name discovery only; no CPU simulation or model forward.
    robot = Entity(robot_cfg())
    model = robot.spec.compile()
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    print("Body order:", bodies, "Joint order:", joints, flush=True)
    reorder = [SOURCE_JOINTS.index(n) for n in joints]
    fk = SesameFK(joints, bodies, "cuda:0")
    source = ROOT / "datasets/supermimic_sesame_20261110"
    dest = ROOT / "datasets/supermimic_official_sesame_20261111"
    dest.mkdir(exist_ok=True)
    index = json.loads((source / "reference_index.json").read_text())
    manifest = {"source":str(source), "device":"cuda:0", "splits":{},
                "source_joint_order":SOURCE_JOINTS, "joint_names":joints, "body_names":bodies,
                "root_velocity_source":"body_lin_vel[:, base] (measured link-origin velocity)",
                "note":"References generated in Isaac; retargeted geometry only, not dynamics-equivalent trajectories."}
    for split in ("train", "validation", "test"):
        clips = [r for r in index if r["split"] == split]
        values, ends, frames = {}, [], 0
        for row in clips:
            path = source / row["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            with np.load(path, allow_pickle=False) as data:
                d = {k:torch.tensor(data[k], device="cuda:0", dtype=torch.float32)
                     for k in ("root_pos","root_quat","body_lin_vel","root_ang_vel","joint_pos","joint_vel","body_pos","foot_pos")}
            pos, quat, ang = d["root_pos"], d["root_quat"], d["root_ang_vel"]
            vel = d["body_lin_vel"][:,0]
            q, dq = d["joint_pos"][:,reorder], d["joint_vel"][:,reorder]
            bp, bq, bv, bw = fk(pos, quat, q, vel, ang, dq)
            source_bodies = json.loads((source / "schema.json").read_text())["body_order"]
            body_err = (bp[:,[bodies.index(n) for n in source_bodies]] - d["body_pos"]).abs().max().item()
            foot_err = (bp[:,[bodies.index(n) for n in ("r3_foot","r4_foot","l3_foot","l4_foot")]] - d["foot_pos"]).abs().max().item()
            if max(body_err, foot_err) > 1e-4:
                raise ValueError(f"Reference FK mismatch: body={body_err}, feet={foot_err}")
            arrays = dict(joint_pos=q, joint_vel=dq, body_pos_w=bp, body_quat_w=bq,
                          body_lin_vel_w=bv, body_ang_vel_w=bw)
            for k, value in arrays.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {k}")
                values.setdefault(k,[]).append(value.detach().cpu().numpy())
            frames += len(pos); ends.append(frames)
        output = dest / f"{split}.npz"
        np.savez_compressed(output, **{k:np.concatenate(v) for k,v in values.items()},
                            motion_ends=np.array(ends), body_names=np.array(bodies),
                            joint_names=np.array(joints), fps=np.array(50))
        manifest["splits"][split] = {"clips":clips, "frames":frames,
            "duration_h":sum(r["duration_s"] for r in clips)/3600,
            "path":str(output), "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Converted:", dest, flush=True)


if __name__ == "__main__":
    main()
