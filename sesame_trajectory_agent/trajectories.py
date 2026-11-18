"""CUDA reference generation and official packed-motion export.

Source clips are finite recordings, not assumed periodic. Out-of-range sampling
is rejected instead of wrapping a clip or padding its end.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_mul, quat_conjugate
from .schema import LEGS
from .decisions import decide

FIELDS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_npz(path, tensors, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
    np.savez_compressed(path, **arrays, **metadata)


def derivative(x, dt):
    return torch.gradient(x, spacing=(dt,), dim=(0,), edge_order=1)[0]


def yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def angular_velocity(q, dt):
    # World-frame quaternion increment: q_next * inverse(q_current).
    delta = quat_mul(q[1:], quat_conjugate(q[:-1]))
    delta = torch.where(delta[:, :1] < 0, -delta, delta)
    norm = delta[:, 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, delta[:, :1].clamp_min(1e-8))
    omega = delta[:, 1:] * torch.where(norm > 1e-7, angle / norm.clamp_min(1e-8), 2.) / dt
    return torch.cat((omega, omega[-1:]), dim=0)


def indices(times, frames, fps):
    if float(times.min()) < 0 or float(times.max()) > (frames-1)/fps + 1e-6:
        raise ValueError("Proposal samples outside its source clip; no cyclic wrap or padding is allowed")
    index = (times*fps).floor().long().clamp(0, frames-2)
    alpha = (times*fps - index).clamp(0, 1)
    return index, alpha


def hermite(q, dq, times, fps, columns):
    i, u = indices(times, len(q), fps)
    q0, q1 = q[i, columns], q[i+1, columns]
    m0, m1 = dq[i, columns]/fps, dq[i+1, columns]/fps
    value = (2*u**3-3*u**2+1)*q0 + (u**3-2*u**2+u)*m0 + (-2*u**3+3*u**2)*q1 + (u**3-u**2)*m1
    vel = ((6*u**2-6*u)*q0 + (3*u**2-4*u+1)*m0 + (-6*u**2+6*u)*q1 + (3*u**2-2*u)*m1)*fps
    return value, vel


class ReferenceLibrary:
    def __init__(self, path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU numerical fallback")
        self.path = Path(path)
        if self.path.name in ("validation", "test"):
            raise ValueError("Use training references only; held-out data must not feed the search")
        self.meta = json.loads((self.path/"meta.json").read_text())
        self.joint_names = json.loads((self.path/"joint_names.json").read_text())
        self.body_names = self.meta["body_names"]
        if self.meta["fps"] != 50 or len(self.joint_names) != 8 or len(set(self.joint_names)) != 8:
            raise ValueError("Expected eight named joints at 50 Hz")
        self.ends = np.load(self.path/"motion_ends.npy", allow_pickle=False).tolist()
        self.names = json.loads((self.path/"motion_names.json").read_text())
        if not self.ends or len(self.names) != len(self.ends) or self.ends[0] < 2:
            raise ValueError("Invalid motion boundaries")
        if any(b-a < 2 for a,b in zip(self.ends,self.ends[1:])):
            raise ValueError("Motion boundaries must be increasing")
        self.data = {k: torch.as_tensor(np.load(self.path/(k+".npy"), allow_pickle=False),
                                       device="cuda:0", dtype=torch.float32) for k in FIELDS}
        sizes = {"root_pos_w":3,"root_quat_w":4,"root_lin_vel_w":3,"root_ang_vel_w":3,"joint_pos":8,"joint_vel":8}
        for k, x in self.data.items():
            if x.shape != (self.ends[-1], sizes[k]) or not bool(torch.isfinite(x).all()):
                raise ValueError(f"Invalid source array {k}")
        self.hashes = {p.name: sha256(p) for p in self.path.iterdir() if p.is_file()}

    def summaries(self):
        """CUDA-computed source statistics help the Agent choose actual offsets."""
        summaries=[]
        for i,(begin,end) in enumerate(zip([0]+self.ends[:-1],self.ends)):
            q=self.data["joint_pos"][begin:end]
            def values(x):return [round(v,4) for v in x.tolist()]
            summaries.append({"source_clip":i,"duration_s":(end-begin-1)/50,
                "mean_forward_speed_m_s":round(float(self.data["root_lin_vel_w"][begin:end,0].mean()),4),
                "joint_position_min_rad":values(q.amin(0)),"joint_position_max_rad":values(q.amax(0)),
                "joint_position_mean_rad":values(q.mean(0))})
        return summaries

    def generate(self, p, config):
        begin = 0 if p.source_clip == 0 else self.ends[p.source_clip-1]
        end = self.ends[p.source_clip]
        src = {k: v[begin:end] for k,v in self.data.items()}
        fps = config["fps"]; dt = 1/fps
        # Preserve source positions, including any out-of-limit samples. The
        # validator rejects invalid generated results; generation never clips.
        original_q = src["joint_pos"]
        if p.smoothing_window_s > (len(original_q)-1)/fps:
            raise ValueError("Smoothing window exceeds source duration")
        width = int(round(p.smoothing_window_s*fps)) | 1
        smooth = original_q
        if width > 1:
            signal = torch.nn.functional.pad(original_q.T[None],(width//2,width//2),mode="replicate")
            smooth = torch.nn.functional.avg_pool1d(signal,width,stride=1)[0].T
        # Monotone cubic Hermite tangents: no spline overshoot between samples.
        secant = (smooth[1:]-smooth[:-1])*fps
        slopes = torch.zeros_like(smooth)
        slopes[0],slopes[-1] = secant[0],secant[-1]
        a,b = secant[:-1],secant[1:]
        same_sign = a*b > 0
        denom = torch.where(same_sign,a+b,torch.ones_like(a))
        slopes[1:-1] = torch.where(same_sign,2*a*b/denom,0.)
        time = torch.arange(round(config["duration_s"]*fps)+1, device="cuda:0")*dt
        base_time = p.start_time_s + p.tempo*time
        shifts = torch.zeros(8, device="cuda:0")
        for leg, names in LEGS.items():
            for name in names:
                shifts[self.joint_names.index(name)] = p.phase_offsets_s.get(leg, 0.)
        sample_times = base_time[:, None] + shifts[None]
        cols = torch.arange(8, device="cuda:0")[None]
        q, dq = hermite(smooth, slopes, sample_times, fps, cols)
        center = smooth.mean(0)
        offsets = q.new_tensor([p.joint_offsets_rad.get(n, 0.) for n in self.joint_names])
        q = center + p.amplitude*(q-center) + offsets
        dq = p.amplitude*p.tempo*dq
        i,u = indices(base_time, len(src["joint_pos"]), fps)
        pos = torch.lerp(src["root_pos_w"][i], src["root_pos_w"][i+1], u[:,None])
        q0,q1 = src["root_quat_w"][i], src["root_quat_w"][i+1]
        q1 = torch.where((q0*q1).sum(-1,keepdim=True)<0, -q1, q1)
        quat = torch.nn.functional.normalize(torch.lerp(q0,q1,u[:,None]), dim=-1)
        # Fixed world +x task. The agent cannot lower the commanded speed to pass.
        pos[:,0] = config["target_speed_m_s"]*time
        pos[:,1] = 0.
        pos[:,2] += p.height_offset_m
        half_yaw = -yaw(quat[0])/2
        rotation = quat.new_zeros(4); rotation[0] = half_yaw.cos(); rotation[3] = half_yaw.sin()
        quat = quat_mul(rotation.expand_as(quat), quat)
        ref = {"root_pos_w":pos, "root_quat_w":quat, "root_lin_vel_w":derivative(pos,dt),
               "root_ang_vel_w":angular_velocity(quat,dt), "joint_pos":q, "joint_vel":dq}
        details = {"source_clip":self.names[p.source_clip],"smoothing_samples":width,
                   "seed_position_projection_max_rad":0.0,
                   "seed_position_projection_fraction":0.0,"generation_clipping":False,
                   "joint_velocity_source":"analytic derivative of generated monotone cubic; not copied from source",
                   "root_goal":"fixed straight-line speed; root height/orientation interpolated from seed",
                   "original_source_modified":False}
        return ref,details


def export_accepted(destination, accepted, library, protocol_path):
    """Only nominal *executed* traces that passed every required trial are exported."""
    destination = Path(destination)
    protocol = json.loads(Path(protocol_path).read_text())
    protocol_hash = sha256(protocol_path)
    config = protocol["config"]
    expected_steps = round(config["duration_s"]*config["fps"])
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    arrays = {k: [] for k in FIELDS}
    ends = []; frames = 0
    for item in accepted:
        report_path = Path(item["report"])
        report = json.loads(report_path.read_text())
        if report.get("accepted") is not True or report.get("validation_complete") is not True:
            raise ValueError("Refusing to export an unvalidated candidate")
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("Cannot mix different acceptance protocols")
        if report.get("stage") != "accepted" or report.get("reasons") != [] or report.get("kinematics",{}).get("passed") is not True:
            raise ValueError("Candidate has not passed all kinematic and execution checks")
        if report.get("generator") == "mimicagent_sesame":
            checks = report["kinematics"].get("checks", {})
            required = {"base_height_envelope", "non_aerial_support_half_cycle", "ik_reachability",
                        "declared_contacts_match_ground", "fixed_forward_goal"}
            if set(checks) != required or not all(v is True for v in checks.values()):
                raise ValueError("Missing or failed MimicAgent structural checks")
            for filename,key in [("motion.py","motion_program_sha256"),
                                 ("candidate_reference.npz","candidate_reference_sha256")]:
                if sha256(report_path.parent/filename) != report.get(key):
                    raise ValueError("Motion program/reference changed after validation")
            if sha256(report_path.parent.parent/"plan.json") != report.get("plan_sha256"):
                raise ValueError("Motion plan changed after validation")
        trials = report.get("randomized",[])
        if len(trials) != config["robust_trials"]:
            raise ValueError("Missing required randomized trials")
        nominal = report.get("nominal",{})
        if nominal.get("trial") != 0 or nominal.get("randomized") is not False:
            raise ValueError("Missing nominal trial")
        for i, trial in enumerate(trials):
            if trial.get("trial") != i or trial.get("randomized") is not True:
                raise ValueError("Invalid randomized trial identity")
        for trial in [nominal,*trials]:
            metrics = trial.get("metrics",{})
            if trial.get("passed") is not True or trial.get("reasons") != [] or decide(metrics,config["gates"],config.get("diagnostic_only_metrics",[])) or metrics.get("steps") != expected_steps:
                raise ValueError("Refusing to export a failed or incomplete trial")
        if trials and sha256(report_path.parent/"randomized_execution.npz") != report.get("randomized_trace_sha256"):
            raise ValueError("Randomized trace changed after validation")
        trace_path = Path(item["trace"])
        if sha256(trace_path) != report["nominal_trace_sha256"]:
            raise ValueError("Executed trace changed after validation")
        with np.load(trace_path, allow_pickle=False) as trace:
            n = len(trace["joint_pos"])
            sizes = {"root_pos_w":3,"root_quat_w":4,"root_lin_vel_w":3,"root_ang_vel_w":3,"joint_pos":8,"joint_vel":8}
            if n != expected_steps+1 or any(trace[k].shape != (n,1,sizes[k]) for k in FIELDS):
                raise ValueError("Executed trace shape differs from the validated rollout")
            for k in FIELDS: arrays[k].append(trace[k][:,0])
        frames += n; ends.append(frames)
        entries.append({"id":item["id"],"source_clip":item["source_clip"],
                        "trace":str(trace_path),"trace_sha256":sha256(trace_path),
                        "validation":str(report_path),"validation_sha256":sha256(report_path)})
    if accepted:
        out = destination/"train"; out.mkdir(exist_ok=False)
        for k, chunks in arrays.items(): np.save(out/(k+".npy"),np.concatenate(chunks),allow_pickle=False)
        np.save(out/"motion_ends.npy",np.asarray(ends,dtype=np.int64),allow_pickle=False)
        write_json(out/"joint_names.json",library.joint_names)
        write_json(out/"motion_names.json",[x["id"] for x in entries])
        write_json(out/"meta.json",{"body_names":library.body_names,"joint_names":library.joint_names,
                   "fps":50,"clips":len(accepted),"frames":frames,"duration_h":(frames-len(accepted))/50/3600,
                   "source":"actual nominal MuJoCo execution, after fixed nominal and perturbation gates",
                   "protocol_sha256":protocol_hash,"split":"train"})
    write_json(destination/"manifest.json",{"accepted_count":len(entries),"entries":entries,
               "protocol_sha256":protocol_hash,"train_directory":str(destination/"train") if entries else None,
               "note":"Only accepted, fully validated executed trajectories are admitted as training data. Unvalidated or failed candidates are diagnostics only. Existing validation/test references are untouched. No train directory is created when zero candidates pass."})
