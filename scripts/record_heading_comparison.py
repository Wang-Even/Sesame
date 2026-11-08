#!/usr/bin/env python3
"""Record and assemble the old versus heading-reward PPO comparison.

Both policies use the same FULL randomized test dynamics.
Selection is fixed at train seed 42 and evaluation seed 15042 before rendering.
Camera, lights, and non-colliding floor graphics are presentation changes only.
Policy inference and simulation remain on CUDA. RGB transfer and video encoding
are presentation work, not CPU model execution.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_ppo import disable_git_imports, write_json
from evaluate_ppo_study import canonical_hash, resolve_checkpoints
from train_ppo_study import require_runtime, sha256


def resolve_video_policy(protocol_path: Path, policy_id: str):
    """Read-only final-model, source and PPO configuration verification."""
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1 or protocol.get("study_id") != "ppo_heading_20261105":
        raise ValueError("Expected the frozen heading PPO experiment protocol")
    if sha256(protocol_path) != protocol_path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Training protocol hash mismatch")
    for relative, expected in protocol["frozen_sources"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Training-frozen source changed: {relative}")
    if policy_id not in ("full_dr_seed42", "heading_seed42"):
        raise ValueError("Video selection is fixed at train seed 42 for both arms")
    collection = protocol["control_runs"] if policy_id == "full_dr_seed42" else protocol["training_runs"]
    selected = [entry for entry in collection if entry["id"] == policy_id]
    if len(selected) != 1 or selected[0]["seed"] != 42:
        raise ValueError("Expected one declared seed-42 final model")
    entry = dict(selected[0], train_seed=42,
                 variant="full_dr" if policy_id == "full_dr_seed42" else "heading_v1")
    if "checkpoint_sha256" in entry:
        entry["sha256"] = entry["checkpoint_sha256"]
    result = json.loads((Path(entry["run_dir"])/"result.json").read_text())
    if policy_id == "heading_seed42" and (result.get("algorithm") != "PPO" or result.get("heading_penalty_scale") != 0.5):
        raise RuntimeError("Heading arm must be the completed ordinary PPO model with coefficient 0.5")
    selected_protocol = dict(protocol, checkpoints=[entry])
    entry = resolve_checkpoints(selected_protocol, smoke=False)[0]
    # Rendering uses exactly the same physical environment for both policies.
    # This in-memory adapter does not mutate the frozen training protocol.
    return dict(protocol, evaluation={"episode_length_s":10.0}), entry


def render_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-id", choices=("full_dr_seed42", "heading_seed42"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25, choices=(25, 50))
    parser.add_argument("--encoder", default="h264_nvenc", choices=("h264_nvenc", "libx264"))
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    protocol, entry = resolve_video_policy(args.protocol, args.policy_id)
    args.seed = 15042
    run_dir = Path(entry["run_dir"])
    checkpoint = Path(entry["checkpoint"])
    checkpoint_hash = entry["sha256"]
    config = entry["ppo_config"]
    episode_length_s = protocol["evaluation"]["episode_length_s"]
    environment = ROOT / "sesame_rl/environment.py"
    dr_source = ROOT / "sesame_rl/dr_environment.py"
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output-dir must remain inside the Sesame workspace")
    output.mkdir(parents=True, exist_ok=True)
    video = output / f"ppo_heading_{args.policy_id}_seed{args.seed}.mp4"
    report_path = output / f"ppo_heading_{args.policy_id}_seed{args.seed}.json"
    if video.exists() or report_path.exists():
        raise FileExistsError("Preserve existing video/report before another recording attempt")
    runtime = ROOT / ".runtime/heading_video"
    for key, sub in {"XDG_CACHE_HOME":"cache", "XDG_CONFIG_HOME":"config", "XDG_DATA_HOME":"data", "CUDA_CACHE_PATH":"cuda_cache"}.items():
        path = runtime / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    report = {
        "status":"running", "started_at_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoint":str(checkpoint), "checkpoint_sha256":checkpoint_hash,
        "policy_id":args.policy_id, "training_variant":entry["variant"], "train_seed":entry["train_seed"],
        "protocol":str(args.protocol), "protocol_sha256":sha256(args.protocol),
        "recording_script_sha256":sha256(__file__),
        "ppo_config_sha256":entry["ppo_config_sha256"], "source_checks":entry["source_checks"],
        "environment_sha256":hashlib.sha256(environment.read_bytes()).hexdigest(),
        "dr_environment_sha256":hashlib.sha256(dr_source.read_bytes()).hexdigest(),
        "domain_mode":"randomized", "evaluation_environment_variant":"full_dr_for_all_policies",
        "seed":args.seed, "num_envs":1, "fps":args.fps,
        "resolution":[args.width,args.height], "device":"cuda:0",
        "policy":"deterministic mean action with trained observation normalizers",
        "scope":"First complete episode with preselected train seed 42 and evaluation seed 15042; actual live Isaac RTX render. Single-environment illustration, separate from the statistical batched evaluation.",
        "camera":"world-aligned oblique translation-follow camera; does not rotate with body yaw",
        "visual_only_changes":"non-colliding 0.1m floor grid, world +X arrows, camera and HUD",
        "comparison_note":"Both comparison videos share the same single-environment seed 15042 initial conditions and full randomized physical parameters, verified by exact snapshot hashes. This is distinct from batched evaluation: CUDA draw shapes can change initial states and physical samples even with the same seed. Actual test parameters and initial states are logged.",
        "configured_episode_length_s":episode_length_s,
        "training_environment_source_unchanged":True,
    }
    write_json(report_path, report)
    app = env = encoder = encoder_log = None
    code = 1
    frames = 0
    started = time.monotonic()
    try:
        require_runtime()
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU model/simulation fallback prohibited")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        app = AppLauncher(
            headless=True, enable_cameras=True, device="cuda:0",
            kit_args=f"--portable-root {runtime / 'kit'} --/log/file={output / 'recording_isaac.log'} --/app/settings/persistent=false --/renderer/multiGpu/enabled=false",
        ).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / "warp_cache")
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from sesame_rl.dr_environment import SesameDREnv as SesameEnv, make_env_cfg
        from pxr import Gf, UsdGeom
        from PIL import Image, ImageDraw, ImageFont

        # Deliberately use the frozen full-DR class for EVERY training arm.
        # The shaping term does not change evaluation physics or observations.
        cfg = make_env_cfg(num_envs=1, seed=args.seed, device="cuda:0", episode_length_s=episode_length_s,
                           domain_randomization=True)
        cfg.viewer.resolution = (args.width, args.height)
        cfg.viewer.eye = (-0.17, -0.32, 0.20)
        cfg.viewer.lookat = (0.02, 0.0, 0.035)
        cfg.viewer.origin_type = "world"
        # Render every 2 controls for 25fps; the physical dt/decimation stay frozen.
        cfg.sim.render_interval = cfg.decimation * (50 // args.fps)
        env = SesameEnv(cfg, render_mode="rgb_array")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)

        # Visual grid and arrows carry no collision or rigid-body APIs.
        for index in range(-6, 24):
            x = index * 0.1
            major = index % 5 == 0
            line = sim_utils.CuboidCfg(
                size=(0.0010 if major else 0.00045, 1.6, 0.00015),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30,0.45,0.49) if major else (0.32,0.35,0.38)),
            )
            line.func(f"/World/VideoGrid/X_{index + 6}", line, translation=(x,0.0,0.00015))
        for index in range(-8, 9):
            line = sim_utils.CuboidCfg(
                size=(3.0,0.00045,0.00015),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32,0.35,0.38)),
            )
            line.func(f"/World/VideoGrid/Y_{index + 8}", line, translation=(0.85,index * 0.1,0.00015))
        for index in range(5):
            x = index * 0.5
            shaft = sim_utils.CuboidCfg(
                size=(0.075,0.003,0.0002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18,0.72,0.78)),
            )
            shaft.func(f"/World/WorldXArrows/shaft{index}",shaft,translation=(x+0.04,-0.15,0.0003))
            arrow = UsdGeom.Mesh.Define(get_current_stage(),f"/World/WorldXArrows/head{index}")
            arrow.CreatePointsAttr([Gf.Vec3f(x+0.10,-0.15,0.0003),Gf.Vec3f(x+0.07,-0.16,0.0003),Gf.Vec3f(x+0.07,-0.14,0.0003)])
            arrow.CreateFaceVertexCountsAttr([3])
            arrow.CreateFaceVertexIndicesAttr([0,1,2])
            arrow.CreateDisplayColorAttr([(0.18,0.72,0.78)])
            arrow.CreateDoubleSidedAttr(True)

        camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(cfg.viewer.cam_prim_path))
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.005,100.0))
        camera.GetFocalLengthAttr().Set(24.0)
        policy_cfg = copy.deepcopy(config["policy"])
        policy_cfg.pop("class_name")
        with torch.device("cuda:0"):
            policy = ActorCritic(wrapper.get_observations(), config["obs_groups"], wrapper.num_actions, **policy_cfg).to("cuda:0")
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed after provenance validation")
        state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        policy.load_state_dict(state["model_state_dict"])
        policy.eval()
        assert all(param.is_cuda for param in policy.parameters())
        assert all(buffer.is_cuda for buffer in policy.buffers())
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 23)
        title_font = ImageFont.truetype(font_path, 30)
        small_font = ImageFont.truetype(font_path, 18)
        encoder_log = (output / "ffmpeg.log").open("wb")
        command = ["ffmpeg","-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pixel_format","rgb24","-video_size",f"{args.width}x{args.height}","-framerate",str(args.fps),"-i","-","-an","-c:v",args.encoder]
        command += (["-preset","slow","-rc","vbr","-cq","19","-b:v","5M"] if args.encoder=="h264_nvenc" else ["-preset","medium","-crf","19"])
        command += ["-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
        trajectory = []
        terminal = None
        stride = 50 // args.fps
        with torch.inference_mode():
            env.reset(seed=args.seed)
            env.episode_records.clear()
            report["domain_parameters"] = env.domain_snapshot()
            report["domain_readback"] = env.verify_domain_parameters()
            obs = wrapper.get_observations()
            initial_position = env.robot.data.root_pos_w[0].clone()
            report["initial_root_state_w"] = env.robot.data.root_state_w[0].tolist()
            report["initial_joint_pos_rad"] = env.robot.data.joint_pos[0].tolist()
            report["initial_joint_vel_rad_s"] = env.robot.data.joint_vel[0].tolist()
            initial = {
                "evaluation_seed":args.seed, "num_envs":1, "condition":"randomized",
                "domain_snapshot":report["domain_parameters"],
                "root_state_w":report["initial_root_state_w"],
                "joint_pos_rad":report["initial_joint_pos_rad"],
                "joint_vel_rad_s":report["initial_joint_vel_rad_s"],
                "applied_targets_firmware_rad":env._targets[0].tolist(),
            }
            report["initial_conditions"] = initial
            report["initial_condition_sha256"] = canonical_hash(initial)
            paired_with = []
            for other_report in output.parent.glob("*/ppo_heading_*_seed15042.json"):
                if other_report == report_path:
                    continue
                other = json.loads(other_report.read_text())
                if other.get("status") != "completed":
                    continue
                if other.get("protocol_sha256") != report["protocol_sha256"]:
                    raise RuntimeError(f"Existing comparison video uses a different protocol: {other_report}")
                if other.get("initial_conditions") != initial:
                    raise RuntimeError(f"Video initial state/domain pairing differs: {other_report}")
                if other.get("initial_condition_sha256") != report["initial_condition_sha256"]:
                    raise RuntimeError(f"Video pairing hash differs: {other_report}")
                paired_with.append(other["policy_id"])
            report["exactly_paired_with_prior_videos"] = paired_with
            # Create RGB annotator and warm RTX without advancing physical time.
            env.render()
            for _ in range(12):
                env.sim.render()
            for step in range(env.max_episode_length):
                root = env.robot.data.root_state_w[0]
                x,y,z = root[:3].tolist()
                delta_x = float((root[0]-initial_position[0]).item())
                q_w,q_x,q_y,q_z = root[3:7].tolist()
                yaw_deg = math.degrees(math.atan2(2*(q_w*q_z+q_x*q_y),1-2*(q_y*q_y+q_z*q_z)))
                if step % stride == 0:
                    # Fixed world orientation deliberately preserves visible turns.
                    env.sim.set_camera_view(eye=(x-0.17,y-0.32,0.20),target=(x+0.02,y,0.035))
                    env.sim.render()
                    rgb = env.render(recompute=True)
                    if rgb is None or not rgb.size:
                        raise RuntimeError("RTX render returned no RGB data")
                    frame = Image.fromarray(rgb).convert("RGB")
                    draw = ImageDraw.Draw(frame, "RGBA")
                    draw.rectangle((0,0,args.width,88),fill=(10,17,25,225))
                    draw.text((28,14),f"SESAME  /  {'OLD PPO + DR' if args.policy_id == 'full_dr_seed42' else 'PPO + DR + HEADING'}  /  TRAIN SEED 42",font=title_font,fill=(235,241,247,255))
                    draw.text((29,54),f"LIVE POLICY  ·  TEST SEED {args.seed}  ·  FULL RANDOMIZED PHYSICS  ·  TARGET +X 0.10 m/s",font=small_font,fill=(161,185,202,255))
                    draw.rectangle((0,args.height-68,args.width,args.height),fill=(10,17,25,225))
                    draw.text((28,args.height-53),f"t  {step*env.step_dt:5.2f} s      dx  {delta_x:5.3f} m      heading  {yaw_deg:+6.1f} deg",font=font,fill=(230,241,247,255))
                    draw.text((args.width-282,args.height-50),"GRID 0.10 m   /   +X →",font=small_font,fill=(91,205,215,255))
                    if frames in (0,args.fps*5):
                        frame.save(output / f"frame_{frames:04d}.png")
                    if frames in (0,args.fps*5):
                        frame.save(output / "thumbnail.png")
                        report["thumbnail_frame_index"] = frames
                    encoder.stdin.write(frame.tobytes())
                    frames += 1
                    if frames % 50 == 0:
                        print(f"VIDEO_PROGRESS frames={frames} simulated_s={step*env.step_dt:.2f} dx_m={delta_x:.3f} yaw_deg={yaw_deg:.1f}",flush=True)
                actions = policy.act_inference(obs)
                if not actions.is_cuda or not bool(torch.isfinite(actions).all()):
                    raise RuntimeError("Policy actions must be finite CUDA tensors")
                trajectory.append({"step":step,"time_s":step*env.step_dt,"root_state_w":root.tolist(),"raw_action":actions[0].tolist()})
                obs,reward,done,extras = wrapper.step(actions)
                if not bool(torch.isfinite(obs["policy"]).all()):
                    raise RuntimeError("Nonfinite rendered rollout observation")
                if bool(done[0]):
                    terminal = dict(env.episode_records[-1])
                    break
        if terminal is None:
            raise RuntimeError("No terminal record for the complete episode")
        frame.save(output / "last_frame.png")
        if frames <= args.fps*5:
            # Keep a valid thumbnail even if this unselected episode falls early.
            frame.save(output / "thumbnail.png")
            report["thumbnail_frame_index"] = frames - 1
        encoder.stdin.close()
        encoder.wait(timeout=60)
        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {encoder.returncode}; see ffmpeg.log")
        encoder = None
        report.update({"status":"completed","video":str(video),"thumbnail":str(output/"thumbnail.png"),
            "frames":frames,"video_duration_s":frames/args.fps,"episode":terminal,
            "encoder":args.encoder,"video_sha256":sha256(video),"wall_seconds_before_shutdown":time.monotonic()-started})
        write_json(output / f"ppo_heading_{args.policy_id}_seed{args.seed}_trajectory.json",{"samples":trajectory,"terminal_record":terminal})
        code=0
    except Exception as exc:
        report.update({"status":"failed","error":repr(exc),"traceback":traceback.format_exc(),"frames":frames})
        traceback.print_exc()
    finally:
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.terminate()
        if encoder_log is not None:
            encoder_log.close()
        report["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
        write_json(report_path,report)
        print("SESAME_VIDEO_RESULT="+json.dumps(report),flush=True)
        if app is not None:
            def timeout_exit():
                report["shutdown"]={"completed":False,"forced_process_exit":True,"timeout_seconds":15}
                write_json(report_path,report)
                os._exit(code)
            timer=threading.Timer(15,timeout_exit)
            timer.daemon=True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
    return code


POLICIES = ("full_dr_seed42", "heading_seed42")
TITLES = ("OLD PPO + DR  |  SEED 42", "PPO + DR + HEADING  |  SEED 42")
PYTHON = Path("/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python")
CONDA = Path("/home/ubuntu/miniforge3/bin/conda")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assert_gpu_slot_idle():
    """Do not interfere with unrelated GPU jobs or an active study process."""
    relevant = {"train_ppo.py", "train_ppo_dr.py", "train_ppo_study.py",
                "train_heading_ppo.py", "evaluate_ppo_study.py", "evaluate_heading_ppo.py"}
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, ProcessLookupError):
            continue
        if any(Path(arg.decode(errors="replace")).name in relevant for arg in argv if arg):
            active.append({"pid": int(proc.name), "command": b" ".join(argv).decode(errors="replace")})
    if active:
        raise RuntimeError(f"Study GPU slot is still occupied; do not launch recording: {active}")


def ffprobe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of", "json", str(path),
    ], text=True))


def validate_video(output, policy_id, protocol_path, entry):
    report_path = output / policy_id / f"ppo_heading_{policy_id}_seed15042.json"
    report = json.loads(report_path.read_text())
    video = output / policy_id / f"ppo_heading_{policy_id}_seed15042.mp4"
    expected = {
        "status": "completed", "policy_id": policy_id, "seed": 15042, "num_envs": 1,
        "protocol_sha256": sha256(protocol_path), "checkpoint_sha256": entry["sha256"],
        "evaluation_environment_variant": "full_dr_for_all_policies", "domain_mode": "randomized",
        "recording_script_sha256": sha256(__file__),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"{policy_id}: mismatching recording metadata {key}")
    if not report["domain_readback"]["passed"]:
        raise RuntimeError(f"{policy_id}: physical parameter readback failed")
    if canonical_hash(report["initial_conditions"]) != report["initial_condition_sha256"]:
        raise RuntimeError(f"{policy_id}: initial condition snapshot hash is invalid")
    if sha256(video) != report["video_sha256"]:
        raise RuntimeError(f"{policy_id}: original video hash changed")
    probe = ffprobe(video)
    stream = probe["streams"][0]
    if (stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"]) != ("h264", 1280, 720, "25/1"):
        raise RuntimeError(f"{policy_id}: unexpected original video encoding")
    frames = int(stream["nb_frames"])
    duration = float(probe["format"]["duration"])
    episode = report["episode"]
    if not 0.0 < episode["duration_s"] <= 10.00001:
        raise RuntimeError(f"{policy_id}: invalid recorded episode duration")
    if frames != report["frames"] or abs(duration - frames / 25) > 1e-4:
        raise RuntimeError(f"{policy_id}: frame count and report differ")
    # Sampling the first frame of each two-control interval may round a final
    # odd control step upward by 0.02 s; never silently drop an early failure.
    if abs(duration - episode["duration_s"]) > 0.02001:
        raise RuntimeError(f"{policy_id}: video does not cover its complete first episode")
    if not episode.get("timeout") and not any(episode.get(k) for k in ("fell", "out_of_bounds", "invalid_state")):
        raise RuntimeError(f"{policy_id}: episode has no declared termination reason")
    summary = {
        "policy_id": policy_id, "training_variant": entry["variant"],
        "video": str(video), "video_sha256": sha256(video), "report": str(report_path),
        "report_sha256": sha256(report_path), "ffprobe": probe,
        "checkpoint_sha256": entry["sha256"], "initial_condition_sha256": report["initial_condition_sha256"],
        "episode": episode, "domain_parameters": report["domain_parameters"],
        "domain_readback_passed": True,
        "terminal_heading_degrees": math.degrees(episode["final_heading_yaw_rad"]),
        "maximum_absolute_heading_degrees": math.degrees(episode["max_abs_heading_yaw_rad"]),
        "comparison_freeze_tail_s": max(0.0, 10.0 - duration),
    }
    return summary, report["initial_conditions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: protocol directory/comparison_videos")
    parser.add_argument("--assemble-only", action="store_true", help="Validate existing completed originals; skip rendering")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output = (args.output_dir or protocol_path.parent / "comparison_videos").resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output-dir must stay inside the Sesame workspace")
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "comparison_validation.json"
    comparison = output / "ppo_heading_comparison_seed15042.mp4"
    thumbnail = output / "comparison_thumbnail.png"
    output_lock = (output / "recording.lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if comparison.exists():
        raise FileExistsError("Preserve the existing completed comparison before another recording")
    report = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "orchestrator_sha256": sha256(__file__), "policies": list(POLICIES),
        "evaluation_seed": 15042, "condition": "randomized", "num_envs_per_video": 1, "selection":"first episode, train seed 42, eval seed 15042 fixed before rendering; no performance selection",
        "comparison_note": "Exact paired single-environment initial states and full randomized test dynamics for both policies. This is separate from batched study evaluation; same seed with different CUDA draw shapes need not produce the same physical samples.",
        "presentation": "Two actual live-policy recordings side by side, scaled to 640x360 with a 40px title strip per column; existing world-aligned camera, grid and HUD preserved. No synthetic robot imagery.",
        "early_termination_handling": "If an episode ends before 10s, retain its last actual video frame with an explicit EPISODE ENDED label for the remaining comparison duration; original video remains unpadded.",
        "originals": [], "visual_inspection": "pending human/agent image inspection",
    }
    try:
        # Validate every model BEFORE creating the first GPU application.
        entries = {key: resolve_video_policy(protocol_path, key)[1] for key in POLICIES}
        env_vars = dict(os.environ, CONDA_NO_PLUGINS="true", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        initial_reference = None
        for policy_id in POLICIES:
            destination = output / policy_id
            destination.mkdir(parents=True, exist_ok=True)
            existing_report = destination / f"ppo_heading_{policy_id}_seed15042.json"
            if not existing_report.exists():
                if args.assemble_only:
                    raise FileNotFoundError(existing_report)
                assert_gpu_slot_idle()
                command = [str(CONDA), "run", "-n", "env_isaaclab", "--no-capture-output", str(PYTHON),
                           "-u", str(ROOT / "scripts/record_heading_comparison.py"), "--render", "--protocol", str(protocol_path),
                           "--policy-id", policy_id, "--output-dir", str(destination)]
                print("RECORD_START " + policy_id, flush=True)
                with (destination / "recording.log").open("xb") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env_vars, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{policy_id} renderer exited {result.returncode}; preserve its recording.log")
            # Existing completed originals may be resumed only if every hash,
            # setting, episode boundary, and actual domain pairing still match.
            summary, initial = validate_video(output, policy_id, protocol_path, entries[policy_id])
            if initial_reference is None:
                initial_reference = initial
            elif initial != initial_reference:
                raise RuntimeError(f"{policy_id}: initial conditions differ between comparison videos")
            report["originals"].append(summary)
            write_json(validation_path, report)
            print("RECORD_VERIFIED " + policy_id + " " + json.dumps({
                "distance_x_m": summary["episode"]["distance_x_m"], "fell": summary["episode"]["fell"],
                "duration_s": summary["episode"]["duration_s"],
            }), flush=True)
        if len({item["initial_condition_sha256"] for item in report["originals"]}) != 1:
            raise RuntimeError("Initial condition hashes differ despite comparison checks")
        report["exact_pairing_passed"] = True
        report["initial_condition_sha256"] = report["originals"][0]["initial_condition_sha256"]
        report["initial_conditions"] = initial_reference
        assert_gpu_slot_idle()
        if comparison.exists():
            raise FileExistsError("Preserve the existing comparison MP4 before assembling again")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]
        for item in report["originals"]:
            command += ["-i", item["video"]]
        filters = []
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        for index, (title, item) in enumerate(zip(TITLES, report["originals"], strict=True)):
            duration = float(item["ffprobe"]["format"]["duration"])
            chain = (f"[{index}:v]scale=640:360:flags=lanczos,setsar=1,fps=25,"
                     "tpad=stop_mode=clone:stop_duration=10,trim=duration=10,setpts=PTS-STARTPTS,"
                     f"pad=640:400:0:40:color=0x101923,drawtext=fontfile={font}:text='{title}':"
                     "fontcolor=white:fontsize=23:x=15:y=7")
            if duration < 9.999:
                reason = "FALL" if item["episode"]["fell"] else "TERMINATED"
                chain += (f",drawtext=fontfile={font}:text='EPISODE ENDED - {reason}':fontcolor=white:"
                          f"fontsize=22:box=1:boxcolor=0x8b2222:boxborderw=8:x=15:y=90:enable='gte(t,{duration:.6f})'")
            filters.append(chain + f"[c{index}]")
        filters.append("[c0][c1]hstack=inputs=2[vout]")
        command += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-an", "-c:v", "h264_nvenc",
                    "-preset", "slow", "-rc", "vbr", "-cq", "19", "-b:v", "6M", "-pix_fmt", "yuv420p",
                    "-r", "25", "-frames:v", "250", "-movflags", "+faststart", str(comparison)]
        report["assembly_command"] = command
        print("ASSEMBLE_START", flush=True)
        with (output / "comparison_ffmpeg.log").open("xb") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        probe = ffprobe(comparison)
        stream = probe["streams"][0]
        if (stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"], stream["nb_frames"]) != ("h264", 1280, 400, "25/1", "250"):
            raise RuntimeError("Unexpected comparison video encoding/frame count")
        if abs(float(probe["format"]["duration"]) - 10.0) > 1e-4:
            raise RuntimeError("Comparison video must last 10 seconds")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", "5", "-i", str(comparison),
                        "-frames:v", "1", "-update", "1", str(thumbnail)], check=True)
        report.update(status="completed", comparison_video=str(comparison), comparison_sha256=sha256(comparison),
                      comparison_ffprobe=probe, thumbnail=str(thumbnail), thumbnail_sha256=sha256(thumbnail))
        print("COMPARISON_COMPLETED " + str(comparison), flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(validation_path, report)


if __name__ == "__main__":
    if "--render" in sys.argv:
        sys.argv.remove("--render")
        raise SystemExit(render_main())
    raise SystemExit(main())
