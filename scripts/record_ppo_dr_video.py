#!/usr/bin/env python3
"""Render a complete sampled domain-randomized Sesame PPO policy episode.

The saved task is evaluated in the selected nominal or randomized domain.
Camera, lights, and non-colliding floor graphics are presentation changes only.
Policy inference and simulation remain on CUDA. RGB transfer and video encoding
are presentation work, not CPU model execution.
"""
from __future__ import annotations
import argparse
import copy
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/ppo_dr_20261102_seed42")
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--domain", choices=("nominal", "randomized"), default="randomized")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25, choices=(25, 50))
    parser.add_argument("--encoder", default="h264_nvenc", choices=("h264_nvenc", "libx264"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "final.pt"
    saved_environment = run_dir / "sources/sesame_rl/environment.py"
    environment = ROOT / "sesame_rl/environment.py"
    if saved_environment.read_bytes() != environment.read_bytes():
        raise RuntimeError("Environment source changed since training; refusing to alter this evaluation")
    dr_source = ROOT / "sesame_rl/dr_environment.py"
    if (run_dir / "sources/sesame_rl/dr_environment.py").read_bytes() != dr_source.read_bytes():
        raise RuntimeError("Domain randomization source changed since training")
    result = json.loads((run_dir / "result.json").read_text())
    saved_cfg = json.loads((run_dir / "environment_config.json").read_text())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == result["checkpoint_sha256"]
    output = run_dir / "videos" / args.domain
    output.mkdir(parents=True, exist_ok=True)
    video = output / f"ppo_dr_{args.domain}_seed{args.seed}.mp4"
    report_path = output / f"ppo_dr_{args.domain}_seed{args.seed}.json"
    runtime = ROOT / ".runtime/ppo_video"
    for key, sub in {"XDG_CACHE_HOME":"cache", "XDG_CONFIG_HOME":"config", "XDG_DATA_HOME":"data", "CUDA_CACHE_PATH":"cuda_cache"}.items():
        path = runtime / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    report = {
        "status":"running", "started_at_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoint":str(checkpoint), "checkpoint_sha256":checkpoint_hash,
        "environment_sha256":hashlib.sha256(environment.read_bytes()).hexdigest(),
        "dr_environment_sha256":hashlib.sha256(dr_source.read_bytes()).hexdigest(),
        "domain_mode":args.domain,
        "seed":args.seed, "num_envs":1, "fps":args.fps,
        "resolution":[args.width,args.height], "device":"cuda:0",
        "policy":"deterministic mean action with trained observation normalizers",
        "scope":"Full unselected live-policy episode; actual Isaac RTX render, not a replay or generated illustration.",
        "camera":"world-aligned oblique translation-follow camera; does not rotate with body yaw",
        "visual_only_changes":"non-colliding 0.1m floor grid, world +X arrows, camera and HUD",
        "comparison_note":"This is a separate single-environment rollout, not the same held-out episode as batched evaluation. CUDA random draws depend on batch shape, so the same seed may produce different initial states and randomized physical parameters; numerical simulation differences are also possible. Actual parameters and initial states are logged, and task/control definitions are unchanged.",
        "configured_episode_length_s":saved_cfg["episode_length_s"],
        "training_environment_source_unchanged":True,
    }
    write_json(report_path, report)
    app = env = encoder = encoder_log = None
    code = 1
    frames = 0
    started = time.monotonic()
    try:
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

        config = json.loads((run_dir / "ppo_config.json").read_text())
        cfg = make_env_cfg(num_envs=1, seed=args.seed, device="cuda:0", episode_length_s=saved_cfg["episode_length_s"],
                           domain_randomization=args.domain == "randomized")
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
        state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        policy.load_state_dict(state["model_state_dict"])
        policy.eval()
        assert all(param.is_cuda for param in policy.parameters())
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
                    draw.text((28,14),"SESAME  /  PPO + DOMAIN RANDOMIZATION",font=title_font,fill=(235,241,247,255))
                    draw.text((29,54),f"LIVE POLICY  ·  SEED {args.seed}  ·  {args.domain.upper()} PHYSICS  ·  TARGET +X 0.10 m/s",font=small_font,fill=(161,185,202,255))
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
            "encoder":args.encoder,"wall_seconds_before_shutdown":time.monotonic()-started})
        write_json(output / f"ppo_dr_{args.domain}_seed{args.seed}_trajectory.json",{"samples":trajectory,"terminal_record":terminal})
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


if __name__=="__main__":
    raise SystemExit(main())
