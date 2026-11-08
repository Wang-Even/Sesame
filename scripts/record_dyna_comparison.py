#!/usr/bin/env python3
"""Sequentially record the three preregistered study policies and assemble video.

This orchestrator imports no numerical/model/simulator packages. Each renderer
runs in a separate correctly activated Isaac Conda process, then exits before
the next one starts. Invoke only after the coordinator hands over the GPU.
The ffmpeg operations are video presentation, not CPU model execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from record_dyna_ppo_video import canonical_hash, resolve_video_policy

POLICIES = ("baseline_seed42", "full_dr_seed42", "dyna_ppo_seed42")
TITLES = ("BASELINE  |  TRAIN SEED 42", "FULL DR  |  TRAIN SEED 42", "DYNA PPO  |  TRAIN SEED 42")
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
                "evaluate_ppo_study.py", "train_dyna_ppo.py", "evaluate_dyna_ppo.py", "record_dyna_ppo_video.py"}
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
    report_path = output / policy_id / f"ppo_study_{policy_id}_seed3042.json"
    report = json.loads(report_path.read_text())
    video = output / policy_id / f"ppo_study_{policy_id}_seed3042.mp4"
    expected = {
        "status": "completed", "policy_id": policy_id, "seed": 3042, "num_envs": 1,
        "protocol_sha256": sha256(protocol_path), "checkpoint_sha256": entry["sha256"],
        "evaluation_environment_variant": "full_dr_for_all_policies", "domain_mode": "randomized",
        "recording_script_sha256": sha256(ROOT / "scripts/record_dyna_ppo_video.py"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"{policy_id}: mismatching recording metadata {key}")
    if not report["domain_readback"]["passed"]:
        raise RuntimeError(f"{policy_id}: physical parameter readback failed")
    if canonical_hash(report["initial_conditions"]) != report["initial_condition_sha256"]:
        raise RuntimeError(f"{policy_id}: initial condition snapshot hash is invalid")
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
    parser.add_argument("--output-dir", type=Path, help="Default: protocol directory/videos")
    parser.add_argument("--assemble-only", action="store_true", help="Validate existing completed originals; skip rendering")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output = (args.output_dir or protocol_path.parent / "videos").resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output-dir must stay inside the Sesame workspace")
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "comparison_validation.json"
    comparison = output / "ppo_study_comparison_seed3042.mp4"
    thumbnail = output / "comparison_thumbnail.png"
    report = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "orchestrator_sha256": sha256(__file__), "policies": list(POLICIES),
        "evaluation_seed": 3042, "condition": "randomized", "num_envs_per_video": 1,
        "comparison_note": "Exact paired single-environment initial states and full randomized test dynamics for all three policies. This is separate from batched study evaluation; same seed with different CUDA draw shapes need not produce the same physical samples.",
        "presentation": "Three actual live-policy recordings side by side, scaled to 640x360 with a 40px title strip per column; existing world-aligned camera, grid and HUD preserved. No synthetic robot imagery.",
        "early_termination_handling": "If an episode ends before 10s, retain its last actual video frame with an explicit EPISODE ENDED label for the remaining comparison duration; original video remains unpadded.",
        "originals": [], "visual_inspection": "pending human/agent image inspection",
    }
    try:
        # Validate every model BEFORE creating the first GPU application.
        entries = {key: resolve_video_policy(protocol_path, key)[1] for key in POLICIES}
        env_vars = dict(os.environ, CONDA_NO_PLUGINS="true", PYTHONDONTWRITEBYTECODE="1")
        initial_reference = None
        for policy_id in POLICIES:
            destination = output / policy_id
            destination.mkdir(parents=True, exist_ok=True)
            existing_report = destination / f"ppo_study_{policy_id}_seed3042.json"
            if not existing_report.exists():
                if args.assemble_only:
                    raise FileNotFoundError(existing_report)
                assert_gpu_slot_idle()
                command = [str(CONDA), "run", "-n", "env_isaaclab", "--no-capture-output", str(PYTHON),
                           "-u", str(ROOT / "scripts/record_dyna_ppo_video.py"), "--protocol", str(protocol_path),
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
        filters.append("[c0][c1][c2]hstack=inputs=3[vout]")
        command += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-an", "-c:v", "h264_nvenc",
                    "-preset", "slow", "-rc", "vbr", "-cq", "19", "-b:v", "6M", "-pix_fmt", "yuv420p",
                    "-r", "25", "-frames:v", "250", "-movflags", "+faststart", str(comparison)]
        report["assembly_command"] = command
        print("ASSEMBLE_START", flush=True)
        with (output / "comparison_ffmpeg.log").open("xb") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        probe = ffprobe(comparison)
        stream = probe["streams"][0]
        if (stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"], stream["nb_frames"]) != ("h264", 1920, 400, "25/1", "250"):
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
    raise SystemExit(main())
