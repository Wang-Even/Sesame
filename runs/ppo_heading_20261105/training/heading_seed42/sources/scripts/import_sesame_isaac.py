#!/usr/bin/env python3
"""Convert the pinned sesame-ml URDF and check its eight joints on CUDA.

Source model parameters are reference estimates, not hardware calibration.
This is an import/physics check, not locomotion training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/upstream/sesame-ml/src/sesame_ml/assets"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=SOURCE / "urdf/sesame.urdf")
parser.add_argument("--output", type=Path, default=ROOT / "assets/isaac/sesame")
parser.add_argument("--steps", type=int, default=32)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
report_path = ROOT / "reports/sesame_isaac_import.json"
runtime = ROOT / ".runtime/sesame_import"
for key, relative in {"XDG_CACHE_HOME":"cache", "XDG_CONFIG_HOME":"config", "XDG_DATA_HOME":"data", "CUDA_CACHE_PATH":"cuda_cache"}.items():
    directory = runtime / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
report = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_repository": "https://github.com/lukehollis/sesame-ml",
    "source_revision": "a3910f67b4aeb9ca9080a1ba1ffa971e53e35a18",
    "source_urdf": str(args.urdf),
    "source_urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
    "device": "cuda:0", "passed": False, "training_performed": False,
    "limitations": [
        "Reference model, not course-provided or hardware-calibrated parameters.",
        "Short free-fall articulation load check; no floor contact, walking policy, or reward task tested.",
        "Isaac Sim 4.5 joint friction semantics differ from MuJoCo frictionloss; no exact simulator equivalence claimed.",
    ],
}
app = None
exit_code = 1
try:
    urdf = ET.parse(args.urdf).getroot()
    joints = [j for j in urdf.findall("joint") if j.get("type") == "revolute"]
    assert len(joints) == 8
    names = [j.get("name") for j in joints]
    limits = {j.get("name"): {k: float(v) for k, v in j.find("limit").attrib.items()} for j in joints}
    masses = {link.get("name"): float(link.find("inertial/mass").get("value")) for link in urdf.findall("link") if link.find("inertial/mass") is not None}
    meshes = sorted({str((args.urdf.parent / m.get("filename")).resolve()) for m in urdf.findall(".//mesh")})
    missing_meshes = [p for p in meshes if not Path(p).is_file()]
    assert not missing_meshes, missing_meshes
    # Isaac Sim 4.5 uses mesh basenames directly as USD prim identifiers and
    # rejects the upstream hyphens. Copy bytes unchanged into a derived import
    # input and rewrite only filenames; leave the pinned source untouched.
    import_source = args.output / "import_source"
    import_meshes = import_source / "meshes"
    import_meshes.mkdir(parents=True, exist_ok=True)
    mesh_name_map = {}
    for mesh in urdf.findall(".//mesh"):
        original = (args.urdf.parent / mesh.get("filename")).resolve()
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", original.stem) + original.suffix
        destination = import_meshes / safe_name
        shutil.copyfile(original, destination)
        assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest()
        mesh_name_map[original.name] = safe_name
        mesh.set("filename", f"meshes/{safe_name}")
    import_urdf = import_source / "sesame.urdf"
    ET.ElementTree(urdf).write(import_urdf, encoding="utf-8", xml_declaration=True)
    report["import_filename_adaptation"] = {"derived_urdf":str(import_urdf), "mesh_names":mesh_name_map, "mesh_bytes_unchanged":True, "reason":"Isaac Sim 4.5 requires valid USD identifiers for mesh basenames; upstream hyphens are invalid."}
    mjcf = ET.parse(SOURCE / "mjcf/sesame.xml").getroot()
    reference_pose = {j.get("name"): float(j.get("ref")) for j in mjcf.findall(".//worldbody//joint") if j.get("name") in names}
    kp = {a.get("joint"): float(a.get("kp")) for a in mjcf.findall("actuator/position")}
    damping = {j.get("name"): float(j.find("dynamics").get("damping")) for j in joints}
    report["source_model"] = {
        "joint_names_firmware_order": names, "joint_limits": limits,
        "link_masses_kg": masses, "total_mass_kg": sum(masses.values()),
        "unique_visual_meshes": len(meshes), "all_meshes_resolve": True,
        "initial_joint_pose_rad_from_mjcf_ref": reference_pose,
        "servo_stiffness_from_mjcf_kp": kp, "damping_from_urdf": damping,
    }
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable; no CPU fallback"
    torch.cuda.set_device(0)
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    kit_args = " ".join(["--portable-root", str(runtime / "kit"), f"--/log/file={runtime / 'kit.log'}", "--/app/settings/persistent=false", "--/app/telemetry/enabled=false", "--/renderer/multiGpu/enabled=false"])
    app = AppLauncher(headless=True, device="cuda:0", kit_args=kit_args).app
    import warp as wp
    wp.config.kernel_cache_dir = str(runtime / "warp_cache")
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    class LocalUrdfConverter(UrdfConverter):
        """Use the installed 4.5 importer instead of Lab's pinned 5.x version.

        Keep source tensors explicitly: Lab's generic converter does not expose
        import_inertia_tensor in its config, and fixed marker links have no mass.
        """
        def __init__(self, cfg):
            enable_extension("isaacsim.asset.importer.urdf")
            from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface
            self._urdf_interface = acquire_urdf_interface()
            AssetConverterBase.__init__(self, cfg=cfg)

        def _convert_asset(self, cfg):
            import omni.kit.commands
            import_config = self._get_urdf_import_config()
            parsed, self._robot_model = omni.kit.commands.execute(
                "URDFParseFile", urdf_path=cfg.asset_path, import_config=import_config,
            )
            if not parsed:
                raise RuntimeError("URDFParseFile failed")
            self._update_joint_parameters()
            self._robot_model.root_link = cfg.root_link_name
            imported, result = omni.kit.commands.execute(
                "URDFImportRobot", urdf_path=cfg.asset_path,
                urdf_robot=self._robot_model, import_config=import_config,
                dest_path=self.usd_path,
            )
            if not imported:
                raise RuntimeError(f"URDFImportRobot failed: {result!r}")

        def _get_urdf_import_config(self):
            import omni.kit.commands
            _, config = omni.kit.commands.execute("URDFCreateImportConfig")
            config.set_distance_scale(1.0)
            config.set_make_default_prim(True)
            config.set_create_physics_scene(False)
            config.set_density(self.cfg.link_density)
            config.set_convex_decomp(False)
            config.set_collision_from_visuals(False)
            config.set_merge_fixed_joints(True)
            config.set_fix_base(False)
            config.set_self_collision(False)
            config.set_parse_mimic(False)
            config.set_replace_cylinders_with_capsules(False)
            config.set_import_inertia_tensor(True)
            if hasattr(config, "set_merge_fixed_ignore_inertia"):
                config.set_merge_fixed_ignore_inertia(False)
            return config

    cfg = UrdfConverterCfg(
        asset_path=str(import_urdf.resolve()), usd_dir=str(args.output.resolve()),
        usd_file_name="sesame.usd", force_usd_conversion=True,
        fix_base=False, root_link_name="base", merge_fixed_joints=True,
        self_collision=False, make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=kp, damping=damping),
        ),
    )
    converter = LocalUrdfConverter(cfg)
    assert Path(converter.usd_path).is_file()
    report["usd_path"] = converter.usd_path
    report["conversion"] = {
        "free_base": True, "import_inertia_tensor": True,
        "merge_fixed_joints": True, "inertia_handling": "4.5 importer import_inertia_tensor=True; no ignore-inertia override supported",
        "fixed_link_note": "Six empty foot/IMU/camera marker links merged into parent bodies to avoid artificial masses; original URDF retains their frames.",
        "self_collision": False,
        "drive_note": "Position stiffness borrowed from upstream MJCF kp=5; damping=0.0025 and effort/velocity limits from URDF. No hardware calibration inferred.",
    }
    stage = Usd.Stage.Open(converter.usd_path)
    usd_joints = []
    usd_masses = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            usd_joints.append({"name":prim.GetName(),"path":str(prim.GetPath()),"axis":j.GetAxisAttr().Get(),"lower_deg":j.GetLowerLimitAttr().Get(),"upper_deg":j.GetUpperLimitAttr().Get()})
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is not None:
                usd_masses[prim.GetName()] = float(mass)
    assert len(usd_joints) == 8, usd_joints
    for joint in usd_joints:
        source_limit = limits[joint["name"]]
        assert abs(math.radians(joint["lower_deg"]) - source_limit["lower"]) < 1e-6
        assert abs(math.radians(joint["upper_deg"]) - source_limit["upper"]) < 1e-6
    assert abs(sum(usd_masses.values()) - sum(masses.values())) < 1e-5, usd_masses
    report["usd_metadata"] = {"revolute_joints":usd_joints,"link_masses_kg":usd_masses,"total_mass_kg":sum(usd_masses.values())}
    del stage

    dt = float(mjcf.find("option").get("timestep"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=dt))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Sesame",
        spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos=reference_pose, joint_vel={".*":0.0}),
        actuators={"reference_servos": ImplicitActuatorCfg(
            joint_names_expr=names, stiffness=kp, damping=damping,
            effort_limit_sim={name:limits[name]["effort"] for name in names},
            velocity_limit_sim={name:limits[name]["velocity"] for name in names},
        )},
    ))
    sim.reset()
    assert robot.num_joints == 8 and not robot.is_fixed_base
    joint_names = robot.joint_names
    assert set(joint_names) == set(names)
    targets = robot.data.default_joint_pos.clone()
    assert targets.is_cuda
    # Articulation stores configured defaults, but direct SimulationContext
    # reset does not write them as an RL environment reset would.
    robot.write_joint_state_to_sim(targets, robot.data.default_joint_vel)
    default_root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.reset()
    for _ in range(args.steps):
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
    torch.cuda.synchronize()
    joint_pos, joint_vel = robot.data.joint_pos, robot.data.joint_vel
    root_state = robot.data.root_state_w
    assert all(v.is_cuda and bool(torch.isfinite(v).all()) for v in [joint_pos,joint_vel,root_state])
    report["gpu_articulation"] = {
        "joint_count": robot.num_joints, "body_count": robot.num_bodies,
        "joint_order_isaac":joint_names,
        "firmware_order_indices_in_isaac": [joint_names.index(name) for name in names],
        "is_fixed_base":robot.is_fixed_base, "device":str(joint_pos.device),
        "steps":args.steps, "dt":dt, "finite_joint_and_root_states":True,
        "control_test": "Fixed upstream reference pose target, free-floating without ground contact; no trained policy.",
        "initial_joint_positions_rad":targets.tolist(),
        "final_joint_positions_rad":joint_pos.tolist(),
        "final_joint_velocities_rad_s":joint_vel.tolist(),
        "final_base_position_m":robot.data.root_pos_w.tolist(),
    }
    report["passed"] = True
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    traceback.print_exc()
finally:
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print("SESAME_ISAAC_RESULT="+json.dumps(report),flush=True)
    if app is not None:
        def timeout_exit():
            report["shutdown"]={"completed":False,"forced_process_exit":True,"timeout_seconds":15}
            report_path.write_text(json.dumps(report,indent=2)+"\n")
            print("SESAME_SHUTDOWN_TIMEOUT: saved results; exiting this process only",flush=True)
            os._exit(exit_code)
        timer=threading.Timer(15.0,timeout_exit)
        timer.daemon=True
        timer.start()
        app.close(wait_for_replicator=False)
        timer.cancel()
        report["shutdown"]={"completed":True,"forced_process_exit":False}
        report_path.write_text(json.dumps(report,indent=2)+"\n")
raise SystemExit(exit_code)
