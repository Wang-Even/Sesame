"""Sesame URDF import; joint gains/limits inherited from the Isaac experiment."""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from pathlib import Path
URDF = Path(__file__).resolve().parents[1] / "assets/isaac/sesame/import_source/sesame.urdf"
SOURCE_JOINTS = ["R1", "R2", "L1", "L2", "R4", "R3", "L3", "L4"]


def get_spec():
    tree = ET.parse(URDF)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str(URDF.parent / mesh.get("filename")))
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", {"discardvisual": "false", "fusestatic": "false", "strippath": "false"})
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    base = spec.body("base")
    base.add_freejoint(name="root")
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # Robot-ground contacts only, matching the original task.
            geom.contype = 0
            geom.conaffinity = 1
            geom.friction = [0.8, .005, .0001]
            geom.solref = [.004, 1.]
            geom.condim = 3
    return spec


def robot_cfg():
    positions = dict(zip(SOURCE_JOINTS, [v * math.pi / 180 for v in (135,45,45,135,0,180,0,180)]))
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,.10), joint_pos=positions),
        articulation=EntityArticulationInfoCfg(
            actuators=(DcMotorActuatorCfg(target_names_expr=(".*",), stiffness=5., damping=.0025,
                                         effort_limit=.18, saturation_effort=.18,
                                         velocity_limit=10.471975512),),
            soft_joint_pos_limit_factor=.95,
        ),
    )
