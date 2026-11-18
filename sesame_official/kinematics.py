"""Differentiable URDF FK for the public mjlab compatibility layer.

The private mjlab_research FK implementation is unavailable. This adapter is
project code, not an upstream implementation. Numerical execution is CUDA only.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/isaac/sesame/import_source/sesame.urdf"
SOURCE_JOINTS = ("R1", "R2", "L1", "L2", "R4", "R3", "L3", "L4")


def rpy_quat(rpy):
    r, p, y = (v / 2 for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)


class SesameFK(torch.nn.Module):
    def __init__(self, joint_names, body_names, device="cuda:0"):
        super().__init__()
        if not str(device).startswith("cuda"):
            raise ValueError("FK requires CUDA")
        urdf = ET.parse(URDF).getroot()
        remaining = list(urdf.findall("joint"))
        names, nodes, offsets, rotations, axes = ["base"], [], [], [], []
        while remaining:
            progressed = False
            for j in list(remaining):
                parent = j.find("parent").get("link")
                if parent not in names:
                    continue
                origin = j.find("origin")
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                idx = -1 if j.get("type") == "fixed" else joint_names.index(j.get("name"))
                axis = [0., 0., 1.] if idx < 0 else [float(x) for x in j.find("axis").get("xyz").split()]
                nodes.append((names.index(parent), idx))
                names.append(j.find("child").get("link"))
                offsets.append(xyz); rotations.append(rpy_quat(rpy)); axes.append(axis)
                remaining.remove(j); progressed = True
            if not progressed:
                raise ValueError("URDF is not a rooted tree")
        self.nodes = nodes
        self.names = names
        self.selected = [names.index(n) for n in body_names]
        for name, val in (("offsets", offsets), ("rotations", rotations), ("axes", axes)):
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32, device=device))

    def forward(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        pos, quat, vel, ang = [root_pos], [root_quat], [root_lin_vel], [root_ang_vel]
        for k, (parent, joint) in enumerate(self.nodes):
            offset = quat_apply(quat[parent], self.offsets[k].expand_as(root_pos))
            p = pos[parent] + offset
            q = quat_mul(quat[parent], self.rotations[k].expand_as(root_quat))
            v = vel[parent] + torch.cross(ang[parent], offset, dim=-1)
            w = ang[parent]
            if joint >= 0:
                axis = self.axes[k].expand_as(root_pos)
                axis_world = quat_apply(q, axis)
                half = joint_pos[..., joint:joint+1] * .5
                rotation = torch.cat((half.cos(), half.sin() * axis), dim=-1)
                q = quat_mul(q, rotation)
                w = w + axis_world * joint_vel[..., joint:joint+1]
            pos.append(p); quat.append(q); vel.append(v); ang.append(w)
        return tuple(torch.stack([values[i] for i in self.selected], dim=-2)
                     for values in (pos, quat, vel, ang))
