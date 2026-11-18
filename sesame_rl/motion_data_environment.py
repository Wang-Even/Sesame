"""Data capture only: preserve the frozen task and snapshot before auto-reset.

Import after Isaac AppLauncher. Contact reporting is enabled by the collector.
All kinematics, policy inputs, simulation and diagnostics execute on CUDA.
"""
import xml.etree.ElementTree as ET

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply

from .heading_environment import SesameHeadingEnv
from .environment import ROOT


class SesameMotionDataEnv(SesameHeadingEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self.contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0,
            history_length=0, track_air_time=False, debug_vis=False,
        ))
        self.scene.sensors["dataset_contacts"] = self.contacts

    def __init__(self, cfg, **kwargs):
        self.capture_enabled = False
        self.capture_next = None
        super().__init__(cfg, **kwargs)
        if cfg.dr_observation_noise or cfg.observation_noise_model or cfg.action_noise_model or cfg.events:
            raise ValueError("Collector expects no hidden observation noise, action noise or interval events")
        xml = ET.parse(ROOT / "assets/isaac/sesame/import_source/sesame.urdf").getroot()
        self.foot_names = ["r3_foot", "r4_foot", "l3_foot", "l4_foot"]
        parents, offsets = [], []
        for name in self.foot_names:
            joint = next(j for j in xml.findall("joint") if j.find("child").get("link") == name)
            parents.append(joint.find("parent").get("link"))
            offsets.append([float(v) for v in joint.find("origin").get("xyz").split()])
        self.foot_parent_ids = [self.robot.body_names.index(name) for name in parents]
        self.foot_offsets = torch.tensor(offsets, device=self.device)
        self.contact_to_body = [self.contacts.body_names.index(name) for name in self.robot.body_names]
        self.capture_enabled = True

    def snapshot(self):
        data = self.robot.data
        root = data.root_state_w.clone()
        root[:, :3] -= self.scene.env_origins
        bodies = data.body_link_state_w.clone()
        bodies[:, :, :3] -= self.scene.env_origins[:, None, :]
        parents = bodies[:, self.foot_parent_ids]
        offsets = self.foot_offsets[None].expand(self.num_envs, -1, -1)
        offset_w = quat_apply(parents[..., 3:7].reshape(-1, 4), offsets.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        feet = parents[..., :3] + offset_w
        feet_vel = parents[..., 7:10] + torch.linalg.cross(parents[..., 10:13], offset_w, dim=-1)
        return {
            "root_state": root,
            "joint_pos": data.joint_pos[:, self.firmware_joint_ids].clone(),
            "joint_vel": data.joint_vel[:, self.firmware_joint_ids].clone(),
            "body_link_state": bodies,
            "foot_pos": feet,
            "foot_vel": feet_vel,
            "applied_target": self._targets.clone(),
            "command_target": self._command_targets.clone(),
            "target_history": self._target_history.clone(),
            "previous_action": self._actions.clone(),
        }

    def _get_rewards(self):
        reward = super()._get_rewards()
        if self.capture_enabled:
            self.capture_next = self.snapshot()
            self.capture_transition = {
                "next_observation": self._get_observations()["policy"].clone(),
                "executed_action": self._actions.clone(),
                "joint_torque_estimate": self.robot.data.applied_torque[:, self.firmware_joint_ids].clone(),
                "body_contact_force": self.contacts.data.net_forces_w[:, self.contact_to_body].clone(),
                "terminated": self.reset_terminated.clone(),
                "truncated": self.reset_time_outs.clone(),
                "fell": self._fell.clone(),
                "out_of_bounds": self._outside.clone(),
                "invalid": self._invalid.clone(),
                "legacy_reward": reward.clone(),
            }
        return reward
