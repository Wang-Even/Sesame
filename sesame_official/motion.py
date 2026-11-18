"""Explicit multi-clip adapter for public mjlab 1.5.0.

Uniform RSI is a supported official SuperMimic option, shared by both methods.
No transition crosses clip boundaries; ends are timeouts, not falls.
"""
from dataclasses import dataclass
import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from sesame_official.kinematics import SesameFK


class SesameMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            ends = data["motion_ends"]
            if tuple(data["body_names"].tolist()) != tuple(self.robot.body_names):
                raise ValueError("Motion body order does not match robot")
            if tuple(data["joint_names"].tolist()) != tuple(self.robot.joint_names):
                raise ValueError("Motion joint order does not match robot")
        m = self.motion
        m.motion_ends = torch.as_tensor(ends, device=self.device, dtype=torch.long)
        m.root_pos_w = m.body_pos_w[:, self.motion_anchor_body_index]
        m.root_quat_w = m.body_quat_w[:, self.motion_anchor_body_index]
        m.body_state = lambda frames: (m.body_pos_w[frames], m.body_quat_w[frames],
                                      m.body_lin_vel_w[frames], m.body_ang_vel_w[frames])
        m.fk = SesameFK(list(self.robot.joint_names), list(cfg.body_names), self.device)

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori,
                                     root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        # Public mjlab's limits may have singleton world dimension when the
        # MuJoCo field is not randomized. Its MotionCommand indexes it as N.
        limits = self.robot.data.soft_joint_pos_limits
        limits = limits if limits.shape[0] == 1 else limits[env_ids]
        joint_pos = joint_pos.clamp(limits[...,0], limits[...,1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos,root_ori,root_lin_vel,root_ang_vel],dim=-1)
        self.robot.write_root_state_to_sim(root_state,env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _update_command(self):
        # Manager calls this after reset too: retain the sampled initial frame.
        advance = (self._env.episode_length_buf > 0).long()
        m = self.motion
        clip = torch.searchsorted(m.motion_ends, self.time_steps, right=True)
        self.time_steps = torch.minimum(self.time_steps + advance, m.motion_ends[clip] - 1)
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class SesameMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        if self.sampling_mode not in ("uniform", "start"):
            raise ValueError("Sesame adapter supports uniform/start RSI only")
        return SesameMotionCommand(self, env)


def motion_end(env):
    c = env.command_manager.get_term("motion")
    clip = torch.searchsorted(c.motion.motion_ends, c.time_steps, right=True)
    return c.time_steps >= c.motion.motion_ends[clip] - 1
