"""Sesame tracking reward without foot tracking or joint-position-limit costs."""
from types import MethodType
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from supermimic.utils.helpers import rotation_error_magnitude
from supermimic.utils.tracking_reward import _reanchor_targets

REWARD_VERSION = "sesame_nonfoot_no_joint_limit_20261115"
FOOT_BODIES = frozenset(("r3_foot", "r4_foot", "l3_foot", "l4_foot"))


def configure_env_reward(cfg):
    """Apply the same body selection to every native tracking reward."""
    bodies = tuple(n for n in cfg.commands["motion"].body_names if n not in FOOT_BODIES)
    if not bodies or bodies[0] != "base":
        raise ValueError("Tracking reward requires base first in body names")
    cfg.rewards.pop("joint_limit", None)
    for name in ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
        cfg.rewards[name].params["body_names"] = bodies


def tracking_reward(aux, positions, rotations, velocities, angular_velocities,
                    action, last_action, target_pos, target_quat, target_vel,
                    target_ang_vel, dt):
    # Inputs contain the base and non-foot links only. Keep upstream weights
    # and tolerances for those links and the root; no joint-range term exists.
    pos, rot = _reanchor_targets(target_pos, target_quat, aux[..., :3], aux[..., 3:7])
    terms = [
        ((positions[..., 0, :] - target_pos[..., 0, :]).square().sum(-1), .5, .3),
        (rotation_error_magnitude(rotations[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])).square(), .5, .4),
        ((positions - pos).square().sum(-1).mean(-1), 1., .3),
        (rotation_error_magnitude(rotations, rot).square().mean(-1), 1., .4),
        ((velocities - target_vel).square().sum(-1).mean(-1), 1., 1.),
        ((angular_velocities - target_ang_vel).square().sum(-1).mean(-1), 1., 3.14),
    ]
    score = sum(weight * torch.exp(-error / std**2) for error, weight, std in terms)
    return dt * (score - .1 * (action - last_action).square().sum(-1))


def _imagination_reward(self, next_rep_state, next_aux_state, next_joint_state,
                        action, last_action, target_body_pos_w, target_body_quat_w,
                        target_body_lin_vel_w, target_body_ang_vel_w):
    ids = self._sesame_reward_body_ids
    pos, rot, vel, avel = self.world_frame_bodies(next_aux_state, next_rep_state)
    return tracking_reward(
        next_aux_state, pos[..., ids, :], rot[..., ids, :, :], vel[..., ids, :], avel[..., ids, :],
        action, last_action, target_body_pos_w[..., ids, :], target_body_quat_w[..., ids, :],
        target_body_lin_vel_w[..., ids, :], target_body_ang_vel_w[..., ids, :], self.dt)


def bind_imagination_reward(representation, env):
    """Install before policy-learner creation/compilation, without editing upstream."""
    raw = env.unwrapped
    command = raw.command_manager.get_term("motion")
    ids = [i for i, name in enumerate(command.cfg.body_names) if name not in FOOT_BODIES]
    representation.register_buffer("_sesame_reward_body_ids",
        torch.tensor(ids, dtype=torch.long, device=raw.device), persistent=False)
    representation.compute_wm_reward_from_reference = MethodType(_imagination_reward, representation)
