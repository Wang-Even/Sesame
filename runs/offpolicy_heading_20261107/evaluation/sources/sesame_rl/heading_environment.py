"""A single reward change for the frozen Sesame walking task.

This environment keeps the original PPO interface, full domain randomization,
actuator processing, termination rules and every original reward component.
It adds a time-integrated absolute-heading penalty and inherits the study's
control-rate evaluation diagnostics. No model or simulation work runs on CPU.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from isaaclab.utils import configclass

from .study_environment import SesameStudyEnv, SesameStudyEnvCfg


@configclass
class SesameHeadingEnvCfg(SesameStudyEnvCfg):
    """The coefficient multiplies a reward rate, before the control timestep."""

    heading_penalty_scale = 0.5


def _checked_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("heading_penalty_scale must be finite and nonnegative")
    return scale


def make_heading_env_cfg(
    num_envs: int = 256,
    seed: int = 42,
    device: str = "cuda:0",
    episode_length_s: float = 10.0,
    heading_penalty_scale: float = 0.5,
) -> SesameHeadingEnvCfg:
    """Make a full-DR configuration without allocating simulation tensors.

    A zero coefficient is allowed as a diagnostic control. The default and
    formal training configuration use a coefficient of 0.5. Evaluation may
    switch the inherited domain randomization on or off before resetting.
    """
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("CUDA is required; CPU fallback is disabled")
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if not math.isfinite(float(episode_length_s)) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and positive")
    cfg = SesameHeadingEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.episode_length_s = float(episode_length_s)
    cfg.study_variant = "full_dr"
    cfg.domain_randomization = True
    cfg.heading_penalty_scale = _checked_scale(heading_penalty_scale)
    return cfg


def heading_reward_rate(
    yaw: torch.Tensor,
    commands_w: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return a CUDA heading reward rate, without multiplying by timestep.

    ``yaw`` is world yaw in radians, shaped [N]. ``commands_w`` is the
    world-frame velocity command, shaped [N, 3]. Only its XY direction defines
    target yaw. Cosine makes the term continuous across the +/-pi boundary.
    An exactly zero XY command disables the term for that environment.
    """
    coefficient = _checked_scale(scale)
    if not yaw.is_cuda or not commands_w.is_cuda:
        raise RuntimeError("Heading reward computation requires CUDA tensors")
    if yaw.device != commands_w.device:
        raise ValueError("Yaw and command tensors must use the same CUDA device")
    if yaw.ndim != 1 or commands_w.ndim != 2 or commands_w.shape != (yaw.shape[0], 3):
        raise ValueError("Expected yaw [N] and world velocity commands [N, 3]")
    moving = commands_w[:, :2].square().sum(dim=1) > 0.0
    command_yaw = torch.atan2(commands_w[:, 1], commands_w[:, 0])
    rate = -coefficient * (1.0 - torch.cos(yaw - command_yaw))
    return torch.where(moving, rate, torch.zeros_like(rate))


def heading_reward_metadata(cfg: SesameHeadingEnvCfg | None = None) -> dict[str, Any]:
    """Serializable specification for launchers, checkpoints and reports."""
    coefficient = _checked_scale(0.5 if cfg is None else cfg.heading_penalty_scale)
    control_dt = 0.02 if cfg is None else float(cfg.sim.dt) * int(cfg.decimation)
    return {
        "reward_version": "absolute_heading_v1",
        "base_reward": "All frozen SesameWalkEnv reward components retained unchanged",
        "new_term": "heading",
        "heading_penalty_scale": coefficient,
        "reward_rate_formula": "-scale * (1 - cos(world_yaw - atan2(command_world_y, command_world_x)))",
        "step_reward_formula": "heading_reward_rate * control_dt_s",
        "control_dt_s": control_dt,
        "zero_planar_command": "heading term is exactly zero",
        "nonfinite_rate": "mapped to zero, consistent with the frozen base reward handling",
        "target_heading": "world XY velocity-command direction; zero radians for the original +X command",
        "accounting": "Added once to returned reward, episode return and RewardRate/heading accumulator",
        "observation_dimension": 36,
        "action_dimension": 8,
        "training_physics": "Full domain randomization, unchanged from DR-PPO",
        "metric_reference": "Inherited study heading metrics remain relative to world +X, matching this fixed +X task",
    }


class SesameHeadingEnv(SesameStudyEnv):
    """Full-DR walking with an additional absolute-heading reward term."""

    cfg: SesameHeadingEnvCfg

    def __init__(self, cfg: SesameHeadingEnvCfg, **kwargs):
        _checked_scale(cfg.heading_penalty_scale)
        if cfg.study_variant != "full_dr":
            raise ValueError("Heading training uses the frozen full_dr dynamics variant")
        self._heading_ready = False
        super().__init__(cfg, **kwargs)
        if "heading" in self._reward_sums:
            raise RuntimeError("The frozen base reward unexpectedly already defines a heading term")
        self._reward_sums["heading"] = torch.zeros_like(self._episode_return)
        self._heading_ready = True

    def heading_reward_metadata(self) -> dict[str, Any]:
        return heading_reward_metadata(self.cfg)

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if not self._heading_ready:
            return reward
        rate = heading_reward_rate(self._heading_yaw(), self._commands_w, self.cfg.heading_penalty_scale)
        component = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0) * self.step_dt
        self._episode_return.add_(component)
        self._reward_sums["heading"].add_(component)
        return reward + component


make_env_cfg = make_heading_env_cfg
