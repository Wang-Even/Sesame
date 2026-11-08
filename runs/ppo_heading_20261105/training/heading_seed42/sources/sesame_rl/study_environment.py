"""Controlled delay ablation and diagnostic metrics for the frozen Sesame task.

The task, observations, reward, limits, and physics time step are inherited.
``fixed_delay`` consumes the complete DR sampler before replacing only its lag
time constant and integer control delay. Diagnostics use CUDA tensors and do
not enter observations or reward. CPU work is terminal-record serialization.
"""
from __future__ import annotations

from typing import Any

import torch
from isaaclab.utils import configclass

from .dr_environment import SesameDREnv, SesameDREnvCfg


@configclass
class SesameStudyEnvCfg(SesameDREnvCfg):
    study_variant = "full_dr"


def make_study_env_cfg(
    num_envs: int = 256, seed: int = 42, device: str = "cuda:0",
    episode_length_s: float = 10.0, variant: str = "full_dr",
) -> SesameStudyEnvCfg:
    if variant not in ("baseline", "full_dr", "fixed_delay"):
        raise ValueError(f"Unknown study variant: {variant}")
    if not device.startswith("cuda") or num_envs < 1 or episode_length_s <= 0:
        raise ValueError("CUDA, positive num_envs and positive duration required")
    cfg = SesameStudyEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.episode_length_s = episode_length_s
    cfg.study_variant = variant
    cfg.domain_randomization = variant != "baseline"
    return cfg


class SesameStudyEnv(SesameDREnv):
    """DR-compatible environment with complete-episode control-rate metrics.

    ``episode_records[*]['study_metrics']`` contains post-physics diagnostics,
    including the terminal sample before automatic reset. Target derivatives
    use the executed firmware-ordered ``_targets`` after clipping, slew, delay
    and lag, at 50 Hz. Target rate includes the first action from the reset
    target; target acceleration omits the first sample. Both RMS values average
    over time and all eight joints. Reset discontinuities never enter them.

    For common randomized evaluation, instantiate ``variant='full_dr'`` for
    every checkpoint, including checkpoints trained with fixed delay/baseline.
    The variant chooses training dynamics, not the policy architecture.
    """
    cfg: SesameStudyEnvCfg
    metric_definitions = {
        "sampling": "post-physics, pre-auto-reset; one sample per 0.02-second control step, including terminal step",
        "velocity_x_rmse_m_s": "sqrt(mean((world_vx - command_forward_m_s)^2))",
        "velocity_xy_tracking_rmse_m_s": "sqrt(mean((world_vx - command_forward_m_s)^2 + world_vy^2))",
        "heading_rms_rad": "sqrt(mean(wrapped world yaw^2)); world +X is zero heading",
        "heading_max_abs_rad": "max(abs(wrapped world yaw)) over post-physics samples",
        "lateral_mean_abs_m": "mean(abs(world_y - reset_world_y))",
        "lateral_rms_m": "sqrt(mean((world_y - reset_world_y)^2))",
        "applied_target_rate_rms_rad_s": "sqrt(mean_time_and_8_joints(((executed_target[t]-executed_target[t-1])/control_dt)^2)); reset target is t=0",
        "applied_target_acceleration_rms_rad_s2": "sqrt(mean_time_t>=2_and_8_joints(((target_rate[t]-target_rate[t-1])/control_dt)^2)); first sample omitted",
        "actual_joint_acceleration_rms_rad_s2": "sqrt(mean_time_and_8_joints(((actual_joint_velocity[t]-actual_joint_velocity[t-1])/control_dt)^2)); first sample uses known reset velocity; 50Hz finite difference, not 500Hz acceleration peaks",
        "acceleration_if_no_valid_samples": "null; not a measured zero",
        "excluded": "reset jumps, physics-step acceleration peaks, mechanical power and raw actor action clipping",
    }

    def __init__(self, cfg: SesameStudyEnvCfg, **kwargs):
        if cfg.study_variant not in ("baseline", "full_dr", "fixed_delay"):
            raise ValueError(f"Unknown study variant: {cfg.study_variant}")
        self._study_ready = False
        super().__init__(cfg, **kwargs)
        self._study_sums = torch.zeros((self.num_envs, 8), device=self.device)
        self._study_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._study_accel_count = torch.zeros_like(self._study_count)
        self._study_heading_max = torch.zeros(self.num_envs, device=self.device)
        self._study_previous_target_rate = torch.zeros_like(self._targets)
        self._study_pending_target_rate = torch.zeros_like(self._targets)
        self._study_pending_target_accel = torch.zeros_like(self._targets)
        self._study_pending_accel_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._study_previous_joint_velocity = self.robot.data.joint_vel[:, self.firmware_joint_ids].clone()
        # These snapshots deliberately survive auto-reset until the next step.
        # They allow an independent CUDA recomputation including terminal state.
        self.last_control_sample = {
            "root_state_w": self.robot.data.root_state_w.clone(),
            "applied_targets": self._targets.clone(),
            "joint_velocity_firmware": self._study_previous_joint_velocity.clone(),
        }
        self._study_ready = True

    def _set_episode_parameters(self, env_ids: torch.Tensor):
        super()._set_episode_parameters(env_ids)
        if self._domain_requested and self.cfg.study_variant == "fixed_delay":
            self.domain_parameters["target_tau_s"][env_ids] = 0.04
            self.domain_parameters["delay_control_steps"][env_ids] = 1

    def _pre_physics_step(self, actions: torch.Tensor):
        if not self._study_ready:
            return super()._pre_physics_step(actions)
        previous_target = self._targets.clone()
        super()._pre_physics_step(actions)
        self._study_pending_target_rate.copy_((self._targets - previous_target) / self.step_dt)
        self._study_pending_accel_valid.copy_(self._study_count > 0)
        self._study_pending_target_accel.copy_(
            (self._study_pending_target_rate - self._study_previous_target_rate) / self.step_dt,
        )

    def _get_rewards(self):
        reward = super()._get_rewards()
        if not self._study_ready:
            return reward
        root = self.robot.data.root_state_w
        vx_error_sq = (root[:, 7] - self.cfg.command_forward_m_s).square()
        yaw = self._heading_yaw()
        lateral = root[:, 1] - self._episode_start_y
        valid_accel = self._study_pending_accel_valid
        joint_velocity = self.robot.data.joint_vel[:, self.firmware_joint_ids]
        joint_acceleration = (joint_velocity - self._study_previous_joint_velocity) / self.step_dt
        self._study_sums.add_(torch.stack((
            vx_error_sq,
            vx_error_sq + root[:, 8].square(),
            yaw.square(), lateral.abs(), lateral.square(),
            self._study_pending_target_rate.square().mean(dim=1),
            torch.where(valid_accel, self._study_pending_target_accel.square().mean(dim=1), 0.0),
            joint_acceleration.square().mean(dim=1),
        ), dim=1))
        self._study_count.add_(1)
        self._study_accel_count.add_(valid_accel.long())
        self._study_heading_max.copy_(torch.maximum(self._study_heading_max, yaw.abs()))
        self._study_previous_target_rate.copy_(self._study_pending_target_rate)
        self._study_previous_joint_velocity.copy_(joint_velocity)
        self.last_control_sample["root_state_w"].copy_(root)
        self.last_control_sample["applied_targets"].copy_(self._targets)
        self.last_control_sample["joint_velocity_firmware"].copy_(joint_velocity)
        return reward

    def _metric_rows(self, env_ids: torch.Tensor) -> list[dict[str, Any]]:
        count = self._study_count[env_ids]
        accel_count = self._study_accel_count[env_ids]
        means = self._study_sums[env_ids] / count.clamp_min(1)[:, None]
        values = torch.stack((
            means[:, 0].sqrt(), means[:, 1].sqrt(), means[:, 2].sqrt(),
            self._study_heading_max[env_ids], means[:, 3], means[:, 4].sqrt(),
            means[:, 5].sqrt(),
            (self._study_sums[env_ids, 6] / accel_count.clamp_min(1)).sqrt(),
            means[:, 7].sqrt(),
        ), dim=1).tolist()
        names = (
            "velocity_x_rmse_m_s", "velocity_xy_tracking_rmse_m_s", "heading_rms_rad",
            "heading_max_abs_rad", "lateral_mean_abs_m", "lateral_rms_m",
            "applied_target_rate_rms_rad_s", "applied_target_acceleration_rms_rad_s2",
            "actual_joint_acceleration_rms_rad_s2",
        )
        result = []
        for row, samples, acceleration_samples in zip(values, count.tolist(), accel_count.tolist(), strict=True):
            metrics = dict(zip(names, row, strict=True))
            metrics.update(samples=samples, target_rate_samples=samples,
                           target_acceleration_samples=acceleration_samples,
                           actual_joint_acceleration_samples=samples,
                           control_dt_s=self.step_dt, target_joint_count=8)
            if acceleration_samples == 0:
                metrics["applied_target_acceleration_rms_rad_s2"] = None
            result.append(metrics)
        return result

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not self._study_ready:
            return super()._reset_idx(env_ids)
        env_ids = self._ids(env_ids)
        completed = env_ids[(self.episode_length_buf[env_ids] > 0) & self.reset_buf[env_ids]]
        metrics = self._metric_rows(completed)
        completed_ids = completed.tolist()
        super()._reset_idx(env_ids)
        if completed_ids:
            records = list(self.episode_records)[-len(completed_ids):]
            for record, env_id, values in zip(records, completed_ids, metrics, strict=True):
                if record["env_id"] != env_id:
                    raise RuntimeError("Terminal record order does not match completed environment IDs")
                record["study_metrics"] = values
        self._study_sums[env_ids] = 0.0
        self._study_count[env_ids] = 0
        self._study_accel_count[env_ids] = 0
        self._study_heading_max[env_ids] = 0.0
        self._study_previous_target_rate[env_ids] = 0.0
        self._study_pending_target_rate[env_ids] = 0.0
        self._study_pending_target_accel[env_ids] = 0.0
        self._study_pending_accel_valid[env_ids] = False
        self._study_previous_joint_velocity[env_ids] = self.robot.data.joint_vel[env_ids][:, self.firmware_joint_ids]

    def verify_domain_parameters(self, env_ids=None):
        result = super().verify_domain_parameters(env_ids)
        ids = self._ids(env_ids)
        active = self._dr_active[ids]
        tau = self.domain_parameters["target_tau_s"][ids]
        delay = self.domain_parameters["delay_control_steps"][ids]
        if not bool(((tau[~active] == 0.0) & (delay[~active] == 0)).all()):
            raise AssertionError("Nominal episodes must have zero added lag and delay")
        if self.cfg.study_variant == "fixed_delay":
            if not torch.allclose(tau[active], torch.full_like(tau[active], 0.04)):
                raise AssertionError("Fixed-delay episode lag is not 0.04 seconds")
            if not bool((delay[active] == 1).all()):
                raise AssertionError("Fixed-delay episode integer delay is not one control step")
        result["study_variant"] = self.cfg.study_variant
        result["control_parameters_verified"] = True
        return result


make_env_cfg = make_study_env_cfg
