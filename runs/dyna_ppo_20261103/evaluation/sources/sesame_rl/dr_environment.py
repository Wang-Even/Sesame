"""Episode-wise dynamics randomization around the frozen Sesame PPO task.

No reward, termination, action ordering, observation layout, or time step is
changed. Sampling and arithmetic use CUDA. Isaac Sim 4.5's property setters
require host tensors; CPU copies here are property-API transport only.

Robot contact coefficients vary while ground friction stays 0.9. With the
verified ``average`` combine mode on both materials, robot coefficients
0.3..1.5 yield effective friction 0.6..1.2. A fixed 64-value material palette
avoids exhausting PhysX's material limit during repeated resets.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from pxr import PhysxSchema, UsdPhysics, UsdShade
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from .environment import (
    SesameWalkEnv, SesameWalkEnvCfg, SERVO_MAX_SPEED, SERVO_MAX_TORQUE,
)


@configclass
class SesameDREnvCfg(SesameWalkEnvCfg):
    domain_randomization = True
    dr_mass_scale_range = (0.8, 1.2)
    dr_robot_friction_range = (0.3, 1.5)
    dr_material_buckets = 64
    dr_kp_scale_range = (0.8, 1.2)
    dr_kd_scale_range = (0.8, 1.2)
    # Conservative hardware envelope: never exceed the source torque/speed.
    dr_effort_scale_range = (0.8, 1.0)
    dr_speed_scale_range = (0.8, 1.0)
    dr_target_tau_range_s = (0.02, 0.06)
    dr_max_delay_control_steps = 2
    dr_reset_joint_noise = 0.035
    dr_reset_yaw_range_rad = (-math.pi / 36, math.pi / 36)
    dr_reset_xy_velocity_range_m_s = (-0.02, 0.02)
    dr_observation_noise = False
    dr_verify_on_reset = False


def make_env_cfg(
    num_envs: int = 256, seed: int = 42, device: str = "cuda:0",
    episode_length_s: float = 10.0, domain_randomization: bool = True,
) -> SesameDREnvCfg:
    if not device.startswith("cuda"):
        raise ValueError("CUDA is required; CPU fallback is disabled")
    if num_envs < 1 or episode_length_s <= 0:
        raise ValueError("num_envs and episode_length_s must be positive")
    cfg = SesameDREnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.episode_length_s = episode_length_s
    cfg.domain_randomization = bool(domain_randomization)
    return cfg


class SesameDREnv(SesameWalkEnv):
    """The original task with independently sampled per-environment episodes.

    ``set_domain_randomization(False)`` takes effect on each next reset and
    restores immutable nominal property copies, the original target controller,
    original 0.015-rad reset noise, no yaw/velocity perturbation, and no obs noise.
    ``reset(seed=...)`` reseeds a dedicated CUDA domain RNG as well as the base
    task RNG. Snapshots contain no time stamps or episode counters.
    """
    cfg: SesameDREnvCfg

    def __init__(self, cfg: SesameDREnvCfg, render_mode: str | None = None, **kwargs):
        self._dr_ready = False
        self._domain_requested = bool(cfg.domain_randomization)
        if cfg.dr_effort_scale_range[1] > 1.0 or cfg.dr_speed_scale_range[1] > 1.0:
            raise ValueError("DR must respect the source maximum torque and speed")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._domain_rng = torch.Generator(device=self.device)
        self._domain_rng.manual_seed(int(cfg.seed or 0) + 14591)
        self._nominal_reset_joint_noise = float(cfg.reset_joint_noise)
        view = self.robot.root_physx_view
        self._nominal = {
            "masses": view.get_masses().to(self.device).clone(),
            "inertias": view.get_inertias().to(self.device).clone(),
            "materials": view.get_material_properties().to(self.device).clone(),
            "stiffness": view.get_dof_stiffnesses().to(self.device).clone(),
            "damping": view.get_dof_dampings().to(self.device).clone(),
            "effort_limits": view.get_dof_max_forces().to(self.device).clone(),
            "velocity_limits": view.get_dof_max_velocities().to(self.device).clone(),
        }
        self._applied = {key: value.clone() for key, value in self._nominal.items()}
        self._dr_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._friction_palette = torch.linspace(
            *cfg.dr_robot_friction_range, cfg.dr_material_buckets, device=self.device,
        )
        self.domain_parameters = {
            "mass_scale": torch.ones(self.num_envs, device=self.device),
            "robot_friction": self._nominal["materials"][:, 0, 0].clone(),
            "effective_contact_friction": self._nominal["materials"][:, 0, 0].clone(),
            "kp_scale": torch.ones(self.num_envs, device=self.device),
            "kd_scale": torch.ones(self.num_envs, device=self.device),
            "effort_scale": torch.ones(self.num_envs, device=self.device),
            "speed_scale": torch.ones(self.num_envs, device=self.device),
            "target_tau_s": torch.zeros(self.num_envs, device=self.device),
            "delay_control_steps": torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            "initial_yaw_rad": torch.zeros(self.num_envs, device=self.device),
            "initial_xy_velocity_m_s": torch.zeros((self.num_envs, 2), device=self.device),
            "initial_joint_offset_rad": torch.zeros((self.num_envs, 8), device=self.device),
        }
        self._command_targets = self._targets.clone()
        self._target_history = self._targets[:, None, :].repeat(1, cfg.dr_max_delay_control_steps + 1, 1)
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._episode_start_y = torch.zeros(self.num_envs, device=self.device)
        self._max_abs_heading_yaw = torch.zeros(self.num_envs, device=self.device)
        # Noise is in the already-normalized base observation units. Command
        # and previous-action channels remain exact even when noise is enabled.
        self._obs_noise_std = torch.tensor(
            [0.025] * 3 + [0.005] * 3 + [0.003] * 3 + [0.0] * 3
            + [0.002] * 8 + [0.05 / SERVO_MAX_SPEED] * 8 + [0.0] * 8,
            device=self.device,
        )
        self.material_combine_metadata = self._verify_material_combine_modes()
        self._dr_ready = True

    def _verify_material_combine_modes(self) -> dict[str, Any]:
        """Inspect authored material bindings; this is USD metadata, not physics."""
        stage = get_current_stage()
        records = {}
        for kind, paths in (
            ("ground", [r["path"] for r in self.contact_setup_metadata["ground_colliders"]]),
            ("robot", [r["path"] for r in self.contact_setup_metadata["robot_colliders"]]),
        ):
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if not material:
                    raise RuntimeError(f"No bound physical material on {path}")
                material_prim = material.GetPrim()
                mode = PhysxSchema.PhysxMaterialAPI(material_prim).GetFrictionCombineModeAttr().Get()
                api = UsdPhysics.MaterialAPI(material_prim)
                record = {
                    "material_path": str(material_prim.GetPath()), "combine_mode": str(mode),
                    "static_friction": api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": api.GetDynamicFrictionAttr().Get(),
                }
                if mode != "average":
                    raise RuntimeError(f"Effective friction formula requires average combine mode: {record}")
                if kind == "ground" and (
                    not math.isclose(record["static_friction"], self.cfg.ground_friction, rel_tol=1e-6)
                    or not math.isclose(record["dynamic_friction"], self.cfg.ground_friction, rel_tol=1e-6)
                ):
                    raise RuntimeError(f"Unexpected fixed ground coefficients: {record}")
                records[kind] = record
        return records

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None and self._dr_ready:
            self._domain_rng.manual_seed(int(seed) + 14591)
        return super().reset(seed=seed, options=options)

    def set_domain_randomization(self, enabled: bool):
        """Request mode for next resets; ongoing episodes retain their domains."""
        self._domain_requested = bool(enabled)

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return self.robot._ALL_INDICES
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _sample_uniform(self, count: int, bounds: tuple[float, float], width: int | None = None):
        shape = (count,) if width is None else (count, width)
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
            shape, device=self.device, generator=self._domain_rng,
        )

    def _set_episode_parameters(self, env_ids: torch.Tensor):
        count = len(env_ids)
        p = self.domain_parameters
        self._dr_active[env_ids] = self._domain_requested
        if self._domain_requested:
            ranges = {
                "mass_scale": self.cfg.dr_mass_scale_range,
                "kp_scale": self.cfg.dr_kp_scale_range,
                "kd_scale": self.cfg.dr_kd_scale_range,
                "effort_scale": self.cfg.dr_effort_scale_range,
                "speed_scale": self.cfg.dr_speed_scale_range,
                "target_tau_s": self.cfg.dr_target_tau_range_s,
                "initial_yaw_rad": self.cfg.dr_reset_yaw_range_rad,
            }
            for key, bounds in ranges.items():
                p[key][env_ids] = self._sample_uniform(count, bounds)
            bucket_ids = torch.randint(
                self.cfg.dr_material_buckets, (count,), device=self.device, generator=self._domain_rng,
            )
            p["robot_friction"][env_ids] = self._friction_palette[bucket_ids]
            p["delay_control_steps"][env_ids] = torch.randint(
                self.cfg.dr_max_delay_control_steps + 1, (count,), device=self.device, generator=self._domain_rng,
            )
            p["initial_xy_velocity_m_s"][env_ids] = self._sample_uniform(
                count, self.cfg.dr_reset_xy_velocity_range_m_s, width=2,
            )
        else:
            for key in ("mass_scale", "kp_scale", "kd_scale", "effort_scale", "speed_scale"):
                p[key][env_ids] = 1.0
            p["robot_friction"][env_ids] = self._nominal["materials"][env_ids, 0, 0]
            for key in ("target_tau_s", "delay_control_steps", "initial_yaw_rad", "initial_xy_velocity_m_s"):
                p[key][env_ids] = 0.0
        p["effective_contact_friction"][env_ids] = (p["robot_friction"][env_ids] + self.cfg.ground_friction) / 2.0
        self._applied["masses"][env_ids] = self._nominal["masses"][env_ids] * p["mass_scale"][env_ids, None]
        self._applied["inertias"][env_ids] = self._nominal["inertias"][env_ids] * p["mass_scale"][env_ids, None, None]
        self._applied["materials"][env_ids] = self._nominal["materials"][env_ids]
        if self._domain_requested:
            self._applied["materials"][env_ids, :, :2] = p["robot_friction"][env_ids, None, None]
        for property_name, scale in (
            ("stiffness", "kp_scale"), ("damping", "kd_scale"),
            ("effort_limits", "effort_scale"), ("velocity_limits", "speed_scale"),
        ):
            self._applied[property_name][env_ids] = self._nominal[property_name][env_ids] * p[scale][env_ids, None]
        view = self.robot.root_physx_view
        # These setters require FULL-size buffers plus selected row indices.
        # Every draw and multiplication above occurs on CUDA; only transfer.
        host_ids = env_ids.cpu()
        view.set_masses(self._applied["masses"].cpu(), host_ids)
        view.set_inertias(self._applied["inertias"].cpu(), host_ids)
        view.set_material_properties(self._applied["materials"].cpu(), host_ids)
        self.robot.write_joint_stiffness_to_sim(self._applied["stiffness"][env_ids], env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(self._applied["damping"][env_ids], env_ids=env_ids)
        self.robot.write_joint_effort_limit_to_sim(self._applied["effort_limits"][env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_limit_to_sim(self._applied["velocity_limits"][env_ids], env_ids=env_ids)
        # Implicit-actuator effort estimates feed the unchanged torque reward;
        # their cached gains/limits must agree with the actual physics engine.
        for actuator in self.robot.actuators.values():
            indices = actuator.joint_indices
            actuator.stiffness[env_ids] = self._applied["stiffness"][env_ids][:, indices]
            actuator.damping[env_ids] = self._applied["damping"][env_ids][:, indices]
            for attr in ("effort_limit", "effort_limit_sim"):
                getattr(actuator, attr)[env_ids] = self._applied["effort_limits"][env_ids][:, indices]
            for attr in ("velocity_limit", "velocity_limit_sim"):
                getattr(actuator, attr)[env_ids] = self._applied["velocity_limits"][env_ids][:, indices]

    def _pre_physics_step(self, actions: torch.Tensor):
        if not actions.is_cuda:
            raise RuntimeError("Actions must reside on CUDA")
        self.extras = {}
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        desired = (self._stand + self.cfg.action_scale * self._actions).clamp(self._joint_low, self._joint_high)
        # Keep the exact base controller for disabled domains. Only active DR
        # domains apply extra speed variation, communication delay, and lag.
        nominal_delta = SERVO_MAX_SPEED * self.step_dt
        nominal_target = self._targets + (desired - self._targets).clamp(-nominal_delta, nominal_delta)
        max_delta = nominal_delta * self.domain_parameters["speed_scale"][:, None]
        self._command_targets.add_((desired - self._command_targets).clamp(-max_delta, max_delta))
        self._target_history[:, 1:] = self._target_history[:, :-1].clone()
        self._target_history[:, 0] = self._command_targets
        delayed = self._target_history[self._env_indices, self.domain_parameters["delay_control_steps"]]
        tau = self.domain_parameters["target_tau_s"].clamp_min(1e-9)
        alpha = (1.0 - torch.exp(-self.step_dt / tau))[:, None]
        filtered = self._targets + alpha * (delayed - self._targets)
        self._targets.copy_(torch.where(self._dr_active[:, None], filtered, nominal_target))

    def _get_observations(self):
        result = super()._get_observations()
        if self._dr_ready and self.cfg.dr_observation_noise:
            noise = torch.randn(
                result["policy"].shape, device=self.device, generator=self._domain_rng,
            ) * self._obs_noise_std
            result["policy"] = (result["policy"] + noise * self._dr_active[:, None]).clamp(-20.0, 20.0)
        return result

    def _heading_yaw(self) -> torch.Tensor:
        q = self.robot.data.root_quat_w
        return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2].square() + q[:, 3].square()))

    def _get_rewards(self):
        reward = super()._get_rewards()
        if self._dr_ready:
            self._max_abs_heading_yaw = torch.maximum(self._max_abs_heading_yaw, self._heading_yaw().abs())
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not self._dr_ready:
            return super()._reset_idx(env_ids)
        env_ids = self._ids(env_ids)
        completed = env_ids[(self.episode_length_buf[env_ids] > 0) & self.reset_buf[env_ids]]
        old_domains = self._parameter_rows(completed)
        old_heading = self._heading_yaw()[completed].tolist()
        old_max_heading = self._max_abs_heading_yaw[completed].tolist()
        old_lateral = (self.robot.data.root_pos_w[completed, 1] - self._episode_start_y[completed]).tolist()
        # Base reset saves OLD terminal records before writing the new state.
        prior_noise = self.cfg.reset_joint_noise
        self.cfg.reset_joint_noise = self.cfg.dr_reset_joint_noise if self._domain_requested else self._nominal_reset_joint_noise
        try:
            super()._reset_idx(env_ids)
        finally:
            self.cfg.reset_joint_noise = prior_noise
        if len(completed):
            records = list(self.episode_records)[-len(completed):]
            for record, parameters, heading, max_heading, lateral in zip(
                records, old_domains, old_heading, old_max_heading, old_lateral, strict=True,
            ):
                record.update({
                    "domain_parameters": parameters, "final_heading_yaw_rad": heading,
                    "max_abs_heading_yaw_rad": max_heading, "lateral_displacement_m": lateral,
                })
        self._set_episode_parameters(env_ids)
        if self._domain_requested:
            root = self.robot.data.default_root_state[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            yaw = self.domain_parameters["initial_yaw_rad"][env_ids]
            root[:, 3] = torch.cos(yaw / 2.0)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(yaw / 2.0)
            root[:, 7:9] = self.domain_parameters["initial_xy_velocity_m_s"][env_ids]
            self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        self.domain_parameters["initial_joint_offset_rad"][env_ids] = (
            self.robot.data.joint_pos[env_ids][:, self.firmware_joint_ids] - self._stand
        )
        self._episode_start_y[env_ids] = self.robot.data.root_pos_w[env_ids, 1]
        self._max_abs_heading_yaw[env_ids] = self.domain_parameters["initial_yaw_rad"][env_ids].abs()
        self._command_targets[env_ids] = self._targets[env_ids]
        # Refill EVERY delayed slot for only the resetting environments; no
        # first-action leak and no corruption of the other episodes' history.
        self._target_history[env_ids] = self._targets[env_ids, None, :]
        if self.cfg.dr_verify_on_reset:
            self.verify_domain_parameters(env_ids)

    def _parameter_rows(self, env_ids: torch.Tensor) -> list[dict[str, Any]]:
        columns = {key: value[env_ids].tolist() for key, value in self.domain_parameters.items()}
        enabled = self._dr_active[env_ids].tolist()
        return [dict(enabled=enabled[index], **{key: value[index] for key, value in columns.items()}) for index in range(len(env_ids))]

    def _physx_readback(self) -> dict[str, torch.Tensor]:
        view = self.robot.root_physx_view
        return {
            "masses": view.get_masses().to(self.device).clone(),
            "inertias": view.get_inertias().to(self.device).clone(),
            "materials": view.get_material_properties().to(self.device).clone(),
            "stiffness": view.get_dof_stiffnesses().to(self.device).clone(),
            "damping": view.get_dof_dampings().to(self.device).clone(),
            "effort_limits": view.get_dof_max_forces().to(self.device).clone(),
            "velocity_limits": view.get_dof_max_velocities().to(self.device).clone(),
        }

    def verify_domain_parameters(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, Any]:
        """Assert actual PhysX coefficients and cached actuator values match."""
        ids = self._ids(env_ids)
        actual = self._physx_readback()
        errors = {}
        for key, values in actual.items():
            error = (values[ids] - self._applied[key][ids]).abs()
            errors[key] = float(error.max().item()) if error.numel() else 0.0
            if not torch.allclose(values[ids], self._applied[key][ids], rtol=1e-5, atol=1e-8):
                raise AssertionError(f"PhysX {key} differs from sampled values: max abs error {errors[key]}")
        for actuator in self.robot.actuators.values():
            for attr, prop in (("stiffness", "stiffness"), ("damping", "damping"), ("effort_limit", "effort_limits"), ("effort_limit_sim", "effort_limits"), ("velocity_limit", "velocity_limits"), ("velocity_limit_sim", "velocity_limits")):
                expected = self._applied[prop][ids][:, actuator.joint_indices]
                if not torch.allclose(getattr(actuator, attr)[ids], expected):
                    raise AssertionError(f"Actuator cache {attr} differs from PhysX")
        if not bool((actual["effort_limits"][ids] <= SERVO_MAX_TORQUE + 1e-6).all()):
            raise AssertionError("Torque limit exceeds the source hardware envelope")
        if not bool((actual["velocity_limits"][ids] <= SERVO_MAX_SPEED + 1e-5).all()):
            raise AssertionError("Velocity limit exceeds the source hardware envelope")
        result = {
            "passed": True, "env_ids": ids.tolist(), "max_abs_error": errors,
            "cpu_usage": "PhysX property transport only; sampling, comparison, physics and policy on CUDA",
            "material_combine": self.material_combine_metadata,
        }
        self.last_domain_verification = result
        return result

    def domain_snapshot(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, Any]:
        """Deterministic JSON metadata for paired runs, including actual readback."""
        ids = self._ids(env_ids)
        actual = self._physx_readback()
        return {
            "env_ids": ids.tolist(), "enabled": self._dr_active[ids].tolist(),
            "parameters": {key: value[ids].tolist() for key, value in self.domain_parameters.items()},
            "physx_readback": {key: value[ids].tolist() for key, value in actual.items()},
            "effective_contact_friction_readback": ((actual["materials"][ids, :, :2] + self.cfg.ground_friction) / 2.0).tolist(),
            "fixed_ground_friction": self.cfg.ground_friction,
            "material_combine_mode": "average", "material_palette_size": self.cfg.dr_material_buckets,
            "observation_noise_configured": bool(self.cfg.dr_observation_noise),
        }

    get_domain_snapshot = domain_snapshot


make_dr_env_cfg = make_env_cfg
