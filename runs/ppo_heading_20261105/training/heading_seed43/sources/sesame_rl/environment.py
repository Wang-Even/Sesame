"""GPU-only flat-ground Sesame locomotion baseline for Isaac Lab.

Import this module only after ``isaaclab.app.AppLauncher`` has started the app.
Use ``cfg = make_env_cfg(num_envs=128, seed=42)`` and ``env = SesameWalkEnv(cfg)``.
The native DirectRLEnv API returns ``obs, reward, terminated, truncated, extras``;
``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper(env, clip_actions=1.0)`` adapts it to RSL-RL.

Actions and joint observations use firmware order R1,R2,L1,L2,R4,R3,L3,L4.
The 36 observations are body linear velocity (3), body angular velocity (3),
projected gravity (3), world forward velocity command expressed in body frame
(3), joint deviation from stand (8), normalized joint velocity (8), and the
latest clipped action (8). Simulator joint feedback is available to this
baseline; these observations are not a deployment-ready hardware interface.

Only a local reference task is defined here. Masses, servos, and meshes come
from sesame-ml reference estimates; no hardware calibration is implied.
"""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any

import torch
from pxr import PhysxSchema, Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse


ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_JOINT_NAMES = ("R1", "R2", "L1", "L2", "R4", "R3", "L3", "L4")
STAND_ANGLES_RAD = tuple(math.radians(x) for x in (135, 45, 45, 135, 0, 180, 0, 180))
JOINT_LIMITS_RAD = tuple(
    (math.radians(low), math.radians(high))
    for low, high in ((45, 180), (0, 135), (0, 135), (45, 180), (0, 180), (0, 180), (0, 180), (0, 180))
)
SERVO_MAX_SPEED = 10.471975512
SERVO_MAX_TORQUE = 0.18


@configclass
class SesameWalkEnvCfg(DirectRLEnvCfg):
    """Explicit baseline parameters; source servo properties stay unchanged."""

    seed = 42
    decimation = 10
    episode_length_s = 10.0
    action_space = 8
    observation_space = 36
    state_space = 0
    is_finite_horizon = False
    wait_for_textures = False
    ui_window_class_type = None

    action_scale = 0.58
    command_forward_m_s = 0.10
    initial_height = 0.049
    reset_joint_noise = 0.015
    termination_height = 0.023
    termination_upright = 0.5
    boundary_distance = 3.0
    contact_offset = 0.001
    ground_friction = 0.9

    # Tracking at rest is exp(-1), avoiding a broad almost-maximal stand reward.
    tracking_variance = 0.01
    tracking_reward_scale = 1.5
    forward_reward_scale = 1.0
    upright_reward_scale = 0.35
    lateral_penalty_scale = 0.10
    yaw_penalty_scale = 0.03
    roll_pitch_rate_penalty_scale = 0.01
    vertical_velocity_penalty_scale = 0.02
    torque_penalty_scale = 0.02
    action_rate_penalty_scale = 0.08
    pose_penalty_scale = 0.10
    termination_penalty = 2.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False,
    )
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        device="cuda:0", dt=0.002, render_interval=10,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.9, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="min",
        ),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT / "assets/isaac/sesame/sesame.usd"),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0, enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1, sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.049),
            joint_pos=dict(zip(FIRMWARE_JOINT_NAMES, STAND_ANGLES_RAD)),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "reference_servos": ImplicitActuatorCfg(
                joint_names_expr=list(FIRMWARE_JOINT_NAMES),
                stiffness=5.0, damping=0.0025,
                effort_limit_sim=SERVO_MAX_TORQUE, velocity_limit_sim=SERVO_MAX_SPEED,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def make_env_cfg(
    num_envs: int = 128,
    seed: int = 42,
    device: str = "cuda:0",
    episode_length_s: float = 10.0,
) -> SesameWalkEnvCfg:
    """Create an independently mutable configuration without allocating tensors."""
    if not device.startswith("cuda"):
        raise ValueError("Sesame training and simulation require CUDA; CPU fallback is disabled")
    if num_envs < 1 or episode_length_s <= 0:
        raise ValueError("num_envs and episode_length_s must be positive")
    cfg = SesameWalkEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.episode_length_s = episode_length_s
    return cfg


class SesameWalkEnv(DirectRLEnv):
    cfg: SesameWalkEnvCfg

    def __init__(self, cfg: SesameWalkEnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.sim.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU simulation or numerical fallback is permitted")
        if not Path(cfg.robot.spawn.usd_path).is_file():
            raise FileNotFoundError(cfg.robot.spawn.usd_path)
        if cfg.command_forward_m_s <= 0:
            raise ValueError("This forward-walking baseline requires a positive forward command")
        super().__init__(cfg, render_mode, **kwargs)
        self.firmware_joint_names = FIRMWARE_JOINT_NAMES
        self.firmware_joint_ids, resolved = self.robot.find_joints(
            FIRMWARE_JOINT_NAMES, preserve_order=True,
        )
        if tuple(resolved) != FIRMWARE_JOINT_NAMES or self.robot.num_joints != 8:
            raise RuntimeError(f"Unexpected Sesame articulation joints: {self.robot.joint_names}")
        if self.robot.is_fixed_base:
            raise RuntimeError("Walking requires the imported floating base")
        self.joint_order = tuple(self.robot.joint_names)
        self._stand = torch.tensor(STAND_ANGLES_RAD, device=self.device)
        joint_limits = torch.tensor(JOINT_LIMITS_RAD, device=self.device)
        self._joint_low, self._joint_high = joint_limits[:, 0], joint_limits[:, 1]
        self._actions = torch.zeros((self.num_envs, 8), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._targets = self._stand.repeat(self.num_envs, 1)
        self._commands_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._commands_w[:, 0] = cfg.command_forward_m_s
        self._episode_start_x = torch.zeros(self.num_envs, device=self.device)
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._fell = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._outside = torch.zeros_like(self._fell)
        self._invalid = torch.zeros_like(self._fell)
        self.episode_records: deque[dict[str, Any]] = deque(maxlen=20000)
        self._reward_sums: dict[str, torch.Tensor] = {}

    def _setup_scene(self):
        # A local cuboid avoids any network/Nucleus dependency. Its top is z=0.
        size = 2.5 * math.ceil(math.sqrt(self.cfg.scene.num_envs)) + 12.0
        ground_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.ground_friction, dynamic_friction=self.cfg.ground_friction,
            restitution=0.0, friction_combine_mode="average", restitution_combine_mode="min",
        )
        ground = sim_utils.CuboidCfg(
            size=(size, size, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=self.cfg.contact_offset, rest_offset=0.0,
            ),
            physics_material=ground_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.27, 0.30)),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
        robot_cfg = self.cfg.robot.copy()
        robot_cfg.init_state.pos = (0.0, 0.0, self.cfg.initial_height)
        # Imported collision groups are USD instances. Applying nested collision
        # properties during spawning silently skips their instance proxies.
        # Deinstance only collision groups, then author and verify each collider.
        robot_cfg.spawn.collision_props = None
        self.robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self.robot
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.ground_friction, dynamic_friction=self.cfg.ground_friction,
            restitution=0.0, friction_combine_mode="average", restitution_combine_mode="min",
        )
        material.func("/World/RobotContactMaterial", material)
        stage = get_current_stage()
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        deinstanced = []
        while True:
            instance_paths = [
                prim.GetPath() for prim in Usd.PrimRange(robot_root)
                if prim.IsInstanceable() and "collision" in str(prim.GetPath()).lower()
            ]
            if not instance_paths:
                break
            for path in instance_paths:
                stage.GetPrimAtPath(path).SetInstanceable(False)
                deinstanced.append(str(path))
        collider_records = []
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if prim.IsInstanceProxy():
                raise RuntimeError(f"Collision prim is still an uneditable instance proxy: {prim.GetPath()}")
            collider = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collider.CreateContactOffsetAttr().Set(self.cfg.contact_offset)
            collider.CreateRestOffsetAttr().Set(0.0)
            sim_utils.bind_physics_material(prim.GetPath(), "/World/RobotContactMaterial")
            actual_contact = collider.GetContactOffsetAttr().Get()
            actual_rest = collider.GetRestOffsetAttr().Get()
            bindings = prim.GetRelationship("material:binding:physics").GetTargets()
            if not math.isclose(actual_contact, self.cfg.contact_offset, rel_tol=1e-6) or actual_rest != 0.0:
                raise RuntimeError(f"Contact settings failed to author on {prim.GetPath()}")
            if [str(path) for path in bindings] != ["/World/RobotContactMaterial"]:
                raise RuntimeError(f"Physics material failed to bind on {prim.GetPath()}: {bindings}")
            collider_records.append({
                "path": str(prim.GetPath()), "contact_offset": actual_contact,
                "rest_offset": actual_rest, "physics_material": str(bindings[0]),
            })
        if not collider_records:
            raise RuntimeError("No authored robot colliders found before cloning")
        ground_records = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Ground")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collider = PhysxSchema.PhysxCollisionAPI(prim)
                offset, rest = collider.GetContactOffsetAttr().Get(), collider.GetRestOffsetAttr().Get()
                if offset is None or not math.isclose(offset, self.cfg.contact_offset, rel_tol=1e-6) or rest != 0.0:
                    raise RuntimeError(f"Ground contact settings missing on {prim.GetPath()}")
                ground_records.append({"path": str(prim.GetPath()), "contact_offset": offset, "rest_offset": rest})
        if not ground_records:
            raise RuntimeError("No ground collider found")
        self.contact_setup_metadata = {
            "deinstanced_collision_groups": deinstanced, "robot_colliders": collider_records,
            "ground_colliders": ground_records, "visual_instances_preserved": True,
        }
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor):
        if not actions.is_cuda:
            raise RuntimeError("Policy actions must reside on CUDA")
        # Clear old terminal summaries so RSL-RL cannot count them repeatedly.
        self.extras = {}
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        desired = (self._stand + self.cfg.action_scale * self._actions).clamp(
            self._joint_low, self._joint_high,
        )
        # Bound target changes as well as the solver's joint velocity limit.
        max_delta = SERVO_MAX_SPEED * self.step_dt
        self._targets.add_((desired - self._targets).clamp(-max_delta, max_delta))

    def _apply_action(self):
        self.robot.set_joint_position_target(self._targets, joint_ids=self.firmware_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        data = self.robot.data
        command_b = quat_apply_inverse(data.root_quat_w, self._commands_w)
        obs = torch.cat((
            data.root_lin_vel_b * 5.0,
            data.root_ang_vel_b * 0.25,
            data.projected_gravity_b,
            command_b * 5.0,
            data.joint_pos[:, self.firmware_joint_ids] - self._stand,
            data.joint_vel[:, self.firmware_joint_ids] / SERVO_MAX_SPEED,
            self._actions,
        ), dim=-1)
        return {"policy": obs.clamp(-20.0, 20.0)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        local_pos = data.root_pos_w - self.scene.env_origins
        upright = -data.projected_gravity_b[:, 2]
        self._invalid = ~(
            torch.isfinite(data.root_state_w).all(dim=1)
            & torch.isfinite(data.joint_pos).all(dim=1)
            & torch.isfinite(data.joint_vel).all(dim=1)
        )
        self._fell = (local_pos[:, 2] < self.cfg.termination_height) | (upright < self.cfg.termination_upright)
        self._outside = torch.any(local_pos[:, :2].abs() > self.cfg.boundary_distance, dim=1)
        terminated = self._fell | self._outside | self._invalid
        timeout = self.episode_length_buf >= self.max_episode_length
        return terminated, timeout & ~terminated

    def _get_rewards(self) -> torch.Tensor:
        data = self.robot.data
        velocity = data.root_lin_vel_w
        upright = (-data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        tracking_error = (velocity[:, 0] - self.cfg.command_forward_m_s).square() + velocity[:, 1].square()
        rates = {
            "tracking": self.cfg.tracking_reward_scale * torch.exp(-tracking_error / self.cfg.tracking_variance) * upright.square(),
            "forward": self.cfg.forward_reward_scale * (velocity[:, 0] / self.cfg.command_forward_m_s).clamp(-1.0, 1.0) * upright.square(),
            "upright": self.cfg.upright_reward_scale * upright,
            "lateral": -self.cfg.lateral_penalty_scale * (velocity[:, 1] / 0.10).square(),
            "yaw": -self.cfg.yaw_penalty_scale * data.root_ang_vel_b[:, 2].square(),
            "roll_pitch_rate": -self.cfg.roll_pitch_rate_penalty_scale * data.root_ang_vel_b[:, :2].square().sum(dim=1),
            "vertical_velocity": -self.cfg.vertical_velocity_penalty_scale * (velocity[:, 2] / 0.10).square(),
            "torque": -self.cfg.torque_penalty_scale * (data.applied_torque / SERVO_MAX_TORQUE).square().mean(dim=1),
            "action_rate": -self.cfg.action_rate_penalty_scale * (self._actions - self._previous_actions).square().mean(dim=1),
            "pose": -self.cfg.pose_penalty_scale * (data.joint_pos[:, self.firmware_joint_ids] - self._stand).square().mean(dim=1),
        }
        # Continuous rates are time-integrated; failure is a one-off event cost.
        components = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * self.step_dt for key, value in rates.items()}
        components["termination"] = -self.cfg.termination_penalty * self.reset_terminated.float()
        reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
        self._episode_return.add_(reward)
        for key, value in components.items():
            if key not in self._reward_sums:
                self._reward_sums[key] = torch.zeros(self.num_envs, device=self.device)
            self._reward_sums[key].add_(value)
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # The parent calls us before returned observations are computed. Save
        # terminal state first; post-reset root positions must not enter eval.
        completed = env_ids[(self.episode_length_buf[env_ids] > 0) & self.reset_buf[env_ids]]
        if completed.numel() > 0:
            distance = self.robot.data.root_pos_w[completed, 0] - self._episode_start_x[completed]
            duration = self.episode_length_buf[completed].float() * self.step_dt
            speed = distance / duration.clamp_min(self.step_dt)
            rows = torch.stack((
                completed.float(), distance, duration, speed, self._episode_return[completed],
                self._fell[completed].float(), self.reset_time_outs[completed].float(),
                self._outside[completed].float(), self._invalid[completed].float(),
            ), dim=1).tolist()
            for row in rows:
                record = {
                    "env_id": int(row[0]), "distance_x_m": row[1], "duration_s": row[2],
                    "avg_speed_m_s": row[3], "return": row[4], "fell": bool(row[5]),
                    "timeout": bool(row[6]), "out_of_bounds": bool(row[7]),
                    "invalid_state": bool(row[8]),
                }
                if record["env_id"] == 0:
                    record["terminal_root_state_w"] = self.robot.data.root_state_w[0].tolist()
                    record["terminal_joint_pos"] = self.robot.data.joint_pos[0].tolist()
                self.episode_records.append(record)
            logs = {
                "Episode/distance_x_m": distance.mean(), "Episode/duration_s": duration.mean(),
                "Episode/avg_speed_m_s": speed.mean(), "Episode/return": self._episode_return[completed].mean(),
                "Episode/fall_fraction": self._fell[completed].float().mean(),
                "Episode/timeout_fraction": self.reset_time_outs[completed].float().mean(),
            }
            for key, value in self._reward_sums.items():
                logs[f"RewardRate/{key}"] = (value[completed] / duration.clamp_min(self.step_dt)).mean()
            self.extras["log"] = logs
        super()._reset_idx(env_ids)
        count = len(env_ids)
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        stand = self._stand.expand(count, -1)
        noise = (torch.rand((count, 8), device=self.device) * 2.0 - 1.0) * self.cfg.reset_joint_noise
        reset_targets = (stand + noise).clamp(self._joint_low, self._joint_high)
        joint_pos[:, self.firmware_joint_ids] = reset_targets
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._targets[env_ids] = reset_targets
        self.robot.set_joint_position_target(reset_targets, joint_ids=self.firmware_joint_ids, env_ids=env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._episode_start_x[env_ids] = default_root[:, 0]
        self._episode_return[env_ids] = 0.0
        for value in self._reward_sums.values():
            value[env_ids] = 0.0


# Short aliases kept for the runner's simple programmatic construction.
SesameEnv = SesameWalkEnv
SesameEnvCfg = SesameWalkEnvCfg
