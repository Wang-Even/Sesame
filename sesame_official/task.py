"""Task configuration; official reward kept identical in real/learned dynamics."""
from pathlib import Path
import dataclasses
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from sesame_official.robot import robot_cfg
from sesame_official.motion import SesameMotionCommandCfg, motion_end


def angular_velocity(env):
    return env.scene["robot"].data.root_link_ang_vel_b


def make_cfg(motion_file, num_envs=512, seed=42):
    cfg = gencoord_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.sensors = ()
    cfg.scene.env_spacing = 1.
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = .5
    cfg.sim.mujoco.timestep = .002
    cfg.sim.mujoco.iterations = 10
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 192
    cfg.decimation = 10
    cfg.episode_length_s = 9.
    # A name map materializes a tensor in public mjlab, as official helpers expect.
    cfg.actions["joint_pos"].scale = {".*": .58}
    cfg.observations["actor"].terms["base_ang_vel"].func = angular_velocity
    cfg.observations["actor"].terms["base_ang_vel"].params = {}
    # Official runners build their critic from full_state + goal + context.
    cfg.observations.pop("critic", None)
    command = cfg.commands["motion"]
    params = {f.name: getattr(command, f.name) for f in dataclasses.fields(command)}
    params.update(motion_file=str(motion_file), anchor_body_name="base", body_names=(),
                  sampling_mode="uniform", debug_vis=False,
                  pose_range={"x":(-.005,.005),"y":(-.005,.005),"z":(-.002,.002),
                              "roll":(-.05,.05),"pitch":(-.05,.05),"yaw":(-.1,.1)},
                  velocity_range={"x":(-.02,.02),"y":(-.02,.02),"z":(-.01,.01)},
                  joint_position_range=(-.05,.05))
    # Load only string metadata here; all kinematics and simulation use CUDA.
    import numpy as np
    with np.load(motion_file, allow_pickle=False) as data:
        params["body_names"] = tuple(data["body_names"].tolist())
    cfg.commands["motion"] = SesameMotionCommandCfg(**params)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*foot.*"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
    cfg.events["base_com"].params["ranges"] = {i:(-.002,.002) for i in range(3)}
    cfg.terminations["anchor_pos"].params["threshold"] = .04
    cfg.terminations["motion_end"] = TerminationTermCfg(func=motion_end, time_out=True)
    return cfg


def make_env(motion_file, num_envs=512, seed=42):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    env = ManagerBasedRlEnv(make_cfg(motion_file, num_envs, seed), device="cuda:0")
    return RslRlVecEnvWrapper(env)
