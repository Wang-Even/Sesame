"""Sesame configuration on unmodified official mjlab_research APIs."""
from pathlib import Path
import json
import torch
import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from sesame_research.robot import robot_cfg
from sesame_research.reward import configure_env_reward

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "third_party/mjlab_research_dd51a9fc"


def angular_velocity(env):
    return env.scene["robot"].data.root_link_ang_vel_b


def make_cfg(motion_file, num_envs=512, seed=42, sampling_mode="uniform", physics_dt_s=.002):
    if not Path(mjlab.__file__).resolve().is_relative_to(RESEARCH):
        raise RuntimeError(f"Expected official research mjlab, imported {mjlab.__file__}")
    cfg = gencoord_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.sensors = ()
    cfg.scene.env_spacing = 1.
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = .5
    if physics_dt_s not in (.002,.001,.0005,.00025):
        raise ValueError("Unsupported physics timestep; control period remains 0.02 s")
    cfg.sim.mujoco.timestep = physics_dt_s
    cfg.sim.mujoco.iterations = 10
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 192
    cfg.decimation = round(.02/physics_dt_s)
    cfg.episode_length_s = 9.
    cfg.actions["joint_pos"].scale = {".*": .58}
    cfg.observations["actor"].terms["base_ang_vel"].func = angular_velocity
    cfg.observations["actor"].terms["base_ang_vel"].params = {}
    cfg.observations.pop("critic", None)
    command = cfg.commands["motion"]
    assert type(command) is MotionCommandCfg
    command.motion_file = str(motion_file)
    command.anchor_body_name = "base"
    command.body_names = tuple(json.loads((Path(motion_file)/"meta.json").read_text())["body_names"])
    configure_env_reward(cfg)
    command.sampling_mode = "external" if sampling_mode == "return" else sampling_mode
    command.debug_vis = False
    command.pose_range = {"x":(-.005,.005),"y":(-.005,.005),"z":(-.002,.002),
                          "roll":(-.05,.05),"pitch":(-.05,.05),"yaw":(-.1,.1)}
    command.velocity_range = {"x":(-.02,.02),"y":(-.02,.02),"z":(-.01,.01)}
    command.joint_position_range = (-.05,.05)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*foot.*"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
    cfg.events["base_com"].params["ranges"] = {i:(-.002,.002) for i in range(3)}
    cfg.terminations["anchor_pos"].params["threshold"] = .04
    # Native MotionCommand, ForwardKinematics, MotionLoader and motion_end.
    return cfg


def make_env(motion_file, num_envs=512, seed=42, sampling_mode="uniform", physics_dt_s=.002):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    env = ManagerBasedRlEnv(make_cfg(motion_file,num_envs,seed,sampling_mode,physics_dt_s),device="cuda:0")
    return RslRlVecEnvWrapper(env)
