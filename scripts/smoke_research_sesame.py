"""CUDA-only compatibility and end-to-end official learner validation."""
import os
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/supermimic_official_5efbf114/src")]
import argparse
import json
import time
import torch
from sesame_research.task import make_env
from sesame_research.resources import GpuMemoryMonitor
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from sesame_research.runner import SesameSuperMimicRunner as SuperMimicRunner
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from mjlab.utils.lab_api.math import matrix_from_quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--wm-batches", type=int, default=1)
    ap.add_argument("--wm-batch", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    monitor = GpuMemoryMonitor(a.output / "gpu_memory.csv")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    # Reserve headroom for Warp and the other user's job; this is not a memory filler.
    torch.cuda.set_per_process_memory_fraction(.50)
    start = time.monotonic()
    ap_marker = "native research mjlab"
    env = make_env(ROOT / "datasets/supermimic_research_sesame_20261112/train", a.envs)
    raw = env.unwrapped
    c = raw.command_manager.get_term("motion")
    robot = raw.scene["robot"]
    from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
    from mjlab.utils.forward_kinematics import ForwardKinematics
    assert type(c) is MotionCommand and type(c.motion) is MotionLoader
    assert type(c.motion.fk) is ForwardKinematics
    def check_fk():
        d = robot.data
        pos, quat, vel, ang = c.motion.fk(d.root_link_pos_w, d.root_link_quat_w,
            d.joint_pos, d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_vel)
        pairs = ((pos, c.robot_body_pos_w), (matrix_from_quat(quat), matrix_from_quat(c.robot_body_quat_w)),
                 (vel, c.robot_body_lin_vel_w), (ang, c.robot_body_ang_vel_w))
        errors = [float((x-y).abs().max()) for x,y in pairs]
        if max(errors) > 1e-3:
            raise AssertionError(f"FK vs GPU MuJoCo mismatch: {errors}")
        return errors
    env.reset()
    fk_reset = check_fk()
    for _ in range(10):
        env.step(torch.randn(a.envs,8,device="cuda:0") * .1)
    fk_step = check_fk()
    cfg = SuperMimicRunnerCfg(
        buffer_size=64, sampling_mode="uniform", actor_obs="proprio", history_length=10,
        dynamics_hidden_dims=(a.width,)*3,
        actor_hidden_dims=(a.width,a.width//2,a.width//4), critic_hidden_dims=(256,128),
        wm=WorldModelLearnerCfg(num_batches=a.wm_batches, batch_size=a.wm_batch or a.batch, horizon=8),
        policy=ActorCriticPolicyLearnerCfg(batch_size=a.batch, horizon=a.horizon),
    )
    runner = SuperMimicRunner(env, cfg, task_id="Sesame-Official-Compat", device="cuda:0")
    runner._seed_sim_history()
    # Algebraic equality of real and differentiable reward on the same snapshot.
    state = env.get_observations()["full_state"]
    rep, aux = runner._representation.represent(state), runner._representation.extract_aux_state(state)
    m = c.motion
    wm_reward = runner._representation.compute_wm_reward_from_reference(rep, aux,
        runner._representation.joint_state(rep), raw.action_manager.action,
        raw.action_manager.prev_action, *m.body_state(c.time_steps))
    real_reward = raw.reward_manager.compute(dt=raw.step_dt)
    reward_error = (wm_reward-real_reward).abs().max().item()
    if reward_error > 1e-4:
        raise AssertionError(f"Real / world-model reward mismatch: {reward_error}")
    # Removed targets and nominal joint ranges cannot affect imagined rewards.
    targets=[x.clone() for x in m.body_state(c.time_steps)]
    foot_ids=[i for i,n in enumerate(c.cfg.body_names) if n.endswith("_foot")]
    targets[0][:,foot_ids]+=100.
    targets[1][:,foot_ids]=targets[1][:,foot_ids].roll(1,-1)
    targets[2][:,foot_ids]+=100.
    targets[3][:,foot_ids]+=100.
    altered=runner._representation.compute_wm_reward_from_reference(rep,aux,
        runner._representation.joint_state(rep)+100.,raw.action_manager.action,
        raw.action_manager.prev_action,*targets)
    torch.testing.assert_close(altered,wm_reward,atol=0.,rtol=0.)
    assert "joint_limit" not in raw.cfg.rewards
    assert all(not set(term.params.get("body_names",())).intersection(c.cfg.body_names[i] for i in foot_ids)
               for term in raw.cfg.rewards.values())
    clip = torch.searchsorted(m.motion_ends, c.time_steps, right=True)
    c.set_time_steps(slice(None), m.motion_ends[clip] - 1)
    env.step(torch.zeros(a.envs,8,device="cuda:0"))
    if not raw.termination_manager.get_term("motion_end").all():
        raise AssertionError("Clip end not marked as timeout")
    env.reset(); runner._seed_sim_history()
    for _ in range(16):
        runner.step()
    logs = []
    for i in range(a.updates):
        t = time.monotonic()
        wm_logs = runner.world_model_learner.update([
            runner.replay_buffer.sample(8, a.wm_batch or a.batch) for _ in range(a.wm_batches)])
        policy_logs = runner.policy_learner.update()
        torch.cuda.synchronize()
        row = {"update":i, "elapsed_s":time.monotonic()-t,
               **{k:float(v) for k,v in (wm_logs | policy_logs).items()}}
        if not all(torch.isfinite(torch.tensor(v,device="cuda:0")) for v in row.values()):
            raise AssertionError("Nonfinite training log")
        logs.append(row)
        monitor.check()
        print(json.dumps(row), flush=True)
    # Compare official FK to the previous reference geometry, without modifying data.
    import numpy as np
    with np.load(ROOT/"datasets/supermimic_official_sesame_20261111/train.npz") as f:
        old_pos=torch.as_tensor(f["body_pos_w"],device="cuda:0")
    native_pos=m.body_state(torch.arange(m.time_step_total,device="cuda:0"))[0]
    reference_error=float((old_pos-native_pos).abs().max())
    if reference_error>1e-4:raise AssertionError(f"Official reference FK differs: {reference_error}")
    result = {"status":"passed", "elapsed_s":time.monotonic()-start, "pid":os.getpid(),
        "backend":ap_marker,"reference_fk_max_abs":reference_error,
        "fk_reset_max_abs":fk_reset, "fk_step_max_abs":fk_step, "reward_max_abs":reward_error,
        "actor_params":sum(p.numel() for p in runner.policy_learner.actor.parameters()),
        "world_params":sum(p.numel() for p in runner._dynamics.parameters()),
        "torch_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "torch_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30, "updates":logs,
        "peak_process_memory_mib":monitor.peak_mib,
        "config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    env.close()
    monitor.close()


if __name__ == "__main__":
    main()
