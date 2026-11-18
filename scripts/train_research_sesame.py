"""Official SuperMimic and official research mjlab; local Sesame task only."""
import os
os.environ.setdefault("WANDB_MODE","disabled")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","4")
os.environ.setdefault("OMP_NUM_THREADS","4")
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"third_party/supermimic_official_5efbf114/src")]
import argparse
import dataclasses
import hashlib
import json
import time
import traceback
import torch
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg, Ratio
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from sesame_research.task import make_env
from sesame_research.ppo import PpoSimRunner
from sesame_research.evaluation import Evaluator
from sesame_research.resources import GpuMemoryMonitor
from sesame_research.runner import SesameSuperMimicRunner
from sesame_research.reward import REWARD_VERSION


def write_json(path, data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+"\n")
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=["supermimic","ppo"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--envs",type=int,default=512)
    p.add_argument("--steps",type=int,default=2097152)
    p.add_argument("--width",type=int,default=512)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--no-critic",action="store_true")
    p.add_argument("--wm-batch",type=int,default=512)
    p.add_argument("--wm-batches",type=int,default=10)
    p.add_argument("--eval-every",type=int,default=262144)
    p.add_argument("--eval-steps",type=int,default=450)
    p.add_argument("--eval-after-first-update",action="store_true",
                   help="Record an actual validation immediately after the first update and use it as post-startup time zero")
    p.add_argument("--motion-file",type=Path,default=ROOT/"datasets/supermimic_research_sesame_20261112/train")
    p.add_argument("--validation-file",type=Path,default=ROOT/"datasets/supermimic_research_sesame_20261112/validation")
    p.add_argument("--test-file",type=Path,default=ROOT/"datasets/supermimic_research_sesame_20261112/test")
    p.add_argument("--diagnostic-only-metrics",nargs="*",default=[],choices=["velocity_rmse_m_s","contact_foot_speed_rms_m_s"])
    p.add_argument("--sampling-mode",choices=["uniform","return"],default="uniform")
    p.add_argument("--physics-dt",type=float,choices=[.002,.001,.0005,.00025],default=.002)
    p.add_argument("--replay-ratio",type=float,default=16.)
    a=p.parse_args()
    if a.output.exists() and (a.output/"protocol.json").exists():
        raise FileExistsError("Choose a fresh run directory; existing runs are never overwritten")
    if len({path.resolve() for path in (a.motion_file,a.validation_file,a.test_file)})!=3:
        raise ValueError("Training, validation and final test must be separate directories")
    a.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.50)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(a.seed)
    monitor=GpuMemoryMonitor(a.output/"gpu_memory.csv")
    policy_cls=BpttPolicyLearnerCfg if a.no_critic else ActorCriticPolicyLearnerCfg
    cfg=SuperMimicRunnerCfg(buffer_size=1000,sampling_mode=a.sampling_mode,actor_obs="proprio",history_length=10,
        dynamics_hidden_dims=(a.width,)*3,actor_hidden_dims=(a.width,a.width//2,a.width//4),
        critic_hidden_dims=(256,128),replay_ratio=a.replay_ratio,
        wm=WorldModelLearnerCfg(batch_size=a.wm_batch,num_batches=a.wm_batches,horizon=8),
        policy=policy_cls(batch_size=a.batch,horizon=a.horizon,gamma=.99))
    sources={}
    for pattern in ("sesame_research/*.py","scripts/*research_sesame*.py"):
        for path in ROOT.glob(pattern):
            rel=path.relative_to(ROOT)
            dst=a.output/"sources"/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            content=path.read_bytes(); dst.write_bytes(content)
            sources[str(rel)]=hashlib.sha256(content).hexdigest()
    import importlib.metadata
    dataset_hours=json.loads((a.motion_file/"meta.json").read_text())["duration_h"]
    protocol={"args":vars(a),"official_commit":"5efbf114e11f725c31d83e217be987c13c0e4ce2",
        "mjlab_commit":"dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd",
        "backend":"unmodified HybridRobotics/mjlab_research; native motion loader, FK, reset and termination",
        "package_versions":{k:importlib.metadata.version(k) for k in ["mjlab","torch","mujoco","mujoco-warp","warp-lang","rsl-rl-lib"]},
        "runner_config":dataclasses.asdict(cfg),"sources":sources,"dataset_hours":dataset_hours,
        "target_normalized_return":.82,"return_ceiling_per_second":5.,"gpu_count":1,
        "time_metric":"training sampling + updates including first compilation; excludes evaluation and env setup",
        "post_startup_time_metric":"training_s minus the first completed update timestamp; evaluation and checkpoint time excluded",
        "evaluation_schedule":"initial, optional first completed update, global eval_every step boundaries, and budget end",
        "shared_gpu":True,"seed_count":1,"normalization":"fixed full horizon; early failure remainder zero",
        "actor_observation":"30-dimensional proprio x 10 history + 220-dimensional motion goal",
        "action_dim":8,"evaluation":"fixed held-out clips from start, no sensor noise, nominal dynamics",
        "diagnostic_only_metrics":a.diagnostic_only_metrics,
        "ppo":"rsl_rl PPO; same MultiEncoderAE and GELU/LayerNorm critic and reset sampling mode as SuperMimic",
        "ppo_config":{"num_steps_per_env":32,"epochs":5,"minibatches":4,"clip_param":.2,
                      "gamma":.99,"lambda":.95,"learning_rate":1e-3,"entropy_coef":.005,
                      "desired_kl":.01,"initial_action_std":.1,"max_grad_norm":1.},
        "reward_version":REWARD_VERSION,
        "reward_note":"Task reward excludes foot links from position/orientation/velocity tracking and removes joint-limit cost; native simulator and imagined rollouts use matching rewards"}
    write_json(a.output/"protocol.json",protocol)
    env=val=test=None
    status={"status":"initializing","pid":os.getpid(),"method":a.method,"started_unix":time.time()}
    write_json(a.output/"status.json",status)
    try:
        env=make_env(a.motion_file,a.envs,a.seed,a.sampling_mode,a.physics_dt)
        write_json(a.output/"environment_config.json",dataclasses.asdict(env.unwrapped.cfg))
        runner_cls=SesameSuperMimicRunner if a.method=="supermimic" else PpoSimRunner
        runner=runner_cls(env,cfg,"Sesame-Official-Research",device="cuda:0")
        runner._seed_sim_history()
        actor=runner.policy_learner.actor if a.method=="supermimic" else runner.actor.core
        val=Evaluator(a.validation_file,81042,a.physics_dt,diagnostic_only_metrics=a.diagnostic_only_metrics)
        test=Evaluator(a.test_file,91042,a.physics_dt,diagnostic_only_metrics=a.diagnostic_only_metrics)
        params={"actor":sum(p.numel() for p in actor.parameters()),
                "world_model":sum(p.numel() for p in runner._dynamics.parameters()) if a.method=="supermimic" else 0}
        critic=getattr(runner.policy_learner,"critic",None) if a.method=="supermimic" else runner.critic
        params["critic"]=sum(p.numel() for p in critic.parameters()) if critic is not None else 0
        write_json(a.output/"parameters.json",params)
        # Track the initial learned parameters separately from running statistics.
        initial_hash=hashlib.sha256()
        for name,param in actor.named_parameters():
            initial_hash.update(name.encode())
            initial_hash.update(param.detach().cpu().contiguous().numpy().tobytes())
        write_json(a.output/"initial_actor.json",{"parameter_sha256":initial_hash.hexdigest(),
            "seed":a.seed,"parameters":params["actor"],"policy_gamma":cfg.policy.gamma})
        total=updates=imagined=0
        training_s=0.; evaluation_s=0.; checkpoint_s=0.; startup_training_s=None
        next_eval=0; best=-float("inf"); time_to_target=None; evaluation_count=0
        first_update_evaluated=False
        ratio=Ratio(a.envs/a.wm_batches/a.wm_batch/8*a.replay_ratio)
        curve=open(a.output/"curve.jsonl","a",buffering=1)
        lossfile=open(a.output/"updates.jsonl","a",buffering=1)
        def checkpoint(name):
            nonlocal checkpoint_s
            checkpoint_tick=time.monotonic()
            bundle={"actor":actor.state_dict(),"steps":total,"cfg":dataclasses.asdict(cfg),"method":a.method}
            if a.method=="supermimic":
                bundle["world_model"]=runner._wm.bundle()
                critic=getattr(runner.policy_learner,"critic",None)
                if critic is not None: bundle["critic"]=critic.state_dict()
            else:
                bundle["critic"]=runner.critic.state_dict()
                bundle["ppo_actor"]=runner.actor.state_dict()
            torch.save(bundle,a.output/name)
            checkpoint_s+=time.monotonic()-checkpoint_tick
        def evaluate(evaluator, **metric_args):
            nonlocal evaluation_s
            torch.cuda.synchronize()
            evaluation_tick=time.monotonic()
            metrics=evaluator.run(actor,runner._representation,runner._sim_context,
                                  cfg.history_length,a.eval_steps,**metric_args)
            torch.cuda.synchronize()
            elapsed=time.monotonic()-evaluation_tick
            evaluation_s+=elapsed
            return metrics,elapsed
        while True:
            first_update_due=(a.eval_after_first_update and startup_training_s is not None
                              and not first_update_evaluated)
            if total>=next_eval or total>=a.steps or first_update_due:
                trigger=("initial" if total==0 else "first_update" if first_update_due else
                         "budget_end" if total>=a.steps else "periodic")
                metric_args={} if a.method=="ppo" else dict(dynamics=runner._dynamics,
                    loss_weights=runner.world_model_learner.loss_group_weights)
                metrics,eval_elapsed=evaluate(val,**metric_args)
                evaluation_count+=1
                row={"real_steps":total,"imagined_steps":imagined,"training_s":training_s,
                    "startup_training_s":startup_training_s,
                    "post_startup_training_s":None if startup_training_s is None else training_s-startup_training_s,
                    "evaluation_s":eval_elapsed,"cumulative_evaluation_s":evaluation_s,
                    "evaluation_index":evaluation_count,"evaluation_trigger":trigger,
                    "gpu_hours":training_s/3600,"wall_clock_h":training_s/3600,"end_to_end_s":time.monotonic()-started,
                    "dataset_hours":dataset_hours,"peak_process_memory_mib":monitor.peak_mib,**metrics}
                curve.write(json.dumps(row,allow_nan=False)+"\n")
                print("EVALUATION "+json.dumps(row,allow_nan=False),flush=True)
                if metrics["normalized_return"]>best:
                    best=metrics["normalized_return"]; checkpoint("best.pt")
                if metrics["normalized_return"]>=.82 and time_to_target is None:
                    time_to_target={"training_s":training_s,"real_steps":total}
                checkpoint("latest.pt")
                if first_update_due:first_update_evaluated=True
                next_eval=((total//a.eval_every)+1)*a.eval_every
                status.update(status="training",real_steps=total,training_s=training_s,
                    startup_training_s=startup_training_s,
                    post_startup_training_s=None if startup_training_s is None else training_s-startup_training_s,
                    cumulative_evaluation_s=evaluation_s,checkpoint_s=checkpoint_s,
                    evaluation_count=evaluation_count,latest_evaluation=row)
                write_json(a.output/"status.json",status)
            if total>=a.steps: break
            monitor.check()
            tick=time.monotonic()
            if a.method=="supermimic":
                runner.step(); total+=a.envs
                logs={}
                for _ in range(ratio(total//a.envs-1)):
                    if runner.replay_buffer.num_transitions<8: continue
                    data=(runner.replay_buffer.sample(8,a.wm_batch) for _ in range(a.wm_batches))
                    logs=runner.world_model_learner.update(data) | runner.policy_learner.update()
                    updates+=1; imagined+=a.batch*a.horizon
            else:
                logs=runner.train_iteration(); total+=a.envs*32; updates+=1
            torch.cuda.synchronize()
            training_s+=time.monotonic()-tick
            if logs:
                if startup_training_s is None:
                    startup_training_s=training_s
                    write_json(a.output/"startup.json",dict(training_s=training_s,
                        real_steps=total,updates=updates,
                        definition="First completed parameter update; includes data accumulation and initial compilation, not a pure compilation measurement"))
                values={k:float(v) for k,v in logs.items()}
                if not all(__import__('math').isfinite(v) for v in values.values()):
                    raise RuntimeError(f"Nonfinite update: {values}")
                lossfile.write(json.dumps({"real_steps":total,"training_s":training_s,
                    "post_startup_training_s":training_s-startup_training_s,**values})+"\n")
        final_metrics,_=evaluate(test)
        checkpoint("final.pt")
        best_bundle=torch.load(a.output/"best.pt",map_location="cuda:0",weights_only=False)
        actor.load_state_dict(best_bundle["actor"])
        best_metrics,_=evaluate(test)
        status.update(status="complete",real_steps=total,imagined_steps=imagined,training_s=training_s,
            startup_training_s=startup_training_s,post_startup_training_s=training_s-startup_training_s,
            cumulative_evaluation_s=evaluation_s,checkpoint_s=checkpoint_s,evaluation_count=evaluation_count,
            finished_unix=time.time(),
            end_to_end_s=time.monotonic()-started,time_to_target=time_to_target,final_test=final_metrics,
            best_validation_checkpoint_test=best_metrics,best_steps=best_bundle["steps"],
            peak_process_memory_mib=monitor.peak_mib,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        sampler=getattr(runner,"_sampler",None)
        if sampler is not None:
            probs=runner.motion_cmd._external_probs
            status["reset_sampler"]={"mode":"return","observed_samples":float(sampler.count.sum()),
                "probability_min":float(probs.min()),"probability_max":float(probs.max())}
        write_json(a.output/"status.json",status)
    except BaseException:
        status.update(status="failed",error=traceback.format_exc(),peak_process_memory_mib=monitor.peak_mib)
        write_json(a.output/"status.json",status)
        raise
    finally:
        monitor.close()
        for e in (val,test,env):
            if e is not None: e.close()


if __name__=="__main__": main()
