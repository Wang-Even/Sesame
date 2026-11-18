#!/usr/bin/env python3
"""MimicAgent workflow adapted to Sesame: plan, code, check, diagnose, revise.

Native physical execution remains mandatory for training admission. No candidate
is admitted on an LLM verdict, best-effort status, or kinematic preview alone.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"third_party/supermimic_official_5efbf114/src")]
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".runtime/research_cache"))
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("WANDB_MODE","disabled")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/trajectory_agent_unclipped.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidates",type=int,default=3)
    p.add_argument("--repairs",type=int,default=3,help="Initial program plus up to N diagnostic revisions")
    p.add_argument("--continue-on-error",action="store_true",help="Record an API/plan/diagnosis error and continue with the next candidate; abort after 3 consecutive API failures")
    p.add_argument("--check-config",action="store_true")
    p.add_argument("--authored-dir",type=Path,help="Read current chat agent's plan.json and motion.py from candidate_NNN/READY directories; no external API calls")
    p.add_argument("--input-timeout",type=float,default=1200.,help="Maximum seconds to wait for each authored candidate")
    p.add_argument("--accepted-target",type=int,help="Stop once this many distinct references pass every required check")
    p.add_argument("--context-file",type=Path,help="Optional recorded engineering hints for the planner/coder; cannot modify task, gates or simulator")
    a=p.parse_args()
    from sesame_trajectory_agent.schema import validate_config
    from sesame_trajectory_agent.api_client import call_json_role,ApiError
    from sesame_trajectory_agent import mimic_prompts as prompts
    from sesame_trajectory_agent.motion_program import parse_program
    config=validate_config(json.loads(a.config.read_text()))
    engineering_context=json.loads(a.context_file.read_text()) if a.context_file else None
    if engineering_context is not None and not isinstance(engineering_context,dict):
        p.error("Engineering context must be a JSON object")
    if not 1<=a.candidates<=(1000 if a.authored_dir else 100) or not 0<=a.repairs<=10:p.error("Invalid candidate/repair budget")
    if a.accepted_target is not None and not 1<=a.accepted_target<=a.candidates:p.error("Invalid accepted target")
    if a.authored_dir and a.repairs!=0:p.error("Authored mode counts each submitted revision as a candidate; use --repairs 0")
    if a.input_timeout<=0:p.error("Input timeout must be positive")
    source=ROOT/config["source"]
    if source.name in ("validation","test"):p.error("Source must be training metadata")
    if a.check_config:
        print(json.dumps({"config_valid":True,"model":config["api"]["model"],"candidates":a.candidates,"repairs":a.repairs,
            "admission":"kinematic + nominal + all randomized trials","gpu_or_api_used":False}));return
    out=a.output.resolve()
    if out.exists():p.error("Output already exists; use a new directory")
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4)
    from sesame_trajectory_agent.validation import ExecutionEngine
    from sesame_trajectory_agent.mimic_motion import MotionProgramRuntime
    from sesame_trajectory_agent.trajectories import write_json,save_npz,sha256,export_accepted
    from sesame_research.resources import GpuMemoryMonitor
    from types import SimpleNamespace
    out.mkdir(parents=True)
    started=time.monotonic();engine=robust=None;accepted=[];history=[];seen=set();reference_seen={};accepted_hashes=set()
    state={"status":"initializing","started_unix":time.time(),"method":"MimicAgent workflow adapted to Sesame",
           "accepted_count":0,"attempts":0,"api_calls":0,"kinematic_pass_count":0,
           "candidate_errors":0,"api_errors":0,"consecutive_api_errors":0,
           "candidate_budget":a.candidates,"completed_candidates":0,"accepted_target":a.accepted_target}
    monitor=GpuMemoryMonitor(out/"gpu_memory.csv")
    write_json(out/"status.json",state)
    try:
        source_files=list((ROOT/"sesame_trajectory_agent").glob("*.py"))+[Path(__file__),ROOT/"sesame_research/robot.py",ROOT/"sesame_research/task.py"]
        hashes={}
        for path in source_files:
            rel=path.relative_to(ROOT); dest=out/"sources"/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(path.read_bytes());hashes[str(rel)]=sha256(path)
        protocol={"config":config,"generator":"mimicagent_sesame","model":"current Codex conversation agent" if a.authored_dir else config["api"]["model"],
            "candidates":a.candidates,"repairs":a.repairs,"accepted_target":a.accepted_target,"source_sha256":hashes,
            "continue_on_candidate_error":a.continue_on_error,
            "paper":"https://arxiv.org/abs/2609.24145",
            "public_workflow":"https://github.com/Luckykantnayak/mimicagent-humnaoid/blob/main/agent/pipeline.py",
            "provenance":"Local Sesame adaptation of paper and public humanoid workflow; quadruped official code not released",
            "acceptance":"all kinematic checks + native PD nominal trial + every configured randomized trial",
            "acceptance_scope":"existing experimental dynamic gates, not course requirements or MimicAgent paper success criteria",
            "execution_feedback":config.get("execution_feedback"),
            "mean_speed_measurement":config.get("mean_speed_measurement","control_samples"),
            "generation":"agent-authored executable expressions on CUDA, no joint clipping, root adjustment or seed resampling",
            "changes_from_paper":[("current Codex conversation agent; no external API" if a.authored_dir else "deepseek-flash instead of paper Claude models"),"Sesame eight-joint CUDA adapter",
                "0.035 m minimum height and twice actual standing height; Go2's 0.1 m would reject all Sesame standing poses",
                "fixed task semantics and native PD dynamics verification before admission at user's request",
                "only successful actual executed traces exported for SuperMimic"],
            "limitations":["PD failure does not imply untrackable by learned RL policy","self collision disabled in inherited model",
                "50 Hz sampled validation; contact velocity is a slip proxy","search trials are not held-out evaluations"]}
        if a.authored_dir:
            protocol["authored_input"]={"directory":str(a.authored_dir.resolve()),"api_calls":0,
                "budget_unit":"one submitted program version, including each revision",
                "comparison_limit":"Author has seen prior DeepSeek results and may run recorded CUDA geometry diagnostics; not a blind or equal-budget model comparison"}
        if engineering_context is not None:
            protocol["engineering_context"]={"source":str(a.context_file.resolve()),"sha256":sha256(a.context_file),
                "content":engineering_context,"scope":"prompt context only; no runtime limits or admission changes"}
            write_json(out/"engineering_context.json",engineering_context)
        write_json(out/"protocol.json",protocol);protocol_hash=sha256(out/"protocol.json")
        engine=ExecutionEngine(source,config)
        if config["robust_trials"]:robust=ExecutionEngine(source,config,config["robust_trials"],True)
        runtime=MotionProgramRuntime(engine,config)
        library=SimpleNamespace(joint_names=engine.joints,body_names=engine.bodies)
        context={"robot":runtime.metadata(),"task":{"description":"Sesame walks straight forward on level ground at 0.1 m/s, upright, without yaw drift or falling.",
            "duration_s":config["duration_s"],"target_speed_m_s":config["target_speed_m_s"],"fps":config["fps"]},
            "gates":config["gates"],"diagnostic_only_metrics":config.get("diagnostic_only_metrics",[]),"program_api":prompts.API}
        if engineering_context is not None:
            context["engineering_context"]=engineering_context
        write_json(out/"context.json",context)
        write_json(out/"simulation_context.json",{"nominal":engine.context(),"randomized":robust.context() if robust else None})
        def call(role,instruction,request,folder):
            write_json(folder/(role+"_request.json"),request)
            if a.authored_dir:
                item=a.authored_dir.resolve()/f"candidate_{state['candidate']:03d}"
                if role=="planner":
                    state.update(status="waiting_for_authored_program");write_json(out/"status.json",state)
                    deadline=time.monotonic()+a.input_timeout
                    while not (item/"READY").is_file():
                        if time.monotonic()>deadline:raise RuntimeError("Timed out waiting for authored program: "+str(item))
                        monitor.check();time.sleep(1.)
                    result=json.loads((item/"plan.json").read_text())
                    if (item/"design.json").exists():
                        write_json(folder/"design.json",json.loads((item/"design.json").read_text()))
                elif role=="coder":
                    result={"code":(item/"motion.py").read_text(),"explanation":"Authored by the current chat agent; see candidate plan and design provenance"}
                else:raise RuntimeError("Authored mode cannot invoke API or automatic repair")
                write_json(folder/(role+"_response.json"),result)
                return result
            state["api_calls"]+=1
            try:
                result,meta=call_json_role(role,instruction,request,config["api"])
            except ApiError as e:
                state["api_errors"]+=1;state["consecutive_api_errors"]+=1
                write_json(folder/(role+"_error.json"),{"role":role,"error":str(e),"unix_s":time.time()})
                if state["consecutive_api_errors"]>=3 or any(f"HTTP {code}" in str(e) for code in (401,402,403,404)):
                    raise RuntimeError("API cannot continue: "+str(e)) from None
                raise
            state["consecutive_api_errors"]=0
            write_json(folder/(role+"_response.json"),result);write_json(folder/(role+"_api.json"),meta)
            return result
        def candidate_error(folder,stage,error):
            if not a.continue_on_error:raise error
            state["candidate_errors"]+=1
            write_json(folder/"candidate_error.json",{"stage":stage,"error":str(error),"unix_s":time.time(),"accepted":False})
            write_json(out/"status.json",state)
            print(json.dumps({"candidate_error":str(folder.relative_to(out)),"stage":stage,"error":str(error)}),flush=True)
        for candidate in range(a.candidates):
            monitor.check();root=out/"candidates"/f"candidate_{candidate:03d}"
            root.mkdir(parents=True)
            state.update(status="planning",candidate=candidate);write_json(out/"status.json",state)
            try:
                plan=prompts.validate_plan(call("planner",prompts.PLANNER,context|{"candidate":candidate,"previous_attempts":history[-4:]},root))
            except (ApiError,ValueError,TypeError,KeyError) as e:
                candidate_error(root,"planner",e)
                state["completed_candidates"]=candidate+1
                write_json(out/"status.json",state)
                continue
            write_json(root/"plan.json",plan)
            code="";diagnosis=None;previous_report=None
            for attempt in range(a.repairs+1):
                monitor.check();name=f"candidate_{candidate:03d}_attempt_{attempt:02d}"
                folder=root/f"attempt_{attempt:02d}";folder.mkdir()
                state.update(status="coding",attempt=attempt);write_json(out/"status.json",state)
                request=context|{"candidate_index":candidate,"plan":plan,"previous_code":code,"diagnosis":diagnosis,"previous_report":previous_report}
                try:
                    reply=call("coder" if attempt==0 else "repair",prompts.CODER if attempt==0 else prompts.REPAIR,request,folder)
                except ApiError as e:
                    candidate_error(folder,"coder",e)
                    break
                report={"id":name,"generator":"mimicagent_sesame","accepted":False,"validation_complete":False,
                        "protocol_sha256":protocol_hash,"plan_sha256":sha256(root/"plan.json")}
                try:
                    if set(reply)!={"code","explanation"} or not isinstance(reply["explanation"],str):raise ValueError("Invalid code response schema")
                    report["generation_explanation"]=reply["explanation"]
                    code=reply["code"];parse_program(code)
                    (folder/"motion.py").write_text(code)
                    digest=sha256(folder/"motion.py");report["motion_program_sha256"]=digest
                    if digest in seen:raise ValueError("Duplicate motion code; revise structure based on feedback")
                    seen.add(digest)
                    ref,contacts,generation=runtime.generate(code)
                    save_npz(folder/"candidate_reference.npz",ref,fps=50,joint_names=engine.joints,declared_contacts=contacts.cpu().numpy())
                    report["candidate_reference_sha256"]=sha256(folder/"candidate_reference.npz")
                    report["generation"]=generation
                    trajectory_hash=hashlib.sha256(b"".join(ref[k].contiguous().cpu().numpy().tobytes() for k in sorted(ref))).hexdigest()
                    report["trajectory_sha256"]=trajectory_hash
                    # Contact declarations are part of kinematic validation. A
                    # corrected declaration must be checked even when q/base
                    # arrays are unchanged. Training admission dedups motion only.
                    contact_hash=hashlib.sha256(contacts.contiguous().cpu().numpy().tobytes()).hexdigest()
                    validation_key=(trajectory_hash,contact_hash)
                    report["declared_contacts_sha256"]=contact_hash
                    if validation_key in reference_seen:
                        raise ValueError("Identical trajectory AND contacts already tested: "+reference_seen[validation_key]+"; changing comments or unused expressions is not a revision")
                    reference_seen[validation_key]=name
                    kin,body=runtime.check(ref,contacts,generation);report["kinematics"]=kin
                    if not kin["passed"]:report.update(stage="kinematics_rejected",reasons=kin["reasons"])
                    else:
                        state["kinematic_pass_count"]+=1
                        state.update(status="validating");write_json(out/"status.json",state)
                        trials,trace=engine.execute(ref,body)
                        save_npz(folder/"nominal_execution.npz",trace,fps=50,joint_names=engine.joints,body_names=engine.bodies)
                        report["nominal_trace_sha256"]=sha256(folder/"nominal_execution.npz")
                        report["nominal"]=trials[0];report["nominal_metrics"]=trials[0]["metrics"]
                        random_trials=[]
                        if robust:
                            random_trials,random_trace=robust.execute(ref,body)
                            save_npz(folder/"randomized_execution.npz",random_trace,fps=50,joint_names=engine.joints,body_names=engine.bodies)
                            report["randomized_trace_sha256"]=sha256(folder/"randomized_execution.npz")
                        report["randomized"]=random_trials
                        passed=trials[0]["passed"] and all(t["passed"] for t in random_trials)
                        reasons=trials[0]["reasons"]+[f"randomized_{t['trial']}:{r}" for t in random_trials for r in t["reasons"]]
                        if passed and trajectory_hash in accepted_hashes:
                            passed=False;reasons.append("duplicate_accepted_reference")
                        if passed:accepted_hashes.add(trajectory_hash)
                        report.update(accepted=passed,validation_complete=True,stage="accepted" if passed else "execution_rejected",reasons=reasons)
                except (ValueError,TypeError,KeyError,IndexError,SyntaxError,RuntimeError) as e:
                    if isinstance(e,torch.cuda.OutOfMemoryError):raise
                    report.update(stage="program_rejected",reasons=[type(e).__name__+": "+str(e)[:1500]])
                write_json(folder/"report.json",report)
                brief={k:v for k,v in report.items() if k in ("id","stage","accepted","reasons","kinematics","nominal_metrics","randomized","generation_explanation")}
                history.append(brief);previous_report=brief
                with (out/"results.jsonl").open("a") as f:f.write(json.dumps(brief,ensure_ascii=False,allow_nan=False)+"\n")
                state["attempts"]+=1
                if report["accepted"]:
                    accepted.append({"id":name,"source_clip":"agent_motion_program","report":str(folder/"report.json"),"trace":str(folder/"nominal_execution.npz")})
                    write_json(out/"accepted_registry.json",{"protocol_sha256":protocol_hash,"entries":accepted,
                               "note":"Validated entries saved incrementally; final export revalidates every report and hash"})
                state["accepted_count"]=len(accepted);write_json(out/"status.json",state)
                print(json.dumps({"id":name,"stage":report["stage"],"accepted_count":len(accepted),"reasons":report["reasons"]}),flush=True)
                if report["accepted"]:break
                if attempt<a.repairs:
                    state.update(status="diagnosing");write_json(out/"status.json",state)
                    try:
                        diagnosis=call("diagnosis",prompts.DIAGNOSIS,context|{"plan":plan,"code":code,"report":brief,"version_history":history[-4:]},folder)
                        if set(diagnosis)!={"failures","causes","corrections"} or not all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in diagnosis.values()):
                            raise ValueError("Invalid diagnostic response schema")
                    except (ApiError,ValueError,TypeError) as e:
                        candidate_error(folder,"diagnosis",e)
                        break
            # Only independently validated executions are considered for export.
            state["completed_candidates"]=candidate+1
            write_json(out/"status.json",state)
            if a.accepted_target is not None and len(accepted)>=a.accepted_target:
                break
        export_accepted(out/"accepted",accepted,library,out/"protocol.json")
        state.update(status="complete",elapsed_s=time.monotonic()-started,finished_unix=time.time(),
                     target_reached=a.accepted_target is None or len(accepted)>=a.accepted_target,
                     peak_process_memory_mib=monitor.peak_mib,torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                     note="Completion is not success; only accepted_count counts admitted training references")
        write_json(out/"summary.json",state);write_json(out/"status.json",state)
        print(json.dumps(state),flush=True)
    except BaseException:
        state.update(status="failed",error=traceback.format_exc(),finished_unix=time.time())
        write_json(out/"status.json",state);raise
    finally:
        if robust:robust.close()
        if engine:engine.close()
        monitor.close()


if __name__=="__main__":main()
