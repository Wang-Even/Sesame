#!/usr/bin/env python3
"""Generate, execute and filter references using offline, file or DeepSeek proposals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/supermimic_official_5efbf114/src")]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/".runtime/research_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("WANDB_MODE", "disabled")

from sesame_trajectory_agent import api_client
from sesame_trajectory_agent.schema import PARAMETER_RULES, LEGS, Proposal, parse_proposal, proposal_key, validate_config
from sesame_trajectory_agent.proposer import offline_proposals


def read_json(path):
    def reject_constant(x): raise ValueError(f"Invalid JSON constant: {x}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/trajectory_agent.json")
    p.add_argument("--mode",choices=("offline","file","api"),default="offline")
    p.add_argument("--proposals",type=Path,help='For file mode: {"rounds":[{"candidates":[...]}]}')
    p.add_argument("--output",type=Path)
    p.add_argument("--rounds",type=int)
    p.add_argument("--candidates",type=int)
    p.add_argument("--robust-trials",type=int)
    p.add_argument("--duration",type=float,help="Development override in seconds; creates a different frozen protocol")
    p.add_argument("--feedback-from",type=Path,help="Reuse measured feedback from matching generator/physics/gates; never imports accepted data")
    p.add_argument("--reasoning-effort",choices=("none","low","high"),default="none")
    p.add_argument("--check-config",action="store_true",help="Metadata checks only; no CUDA or API calls")
    a = p.parse_args()
    config = read_json(a.config)
    for arg,key in [(a.rounds,"rounds"),(a.candidates,"candidates_per_round"),
                    (a.robust_trials,"robust_trials"),(a.duration,"duration_s")]:
        if arg is not None: config[key] = arg
    validate_config(config)
    source = Path(config["source"])
    if not source.is_absolute(): source = ROOT/source
    if not (source/"meta.json").is_file(): p.error(f"Missing packed source: {source}")
    if a.check_config:
        print(json.dumps({"config_valid":True,"mode":a.mode,"api_implemented":api_client.API_IMPLEMENTED,
                          "source":str(source),"gpu_or_api_used":False},ensure_ascii=False))
        return
    if a.mode == "api" and not api_client.API_IMPLEMENTED:
        p.error("API is disabled; use --mode offline or enable the API client")
    if a.mode == "api":
        try:
            api_client.completion_url(config["api"])
            api_client.load_api_environment(config["api"])
        except api_client.ApiError as e:
            p.error(str(e))
    supplied = None
    if a.mode == "file":
        if a.proposals is None: p.error("--proposals is required for file mode")
        data = read_json(a.proposals)
        if not isinstance(data,dict) or set(data)!={"rounds"} or not isinstance(data["rounds"],list):
            p.error("Proposal file must contain only a rounds array")
        supplied = data["rounds"]
        if len(supplied) != config["rounds"]: p.error("Proposal-file round count must equal config.rounds")
    if a.output is None: p.error("--output is required")
    out = a.output.resolve()
    if out.exists(): p.error("Choose a new output directory; existing runs are not overwritten")

    # All numerical processing and physics is on CUDA. CLI/schema checks above
    # intentionally work without importing the numerical runtime.
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.4)
    from sesame_research.resources import GpuMemoryMonitor
    from sesame_trajectory_agent.trajectories import ReferenceLibrary, write_json, save_npz, sha256, export_accepted
    from sesame_trajectory_agent.validation import ExecutionEngine
    started = time.monotonic(); out.mkdir(parents=True)
    monitor = GpuMemoryMonitor(out/"gpu_memory.csv")
    nominal = robust = None
    feedback = []; accepted = []; prior_feedback=[]; tested={}
    state = {"status":"initializing","mode":a.mode,"api_called":False,"completed_candidates":0,
             "accepted_count":0,"started_unix":time.time()}
    write_json(out/"status.json",state)
    try:
        library = ReferenceLibrary(source)
        source_summaries = library.summaries()
        source_hashes = {}
        for path in list((ROOT/"sesame_trajectory_agent").glob("*.py")) + [Path(__file__),
                ROOT/"sesame_research/task.py",ROOT/"sesame_research/robot.py",
                ROOT/"assets/isaac/sesame/import_source/sesame.urdf"]:
            rel = path.relative_to(ROOT); saved = out/"sources"/rel
            saved.parent.mkdir(parents=True,exist_ok=True); saved.write_bytes(path.read_bytes())
            source_hashes[str(rel)] = sha256(path)
        prior_metadata=None
        if a.feedback_from:
            prior=a.feedback_from.resolve();old=read_json(prior/"protocol.json")
            for key in ("source","seed","duration_s","fps","target_speed_m_s","robust_trials","gates","physics_dt_s","diagnostic_only_metrics"):
                if old["config"].get(key)!=config.get(key):
                    raise ValueError(f"Feedback protocol differs in {key}")
            for key in ("sesame_trajectory_agent/trajectories.py","sesame_trajectory_agent/validation.py",
                        "sesame_research/task.py","sesame_research/robot.py"):
                if old["source_sha256"].get(key)!=source_hashes.get(key):
                    raise ValueError(f"Feedback generator or simulator differs: {key}")
            if old["input_sha256"]!=library.hashes:raise ValueError("Feedback source data differs")
            for line in (prior/"results.jsonl").read_text().splitlines():
                row=json.loads(line);row["id"]=prior.name+"/"+row["id"]
                prior_feedback.append(row)
                if row.get("proposal"):
                    parsed=parse_proposal(row["proposal"],library.joint_names,len(library.ends))
                    tested[proposal_key(parsed)]=row["id"]
            prior_metadata={"directory":str(prior),"results_sha256":sha256(prior/"results.jsonl"),"feedback_records":len(prior_feedback)}
        protocol = {"config":config,"mode":a.mode,"source_directory":str(source),
                    "reasoning_effort":a.reasoning_effort,"prior_feedback":prior_metadata,
                    "input_sha256":library.hashes,"source_sha256":source_hashes,
                    "official_mjlab_commit":"dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd",
                    "controller":"joint reference -> native DC-motor PD; no policy network is used",
                    "acceptance":"all kinematic checks, nominal trial and every configured randomized trial",
                    "acceptance_scope":"assistant-selected experimental filter, not course requirements; rejection does not establish that a reference cannot be used for training",
                    "generation":"no seed projection or action clipping; optional agent-selected smoothing (default off), monotone Hermite interpolation; fixed +x root-speed goal",
                    "limitations":["Not a proof of global dynamic feasibility or real-robot safety",
                        "Ground contacts only; inherited robot self-collision is disabled",
                        "Metrics sampled at 50 Hz; may miss physics-substep peaks",
                        "Contact foot-origin speed is a slip proxy, not exact contact-point slip",
                        "Randomized trials are search feedback, not independent final tests",
                        "Offline mode is a deterministic heuristic, not an LLM experiment"]}
        write_json(out/"protocol.json",protocol)
        protocol_hash = sha256(out/"protocol.json")
        nominal = ExecutionEngine(source,config)
        if nominal.joints != library.joint_names or nominal.bodies != library.body_names:
            raise ValueError("Source joint/body order does not match the current robot")
        if config["robust_trials"]:
            robust = ExecutionEngine(source,config,config["robust_trials"],randomized=True)
        write_json(out/"simulation_context.json",{"robot":nominal.constraints(),"nominal":nominal.context(),
                   "randomized":robust.context() if robust else None})
        for round_index in range(config["rounds"]):
            monitor.check()
            state.update(status="running",round=round_index)
            write_json(out/"status.json",state)
            request = {"instruction":"Propose NEW gait-reference variations, different from all tested_parameters. Repeating a failed proposal provides no information and will be skipped. Generation has no artificial amplitude or offset limits and never clips joint positions. Use measured failures to explore tempo/amplitude/phase/smoothing/offset changes. Do not change task, gates, controller, data or schema.",
                       "round":round_index,"seed":config["seed"],"candidate_count":config["candidates_per_round"],
                       "task":{k:config[k] for k in ("duration_s","fps","target_speed_m_s")},
                       "gates":config["gates"],"diagnostic_only_metrics":config.get("diagnostic_only_metrics",[]),"robot":nominal.constraints(),"source_clips":len(library.ends),
                       "source_names":library.names,"parameter_rules":PARAMETER_RULES,
                       "source_summaries":source_summaries,
                       "source_durations_s":[(end-begin-1)/50 for begin,end in zip([0]+library.ends[:-1],library.ends)],
                       "legs":LEGS,
                       "mirror_joint_pairs":[["R1","L1"],["R2","L2"],["R3","L3"],["R4","L4"]],
                       "mirror_note":"Neutral positions in each pair sum to pi. Mirrored angular offsets use opposite signs; equal signed offsets are not a symmetric adjustment.",
                       "tested_parameters":[json.loads(key) for key in tested],
                       "response_example":{"candidates":[Proposal().to_dict()]},"feedback":(prior_feedback+feedback)[-16:]}
            write_json(out/"requests"/f"round_{round_index:03d}.json",request)
            if a.mode == "offline": response = offline_proposals(request)
            elif a.mode == "file": response = supplied[round_index]
            else:
                state["api_called"] = True
                response,api_metadata = api_client.call_model_with_metadata(request,config["api"],a.reasoning_effort)
                write_json(out/"responses"/f"round_{round_index:03d}_api.json",api_metadata)
            write_json(out/"responses"/f"round_{round_index:03d}.json",response)
            if not isinstance(response,dict) or set(response)!={"candidates"} or not isinstance(response["candidates"],list):
                raise ValueError("Response must contain only a candidates array")
            if len(response["candidates"]) != config["candidates_per_round"]:
                raise ValueError("Response candidate count differs from the frozen search budget")
            for candidate_index, raw in enumerate(response["candidates"]):
                monitor.check()
                name = f"round_{round_index:03d}_candidate_{candidate_index:03d}"
                folder = out/"candidates"/name; folder.mkdir(parents=True)
                write_json(folder/"proposal.json",raw)
                report = {"id":name,"accepted":False,"validation_complete":False,
                          "protocol_sha256":protocol_hash,"mode":a.mode}
                try:
                    proposal = parse_proposal(raw,library.joint_names,len(library.ends))
                    key=proposal_key(proposal)
                    report["proposal"]=proposal.to_dict()
                    if key in tested:
                        report.update(stage="duplicate_proposal",reasons=["already_tested:"+tested[key]])
                        proposal=None
                    else:
                        tested[key]=name
                        ref, generation = library.generate(proposal,config)
                        report["generation"] = generation
                except ValueError as e:
                    report.update(stage="proposal_rejected",reasons=[str(e)])
                    proposal = None
                if proposal is not None:
                    save_npz(folder/"candidate_reference.npz",ref,fps=50,joint_names=library.joint_names)
                    kin, body = nominal.check_kinematics(ref)
                    report["kinematics"] = kin
                    report["proposal"] = proposal.to_dict()
                    if not kin["passed"]:
                        report.update(stage="kinematics_rejected",reasons=kin["reasons"])
                    else:
                        trials, trace = nominal.execute(ref,body)
                        save_npz(folder/"nominal_execution.npz",trace,fps=50,joint_names=library.joint_names,
                                 body_names=library.body_names,contact_names=nominal.contact.primary_names)
                        report["nominal_trace_sha256"] = sha256(folder/"nominal_execution.npz")
                        report["nominal"] = trials[0]
                        report["nominal_metrics"] = trials[0]["metrics"]
                        # Check every configured trial even when nominal fails,
                        # providing consistent feedback and recording real failures.
                        random_trials = []
                        if robust is not None:
                            random_trials, random_trace = robust.execute(ref,body)
                            save_npz(folder/"randomized_execution.npz",random_trace,fps=50,
                                     joint_names=library.joint_names,body_names=library.body_names,
                                     contact_names=robust.contact.primary_names)
                            report["randomized_trace_sha256"] = sha256(folder/"randomized_execution.npz")
                        report["randomized"] = random_trials
                        passed = trials[0]["passed"] and all(t["passed"] for t in random_trials)
                        report.update(stage="accepted" if passed else "execution_rejected",accepted=passed,
                                      validation_complete=True,reasons=trials[0]["reasons"] +
                                      [f"randomized_trial_{t['trial']}:{reason}" for t in random_trials for reason in t["reasons"]])
                        if passed:
                            accepted.append({"id":name,"source_clip":proposal.source_clip,
                                             "report":str(folder/"report.json"),"trace":str(folder/"nominal_execution.npz")})
                write_json(folder/"report.json",report)
                brief = {k:v for k,v in report.items() if k in ("id","stage","accepted","proposal","reasons","kinematics","nominal_metrics")}
                brief["randomized_trials"] = report.get("randomized",[])
                feedback.append(brief)
                with (out/"results.jsonl").open("a") as f: f.write(json.dumps(brief,ensure_ascii=False,allow_nan=False)+"\n")
                state.update(completed_candidates=len(feedback),accepted_count=len(accepted),current_candidate=name)
                write_json(out/"status.json",state)
                print(json.dumps(brief,ensure_ascii=False,allow_nan=False),flush=True)
        export_accepted(out/"accepted",accepted,library,out/"protocol.json")
        torch.cuda.synchronize()
        state.update(status="complete",finished_unix=time.time(),elapsed_s=time.monotonic()-started,
                     peak_process_memory_mib=monitor.peak_mib,torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                     success_note="Pipeline completion does not imply any candidate passed; inspect accepted_count.")
        write_json(out/"summary.json",state)
        write_json(out/"status.json",state)
        print(json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    except BaseException:
        state.update(status="failed",error=traceback.format_exc(),finished_unix=time.time())
        write_json(out/"status.json",state)
        raise
    finally:
        for engine in (robust,nominal):
            if engine is not None: engine.close()
        monitor.close()


if __name__ == "__main__":
    main()
