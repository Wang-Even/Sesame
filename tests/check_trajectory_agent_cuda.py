"""CUDA interpolation, FK and export-contract checks. Temporary fixtures are
synthetic test inputs, NOT certified motions; they are deleted after this test.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"third_party/supermimic_official_5efbf114/src")]
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".runtime/research_cache"))
os.environ.setdefault("OMP_NUM_THREADS","4")
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from sesame_trajectory_agent.decisions import METRIC_GATES
from sesame_trajectory_agent.schema import Proposal
from sesame_trajectory_agent.trajectories import ReferenceLibrary, hermite, export_accepted, save_npz, sha256, write_json
from sesame_trajectory_agent.validation import ExecutionEngine


def expect_rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Invalid export was accepted")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU numerical tests")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.4)
    config = json.loads((ROOT/"configs/trajectory_agent.json").read_text())
    config.update(duration_s=.2,robust_trials=1)
    source = ROOT/config["source"]
    library = ReferenceLibrary(source)
    engine = ExecutionEngine(source,config)
    try:
        # Independent central-difference check on a cubic between sample knots.
        q = torch.tensor([[0.],[.1],[.16]],device="cuda:0",dtype=torch.float64)
        dq = torch.tensor([[4.],[3.],[2.]],device="cuda:0",dtype=torch.float64)
        times = torch.tensor([[.007],[.027]],device="cuda:0",dtype=torch.float64)
        cols = torch.tensor([[0]],device="cuda:0")
        _,analytic = hermite(q,dq,times,50,cols)
        plus,_ = hermite(q,dq,times+1e-6,50,cols)
        minus,_ = hermite(q,dq,times-1e-6,50,cols)
        torch.testing.assert_close(analytic,(plus-minus)/2e-6,atol=1e-6,rtol=1e-6)
        ref,_ = library.generate(Proposal(amplitude=.9,smoothing_window_s=.12),config)
        assert all(t.is_cuda and bool(torch.isfinite(t).all()) for t in ref.values())
        kin,body = engine.check_kinematics(ref)
        assert kin["passed"],kin
        assert body.is_cuda and body.shape == (11,len(library.body_names),3)

        raw,_ = library.generate(Proposal(start_time_s=.2),config)
        torch.testing.assert_close(raw["joint_pos"][0],library.data["joint_pos"][10],atol=1e-6,rtol=0)
        excess,details = library.generate(Proposal(joint_offsets_rad={"R1":4.}),config)
        assert not details["generation_clipping"]
        assert bool((excess["joint_pos"][:,0]>engine.robot.data.joint_pos_limits[0,0,1]+1).all())
        checked,_=engine.check_kinematics(excess)
        assert "reference_joint_limit" not in checked["reasons"]
        assert "joint_limit_violation_max_rad" not in checked["metrics"]
        equivalent={k:v.clone() for k,v in ref.items()}
        equivalent["joint_pos"]+=2*torch.pi
        checked,_=engine.check_kinematics(equivalent)
        assert checked["passed"],checked

        with tempfile.TemporaryDirectory(prefix="agent_contract_",dir=ROOT/".runtime") as temp:
            temp = Path(temp)
            protocol = temp/"protocol.json"
            write_json(protocol,{"config":config,"note":"Synthetic format-test fixture; not a validation result"})
            trace = temp/"nominal_execution.npz"
            random_trace = temp/"randomized_execution.npz"
            save_npz(trace,{k:v[:,None] for k,v in ref.items()})
            save_npz(random_trace,{k:v[:,None] for k,v in ref.items()})
            metrics = {k:0. for k in METRIC_GATES}
            metrics.update(finite=True,completed=True,fell=False,steps=10)
            # New user-approved protocol must export even if these diagnostic
            # values exceed their historical thresholds. Other gates still apply.
            metrics.update(velocity_rmse_m_s=99.,contact_foot_speed_rms_m_s=99.)
            nominal = {"trial":0,"randomized":False,"passed":True,"reasons":[],"metrics":metrics}
            report = {"accepted":True,"validation_complete":True,"stage":"accepted","reasons":[],
                      "kinematics":{"passed":True},"protocol_sha256":sha256(protocol),
                      "nominal_trace_sha256":sha256(trace),"randomized_trace_sha256":sha256(random_trace),
                      "nominal":nominal,"randomized":[copy.deepcopy(nominal) | {"randomized":True}]}
            report_path = temp/"report.json"
            items = [{"id":"TEST_FIXTURE_NOT_FOR_TRAINING","source_clip":0,"trace":str(trace),"report":str(report_path)}]
            def export(destination):
                export_accepted(temp/destination,items,library,protocol)
            variants = [report | {"validation_complete":False},report | {"protocol_sha256":"changed"},
                        report | {"randomized":[]},report | {"nominal_trace_sha256":"changed"},
                        report | {"randomized_trace_sha256":"changed"}]
            bad = copy.deepcopy(report)
            bad["randomized"][0]["metrics"]["heading_max_deg"] = 21.
            variants.append(bad)
            bad = copy.deepcopy(report)
            bad["nominal"]["metrics"]["mean_speed_error_m_s"] = .026
            variants.append(bad)
            for i,invalid in enumerate(variants):
                write_json(report_path,invalid)
                expect_rejected(lambda:export(f"rejected_{i}"))
                assert not (temp/f"rejected_{i}"/"train").exists()
            write_json(report_path,report)
            export("format_fixture")
            loaded = MotionLoader(str(temp/"format_fixture/train"),engine.fk,device="cuda:0")
            torch.testing.assert_close(loaded.joint_pos,ref["joint_pos"])
            states = loaded.body_state(torch.arange(11,device="cuda:0"))
            torch.testing.assert_close(states[0],body)
            legacy=copy.deepcopy(config);legacy.pop("diagnostic_only_metrics",None)
            write_json(protocol,{"config":legacy,"note":"Archived strict protocol fixture"})
            strict=copy.deepcopy(report);strict["protocol_sha256"]=sha256(protocol)
            write_json(report_path,strict)
            expect_rejected(lambda:export("legacy_rejected"))
            assert not (temp/"legacy_rejected/train").exists()
            export_accepted(temp/"empty",[],library,protocol)
            assert not (temp/"empty/train").exists()
            assert json.loads((temp/"empty/manifest.json").read_text())["accepted_count"] == 0
        torch.cuda.synchronize()
        print(json.dumps({"cuda_checks":"passed","export_rejection_cases":len(variants),
                          "native_motion_loader":"passed","fixture_retained":False}))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
