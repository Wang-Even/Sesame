"""Metadata tests only: no numerical runtime or model computation on CPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sesame_trajectory_agent.decisions import METRIC_GATES, decide
from sesame_trajectory_agent.schema import Proposal, parse_proposal, proposal_key, validate_config


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/trajectory_agent.json").read_text())
        self.joints = ["R1","R3","R2","R4","L1","L3","L2","L4"]

    def test_agent_cannot_change_task_or_gate(self):
        for raw in ({"gates":{}},{"target_speed_m_s":0.},{"controller":"teleport"}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_proposal(raw,self.joints,24)

    def test_invalid_numbers_and_joint_names(self):
        for raw in ({"tempo":float("nan")},{"tempo":float("inf")},{"amplitude":True},
                    {"source_clip":True},{"source_clip":24}, {"tempo":0.},
                    {"phase_offsets_s":{"tail":0}}, {"joint_offsets_rad":{"R9":.01}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_proposal(raw,self.joints,24)
        parse_proposal(Proposal().to_dict(),self.joints,24)
        validate_config(self.config)

    def test_generation_parameters_have_no_artificial_caps(self):
        p=parse_proposal({"amplitude":3.,"height_offset_m":.1,"phase_offsets_s":{"right_front":.4},
                          "joint_offsets_rad":{"R1":2.},"smoothing_window_s":0.},self.joints,24)
        self.assertEqual(p.amplitude,3.)
        self.assertEqual(p.joint_offsets_rad["R1"],2.)
        self.assertEqual(p.smoothing_window_s,0.)

    def test_duplicate_detection_ignores_explanation_and_explicit_zeros(self):
        a=parse_proposal({"rationale":"first","phase_offsets_s":{}},self.joints,24)
        b=parse_proposal({"rationale":"second","joint_offsets_rad":{"R1":0}},self.joints,24)
        c=parse_proposal({"amplitude":1.01},self.joints,24)
        self.assertEqual(proposal_key(a),proposal_key(b))
        self.assertNotEqual(proposal_key(a),proposal_key(c))

    def test_failures_cannot_pass(self):
        metrics = {k:0. for k in METRIC_GATES}
        metrics.update(finite=True,completed=True,fell=False)
        self.assertEqual(decide(metrics,self.config["gates"]),[])
        changes = [{"finite":False},{"completed":False},{"fell":True},
                   {"heading_max_deg":21.},{"contact_foot_speed_rms_m_s":None},
                   {"root_tracking_rmse_m":float("nan")},
                   {"velocity_rmse_m_s":-1.}]
        for change in changes:
            with self.subTest(change=change):
                self.assertTrue(decide(metrics | change,self.config["gates"]))
        self.assertTrue(decide({},self.config["gates"]))

    def test_removed_checks_do_not_gate_admission(self):
        metrics={k:0. for k in METRIC_GATES}
        metrics.update(finite=True,completed=True,fell=False)
        self.assertEqual(decide(metrics,self.config["gates"]),[])
        self.assertEqual(decide(metrics|{"foot_tracking_rmse_m":99.,"joint_limit_violation_max_rad":99.},self.config["gates"]),[])
        self.assertNotIn("foot_tracking_rmse_max_m",self.config["gates"])
        self.assertNotIn("joint_limit_violation_max_rad",self.config["gates"])

    def test_user_exempted_metrics_are_diagnostic_only(self):
        metrics={k:0. for k in METRIC_GATES}
        metrics.update(finite=True,completed=True,fell=False,
                       velocity_rmse_m_s=99.,contact_foot_speed_rms_m_s=None)
        ignored=["velocity_rmse_m_s","contact_foot_speed_rms_m_s"]
        self.assertEqual(decide(metrics,self.config["gates"],ignored),[])
        # Archived configs without the new field retain their original decisions.
        self.assertEqual(len(decide(metrics,self.config["gates"])),2)
        for change in [{"fell":True},{"heading_max_deg":21.},{"mean_speed_error_m_s":.026}]:
            with self.subTest(change=change):
                self.assertTrue(decide(metrics|change,self.config["gates"],ignored))

    def test_diagnostic_config_cannot_disable_other_requirements(self):
        for ignored in [["heading_max_deg"],["velocity_rmse_m_s"]*2,"velocity_rmse_m_s",None]:
            with self.subTest(ignored=ignored),self.assertRaises(ValueError):
                validate_config(self.config|{"diagnostic_only_metrics":ignored})
        legacy={k:v for k,v in self.config.items() if k!="diagnostic_only_metrics"}
        validate_config(legacy)
        validate_config(self.config)

    def test_api_disabled_exits_before_numerical_import_or_output_creation(self):
        spec = importlib.util.spec_from_file_location("agent_cli_test",ROOT/"scripts/run_trajectory_agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT/".runtime") as temp:
            output = Path(temp)/"must_not_exist"
            args = ["run_trajectory_agent.py","--mode","api","--output",str(output)]
            with patch.object(module.api_client,"API_IMPLEMENTED",False), patch.object(sys,"argv",args), patch.dict(sys.modules,{"torch":None}), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:
                    module.main()
            self.assertEqual(caught.exception.code,2)
            self.assertIn("API is disabled",err.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
