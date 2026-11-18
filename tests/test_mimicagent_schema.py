"""Metadata-only checks: no model or numerical computation on CPU."""
import unittest
from sesame_trajectory_agent.motion_program import parse_program
from sesame_trajectory_agent.mimic_prompts import validate_plan


class TestProgramBoundary(unittest.TestCase):
    def test_reject_host_access_and_mutation(self):
        programs=["import os\ndef generate(t, home, m):\n    return {}",
            "def generate(t, home, m):\n    x = t.cpu()\n    return {}",
            "def generate(t, home, m):\n    x = m.__class__\n    return {}",
            "def generate(t, home, m):\n    x = open('/etc/passwd')\n    return {}",
            "def generate(t, home, m):\n    home['speed'] = 0\n    return {}",
            "def generate(t, home, m):\n    while True: pass\n    return {}",
            "def generate(t, home, m):\n    x = m.clamp(t,0,1)\n    return {}"]
        for code in programs:
            with self.subTest(code=code),self.assertRaises(ValueError):parse_program(code)

    def test_vectorized_code_allowed(self):
        parse_program("def generate(t, home, m):\n    p = m.sin(t)\n    return {'joint_pos': p[:,None] + home['joint_pos']}\n")

    def test_plan_has_real_coverage(self):
        phase={"start":0.,"end":1.,"name":"walk","contacts":["right_front"],"base_motion":"forward","limb_motion":"cycle"}
        plan={"intent":"walk","cyclic":True,"phases":[phase],"coding_notes":"preserve goal"}
        validate_plan(plan)
        for interval in [(0.1,1.),(0.,.9),(0.,float('nan'))]:
            invalid=plan|{"phases":[phase|dict(zip(("start","end"),interval))]}
            with self.assertRaises(ValueError):validate_plan(invalid)


if __name__=="__main__":unittest.main()
