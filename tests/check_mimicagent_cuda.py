"""CUDA execution/IK/no-clipping checks. Standing fixture is NOT walking data."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"third_party/supermimic_official_5efbf114/src")]
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".runtime/research_cache"))
os.environ.setdefault("OMP_NUM_THREADS","4")
import torch
from sesame_trajectory_agent.validation import ExecutionEngine
from sesame_trajectory_agent.mimic_motion import MotionProgramRuntime

CODE='''def generate(t, home, m):
    z = 0*t
    o = 1+z
    q = home['joint_pos'] + z[:,None]
    pos = m.stack([home['speed']*t,z,o*home['base_height']],dim=1)
    rpy = m.stack([z,z,z],dim=1)
    contact = m.stack([o>0,o>0,o>0,o>0],dim=1)
    return {'root_pos_w':pos,'root_rpy':rpy,'joint_pos':q,'contacts':contact}
'''

def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    c=json.loads((ROOT/'configs/trajectory_agent_unclipped.json').read_text());c['duration_s']=.2
    e=ExecutionEngine(ROOT/c['source'],c)
    try:
        rt=MotionProgramRuntime(e,c)
        ref,contacts,g=rt.generate(CODE)
        kin,_=rt.check(ref,contacts,g)
        assert kin['passed'],kin
        missing,_=rt.check(ref,torch.zeros_like(contacts),g)
        assert not missing['passed'] and 'declared_contacts_match_ground' in missing['reasons']
        # This only checks kinematic structure; fake forward translation of a
        # standing body is intentionally not declared a successful walking demo.
        assert all(x.is_cuda for x in ref.values())
        q=ref['joint_pos'];feet=rt.feet(q)
        recovered=rt.ik(feet)
        assert float((rt.feet(recovered)-feet).abs().max())<1e-4
        bad=CODE.replace("home['joint_pos'] + z[:,None]","home['joint_pos'] + z[:,None] + 4.0")
        excess,contacts,g=rt.generate(bad)
        torch.testing.assert_close(excess['joint_pos'],q+4.,atol=1e-6,rtol=0)
        invalid,_=rt.check(excess,contacts,g)
        assert 'reference_joint_limit' not in invalid['reasons']
        assert 'joint_limit_violation_max_rad' not in invalid['metrics']
        # Unreachable feet must leave a measurable residual, never be projected.
        rt.ik(feet+1.)
        assert rt.ik_error>.1
        print(json.dumps({'cuda_program':'passed','fk_ik':'passed','no_clipping':'passed',
                          'unreachable_rejected':'passed','admitted_training_trajectories':0}))
    finally:e.close()

if __name__=='__main__':main()
