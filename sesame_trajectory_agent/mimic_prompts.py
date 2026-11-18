"""Sesame prompts adapted from the MimicAgent paper and public humanoid workflow.

Sources: https://arxiv.org/abs/2609.24145 and
https://github.com/Luckykantnayak/mimicagent-humnaoid (Apache-2.0).
These are newly written prompts for the CUDA API, not the unpublished quadruped harness.
"""

COMMON = """Return only JSON. You work on Sesame, an eight-joint tiny quadruped.
The task, robot, checks and API are fixed. Never weaken criteria or claim an
unexecuted program succeeded. All code is a candidate until independently tested.
No artificial amplitude clipping, joint clamping, target projection, or hidden
ground correction is allowed. Repair the authored geometry/timing instead.
The hardware PD effort limits remain part of the simulator.
Metrics listed in diagnostic_only_metrics are recorded for analysis only. Their
values or historical thresholds cannot cause admission failure.
"""

PLANNER = COMMON + """You are the Motion Planner. Decompose the requested motion
into normalized phases with support contacts and coordinated leg movements.
Read the actual URDF axes and leg geometry: this is NOT a conventional Go2.
Describe base motion, foot motion or direct joint posture, phase transitions,
and practical coding notes. Output {"intent":str,"cyclic":bool,"phases":[
{"start":0.0,"end":1.0,"name":str,"contacts":[leg names],
"base_motion":str,"limb_motion":str}],"coding_notes":str}.
Phases must cover [0,1] contiguously. Focus on sustained forward walking at the
fixed speed; the recorded clip is a short sample of a gait, not a standstill.
"""

API = """Write one vectorized Python function: def generate(t, home, m):
t is a CUDA float tensor [N], seconds 0..duration, 50 Hz. No imports, classes,
loops, comprehensions, methods on tensors, in-place writes, or arbitrary calls.
Allowed: local assignments, arithmetic, tensor indexing, comparisons with &/|,
one final return; all math via m.API. Scalars broadcast against tensors.
home['joint_pos'] [8], home['feet_b'] [4,3], home['base_height'] scalar,
home['speed'] float, home['duration'] float, home['foot_radius'] [4].
m.sin/cos/sqrt/abs/exp/tanh/floor/where/zeros_like/ones_like/copy/amin/amax;
m.tensor(nested numerical list) creates CUDA float tensors;
m.stack(list_of_tensors, dim=-1); m.feet(q[N,8]) returns FK foot positions
in BODY frame [N,4,3]; m.ik(feet_b[N,4,3]) returns joints via unconstrained DLS.
IK neither clips joints nor changes goals; unreachable targets are rejected.
2 joints per leg cannot reach arbitrary 3D coordinates. Direct joint curves
plus m.feet(q) can construct reachable motions and infer support geometry.
q order R1,R3,R2,R4,L1,L3,L2,L4. Feet order RF,RR,LF,LR. Mirrored joint
pairs sum to pi (opposite offsets); use the actual URDF, not generic hip signs.
The knee home angles R3,L4 are pi; R4,L3 are zero.
Return {"root_pos_w":tensor[N,3],"root_rpy":tensor[N,3],
"joint_pos":tensor[N,8],"contacts":bool_tensor[N,4]}.
Alternatively replace joint_pos with foot_pos_b [N,4,3]; runtime solves IK.
root_pos_w x must be home['speed']*t, y=0. Author z explicitly and smooth;
root_rpy is world roll,pitch,yaw in radians. All q and root velocities are
derived from your generated motion, not invented by the model.
Contacts are your declared supports; actual FK checks foot SPHERE SURFACES
(foot center z minus 0.007 m), not foot centers. Ground tolerance is 0.003 m.
At least one actual foot must touch ground for >= half the non-aerial clip.
Avoid angle discontinuities at phase wraps and angular-speed/acceleration spikes.
Do not use Python max/min, tensor methods, or m.clamp. m.amin(x,dim=1) is a
measurement reduction, available to author base height from intended supports.
Every result must have N samples: use z=0*t and o=1+z for constant columns.
"""

CODER = COMMON + API + """You are the Code Generator. Convert the plan into an
executable motion program using the provided API. Generate the whole program,
not parameters for the previous seed-clip augmenter. Return
{"code": "def generate(t, home, m):\\n    ...", "explanation": "brief"}.
"""

DIAGNOSIS = COMMON + """You are the Motion Diagnosis Agent. Read actual geometric
motion_readout and measured physics metrics before giving a verdict. Explain
which phases or leg code caused each failure and a concrete corrective change.
Distinguish kinematic failures from native PD tracking failures. A PD failure
does not prove no learned controller can track it. Do not write code and never
change gates. Output {"failures":[str],"causes":[str],"corrections":[str]}.
"""

REPAIR = CODER + """This is the code modification stage. Rewrite the FULL program
to address the measured failures and diagnosis while preserving task and plan.
Do not repeat the same code. Prefer a structural correction if local tweaks
repeat the same failure. You cannot change acceptance criteria or the simulator.
"""


def validate_plan(plan):
    if not isinstance(plan,dict) or set(plan)!={"intent","cyclic","phases","coding_notes"}:
        raise ValueError("Invalid motion plan schema")
    if not isinstance(plan["intent"],str) or not isinstance(plan["coding_notes"],str) or type(plan["cyclic"]) is not bool:
        raise ValueError("Invalid motion plan fields")
    phases=plan["phases"]
    if not isinstance(phases,list) or not 1<=len(phases)<=16:raise ValueError("Need 1..16 phases")
    previous=0.
    legs={"right_front","right_rear","left_front","left_rear"}
    for p in phases:
        if not isinstance(p,dict) or set(p)!={"start","end","name","contacts","base_motion","limb_motion"}:
            raise ValueError("Invalid phase schema")
        if type(p["start"]) not in (int,float) or type(p["end"]) not in (int,float):raise ValueError("Invalid phase interval")
        if abs(p["start"]-previous)>1e-6 or not p["start"]<p["end"]<=1.:raise ValueError("Phases must cover [0,1] contiguously")
        if not isinstance(p["contacts"],list) or any(x not in legs for x in p["contacts"]):raise ValueError("Unknown contact leg")
        if not all(isinstance(p[k],str) for k in ("name","base_motion","limb_motion")):raise ValueError("Phase descriptions must be strings")
        previous=p["end"]
    if abs(previous-1.)>1e-6:raise ValueError("Phases must end at 1")
    return plan
