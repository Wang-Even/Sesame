"""Write chat-agent-authored gait programs; no API, tensor computation or simulation.

Parameter choices are manually supplied after reading saved verification reports.
Every submitted revision counts toward the experiment's fixed 30-version budget.
"""
import argparse
import json
from pathlib import Path


def program(p):
    f=p['frequency'];d=p.get('duty',.5);a=p['amplitude'];b=p.get('knee_bias',.15)
    lift=p.get('knee_lift',.45);center=p.get('hip_center',1.3685)
    height=p.get('height',.04673357);phase=p.get('phase',.125)
    lines=['def generate(t, home, m):','    z = 0*t',f'    pA = {f}*t + {phase}',
           '    pA = pA - m.floor(pA)','    pB = pA + 0.5','    pB = pB - m.floor(pB)']
    for label in ['A','B']:
        lead=p.get('lift_lead',0.)
        shape=p.get('lift_shape','squared_sine')
        wave=(f'm.cos(6.283185307179586*p{label})' if p.get('hip_wave')=='cosine' else
              f'm.where(p{label} < {d}, 1 - 2*p{label}/{d}, -1 + {-2*(1-d)/d}*s{label} + {6+6*(1-d)/d}*s{label}**2 + {-4-4*(1-d)/d}*s{label}**3)')
        lines += [f'    s{label} = (p{label} - {d}) / {1-d}',
                  f'    w{label} = {wave}',
                  f'    pk{label} = p{label} + {lead}',f'    pk{label} = pk{label} - m.floor(pk{label})',
                  f'    sk{label} = (pk{label} - {d}) / {1-d}',
                  f'    k{label} = {b} + m.where(pk{label} < {d}, z, {lift}*m.sin(3.141592653589793*sk{label})'+('**2)' if shape=='squared_sine' else ')'),
                  f'    c{label} = pk{label} < {d}']
    phase_legs=p.get('phases',['A','B','B','A'])
    ra,rb,la,lb=phase_legs
    asym=p.get('left_scale',1.);rear=p.get('rear_scale',1.)
    lines += [f'    q = m.stack([{center}+{a}*w{ra}, 3.141592653589793-k{ra}, {3.141592653589793-center}+{a*rear}*w{rb}, k{rb}, {3.141592653589793-center}-{a*asym}*w{la}, k{la}, {center}-{a*asym*rear}*w{lb}, 3.141592653589793-k{lb}], dim=-1)',
              f'    pos = m.stack([home["speed"]*t, z, {height}+z], dim=-1)',
              '    rpy = m.stack([z,z,z],dim=-1)',
              f'    contacts = m.stack([c{ra},c{rb},c{la},c{lb}],dim=-1)',
              '    return {"root_pos_w":pos,"root_rpy":rpy,"joint_pos":q,"contacts":contacts}']
    return '\n'.join(lines)+'\n'


def submit(inbox,index,params,rationale,parents=None):
    folder=inbox/f'candidate_{index:03d}'
    folder.mkdir(parents=True,exist_ok=False)
    plan={'intent':'Walk straight at 0.1 m/s using hip rotation about the actual Sesame vertical axes',
        'cyclic':True,'phases':[
            {'start':0.,'end':.5,'name':'support_A_transfer_B','contacts':['right_front','left_rear'],
             'base_motion':'x = 0.1 t, fixed authored height, level body',
             'limb_motion':'Support hips retract, opposite pair lifts; piecewise cubic hip recovery and smooth knee lift'},
            {'start':.5,'end':1.,'name':'support_B_transfer_A','contacts':['right_rear','left_front'],
             'base_motion':'Continue forward; swap support pairs',
             'limb_motion':'Opposite diagonal supports and retracts; other pair transfers'}],
        'coding_notes':rationale+'; phase origin and support overlap are explicitly encoded in motion.py.'}
    (folder/'plan.json').write_text(json.dumps(plan,indent=2))
    (folder/'design.json').write_text(json.dumps({'author':'current Codex conversation agent','api_calls':0,
        'candidate_index':index,'parameters':params,'rationale':rationale,'feedback_reports':parents or [],
        'note':'Manual adaptive design; no learned policy clips, source resampling, clipping or gate changes'},indent=2))
    (folder/'motion.py').write_text(program(params))
    (folder/'READY').write_text('Inputs complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inbox',type=Path);parser.add_argument('index',type=int)
    parser.add_argument('parameters',help='JSON parameters');parser.add_argument('rationale')
    args=parser.parse_args();submit(args.inbox,args.index,json.loads(args.parameters),args.rationale)
