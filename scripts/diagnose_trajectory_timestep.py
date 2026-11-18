"""CUDA-only timestep sensitivity diagnostic; no accepted-data export."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/supermimic_official_5efbf114/src')]
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.runtime/research_cache'))
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
from sesame_trajectory_agent import validation
from sesame_trajectory_agent.trajectories import ReferenceLibrary,write_json,save_npz
from sesame_trajectory_agent.schema import Proposal


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timesteps',type=float,nargs='+',default=[.002,.001,.0005])
    p.add_argument('--cases',nargs='+',choices=['standing','moving'],default=['standing','moving'])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    config=json.loads((ROOT/'configs/trajectory_agent.json').read_text())
    config.update(duration_s=1.,robust_trials=0)
    source=ROOT/config['source'];library=ReferenceLibrary(source)
    original=validation.make_cfg
    rows=[]
    for dt in a.timesteps:
        def configured(*args,**kwargs):
            cfg=original(*args,**kwargs)
            cfg.sim.mujoco.timestep=dt;cfg.decimation=round(.02/dt)
            return cfg
        validation.make_cfg=configured
        engine=validation.ExecutionEngine(source,config)
        try:
            for kind in a.cases:
                ref,_=library.generate(Proposal(smoothing_window_s=.12),config)
                if kind=='standing':
                    for name in ('joint_pos','root_pos_w','root_quat_w'):
                        ref[name]=ref[name][0:1].expand_as(ref[name]).clone()
                    for name in ('joint_vel','root_lin_vel_w','root_ang_vel_w'):ref[name].zero_()
                kin,body=engine.check_kinematics(ref)
                trials,trace=engine.execute(ref,body)
                tail=trace['joint_vel'][11:]
                row={'physics_dt_s':dt,'case':kind,'kinematics':kin,'metrics':trials[0]['metrics'],
                     'after_200ms_joint_velocity_rms_rad_s':float(tail.square().mean().sqrt()),
                     'after_200ms_root_speed_rms_m_s':float(trace['root_lin_vel_w'][11:].square().sum(-1).mean().sqrt())}
                rows.append(row)
                save_npz(a.output/f'{kind}_{dt}.npz',trace)
                write_json(a.output/'results.json',rows)
                print(json.dumps(row),flush=True)
        finally:engine.close()
    validation.make_cfg=original


if __name__=='__main__':main()
