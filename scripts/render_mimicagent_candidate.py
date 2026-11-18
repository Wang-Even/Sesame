"""Render a generated reference beside its recorded physical execution on GPU.

This is not a trained-policy inference video. No new dynamics rollout is run.
"""
import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"third_party/supermimic_official_5efbf114/src")]
from export_research_evaluation_videos import setup,load_trace,analyze,RecordedStateRenderer,panel,bounds_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--attempt",type=Path,help="Explicit attempt directory for a diagnostic video; selection is recorded")
    a=p.parse_args();torch=setup()
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image
    from sesame_trajectory_agent.validation import ExecutionEngine
    from sesame_trajectory_agent.trajectories import FIELDS,sha256,write_json
    protocol=json.loads((a.run/"protocol.json").read_text())
    # First executed attempt in stored order, never best-of-N selection.
    executions=sorted((a.run/"candidates").glob("candidate_*/attempt_*/nominal_execution.npz"))
    if not executions:raise RuntimeError("No physical execution exists yet")
    selected=a.attempt if a.attempt else executions[0].parent
    if not selected.resolve().is_relative_to((a.run/"candidates").resolve()):raise ValueError("Attempt must belong to this run")
    report=json.loads((selected/"report.json").read_text())
    actual=load_trace(selected/"nominal_execution.npz")
    reference={k:v[:,None] for k,v in load_trace(selected/"candidate_reference.npz").items() if k in FIELDS}
    rv,rs=analyze(reference,0,target=protocol["config"]["target_speed_m_s"])
    av,as_=analyze(actual,0,target=protocol["config"]["target_speed_m_s"])
    displacement_speed=protocol["config"].get("mean_speed_measurement")=="displacement"
    if displacement_speed:
        # Display interval-average velocity; do not rewrite the recorded trace.
        for trace,values in ((reference,rv),(actual,av)):
            pos=trace["root_pos_w"][:,0]
            ends=torch.arange(len(pos),device=pos.device)
            starts=(ends-10).clamp_min(0)
            elapsed=((ends-starts)*.02).clamp_min(.02)
            values["velocity"]=((pos[ends]-pos[starts])/elapsed[:,None]).cpu().tolist()
    controller=("反馈控制＋原生 PD 实际执行" if protocol["config"].get("execution_feedback")
                else "原生 PD 控制器实际执行")
    bounds=bounds_for(rv,av)
    output=a.run/"reference_vs_execution.mp4"
    if output.exists():raise RuntimeError("Video exists; do not overwrite")
    e=ExecutionEngine(ROOT/protocol["config"]["source"],protocol["config"])
    r=RecordedStateRenderer(e.env)
    try:
        with imageio.get_writer(output,fps=25,codec="libx264",quality=7,macro_block_size=16) as writer:
            for t in range(0,min(len(reference["joint_pos"]),len(actual["joint_pos"]))-1,2):
                left=panel(r.frame(reference,t,0,rv,agent=True),rv,t,"Agent 生成的参考轨迹","运动学播放，不代表动力学成功",bounds)
                right=panel(r.frame(actual,t,0,av,agent=True),av,t,controller,
                            "全部检验通过" if report["accepted"] else "未通过入库检验",bounds)
                frame=Image.new("RGB",(1280,720));frame.paste(left,(0,0));frame.paste(right,(640,0))
                writer.append_data(np.asarray(frame))
        write_json(a.run/"video.json",{"video":str(output),"selected_attempt":str(selected),
            "selection":"explicit diagnostic attempt; not an aggregate result" if a.attempt else "first physically executed attempt in candidate/attempt order",
            "reference":rs,"execution":as_,"accepted":report["accepted"],"reasons":report["reasons"],
            "controller":controller,
            "display_velocity":"position difference over preceding 0.2 s" if displacement_speed else "mean sampled velocity over preceding 0.2 s",
            "execution_feedback":protocol["config"].get("execution_feedback"),
            "reference_sha256":sha256(selected/"candidate_reference.npz"),
            "execution_sha256":sha256(selected/"nominal_execution.npz"),"video_sha256":sha256(output),
            "note":"Reference animation vs recorded physical execution; NOT trained-policy inference; GPU FK/EGL, 1x speed"})
        print(json.dumps({"video":str(output),"accepted":report["accepted"]}))
    finally:r.close();e.close()

if __name__=="__main__":main()
