"""Record CUDA policy evaluation and render measured states with GPU graphics.

Videos are visualizations of saved executed states, never imagined WM rollouts.
All policy, physics, forward kinematics and trajectory statistics run on CUDA.
Host work is file I/O, drawing annotations and encoding recorded images.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('XDG_CACHE_HOME','/home/ywwang/sesame/.runtime/research_cache')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/supermimic_official_5efbf114/src')]


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.40)
    return torch


def collect(a):
    import numpy as np
    torch=setup()
    from sesame_research.evaluation import Evaluator
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor,build_world_model
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'sources').mkdir()
    sources={}
    for p in [Path(__file__),ROOT/'sesame_research/evaluation.py',ROOT/'sesame_research/task.py',ROOT/'sesame_research/robot.py']:
        (a.output/'sources'/p.name).write_bytes(p.read_bytes());sources[str(p.relative_to(ROOT))]=digest(p)
    protocol={'seed':91042,'physics_dt_s':.0005,'control_hz':50,'video_fps':25,'playback_speed':1,
        'test_source':str(a.test),'test_clips':read(a.test/'meta.json')['clips'],
        'clip_order':read(a.test/'motion_names.json'),'selection':'all test clips in stored order; first clip preview is not best-of-N',
        'agent_selection':'all candidates in stored order; nominal and randomized trial 0 shown side by side',
        'sources':sources,'models':{},'course_scoring':False,
        'video_note':'EGL rendering of recorded actual CUDA simulator states; no world-model prediction or target-pose playback',
        'velocity_overlay':'0.2 s trailing mean of measured world-frame root-link velocity; raw samples retained',
        'policy_statistics_interval_s':[0.,8.98],
        'heading':'body yaw in world frame; movement direction separately derived from net displacement'}
    write(a.output/'protocol.json',protocol)
    evaluator=Evaluator(a.test,91042,.0005)
    try:
        for method,group in [('supermimic','supermimic_original_seed42'),('ppo','ppo_original_seed42')]:
            checkpoint=a.suite/group/'final.pt'
            saved=read(a.suite/group/'protocol.json')
            assert saved['args']['physics_dt']==.0005
            cfg=load_runner_cfg({'runner_cfg':saved['runner_config']})
            torch.manual_seed(42)
            wm=build_world_model(cfg,evaluator.env,'cuda:0')
            actor=build_actor(cfg,wm,'cuda:0')
            bundle=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
            actor.load_state_dict(bundle['actor'],strict=True);actor.eval()
            store={}
            def capture(t,raw,alive,reference):
                d=raw.scene['robot'].data
                values={'qpos':raw.sim.data.qpos,'qvel':raw.sim.data.qvel,
                    'root_pos_w':d.root_link_pos_w-raw.scene.env_origins,
                    'root_quat_w':d.root_link_quat_w,'root_lin_vel_w':d.root_link_lin_vel_w,
                    'reference_pos_w':reference[0][:,0],'reference_lin_vel_w':reference[2][:,0],
                    'alive':alive}
                for k,v in values.items():store.setdefault(k,[]).append(v.clone())
            print('EVALUATING '+method,flush=True)
            metrics=evaluator.run(actor,wm.representation,wm.sim_context,cfg.history_length,450,trace_callback=capture)
            trace={k:torch.stack(v) for k,v in store.items()}
            assert len(trace['qpos'])==450
            assert all(bool(torch.isfinite(v).all()) for v in trace.values())
            np.savez_compressed(a.output/(method+'_trace.npz'),**{k:v.cpu().numpy() for k,v in trace.items()},fps=50)
            write(a.output/(method+'_evaluation.json'),metrics)
            protocol['models'][method]={'checkpoint':str(checkpoint),'sha256':digest(checkpoint),
                'trace_sha256':digest(a.output/(method+'_trace.npz')),'actor_parameters':sum(p.numel() for p in actor.parameters())}
            write(a.output/'protocol.json',protocol)
            print(method+' '+json.dumps(metrics,ensure_ascii=False),flush=True)
            del trace,store,actor,wm,bundle
        write(a.output/'collection_status.json',{'status':'complete','finished_unix':time.time()})
    finally:evaluator.close()


def load_trace(path):
    import numpy as np
    import torch
    with np.load(path,allow_pickle=False) as x:
        return {k:torch.as_tensor(x[k].copy(),device='cuda:0') for k in x.files if x[k].dtype.kind in 'fbiu' and x[k].ndim>0}


def analyze(trace,episode,reference=None,target=.1,state_limit=None):
    """Compute display values/statistics on CUDA, then transfer scalar records."""
    import torch
    pos=trace['root_pos_w'][:state_limit,episode];vel=trace['root_lin_vel_w'][:state_limit,episode]
    quat=trace['root_quat_w'][:state_limit,episode]
    w,x,y,z=quat.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    delta=pos-pos[0]
    if reference is None:
        ref=pos[0].expand_as(pos).clone()
        ref[:,0]+=torch.arange(len(pos),device='cuda:0')*.02*target
        refv=torch.zeros_like(vel);refv[:,0]=target
    else:
        ref=reference['reference_pos_w'][:state_limit,episode]
        refv=reference['reference_lin_vel_w'][:state_limit,episode]
    def smooth(v):
        sums=torch.cat([v.new_zeros(1,v.shape[-1]),v.cumsum(0)])
        ends=torch.arange(1,len(v)+1,device='cuda:0');starts=(ends-10).clamp_min(0)
        return (sums[ends]-sums[starts])/(ends-starts)[:,None]
    duration=(len(pos)-1)*.02
    net=delta[-1,:2]
    direction=float(torch.atan2(net[1],net[0])*180/math.pi) if float(net.norm())>1e-6 else None
    alive=trace.get('alive')
    if alive is not None:alive=alive[:state_limit]
    summary={'duration_s':duration,'forward_displacement_m':float(net[0]),'lateral_displacement_m':float(net[1]),
        'forward_speed_from_displacement_m_s':float(net[0]/duration),
        'mean_vx_m_s':float(vel[1:,0].mean()),'mean_vy_m_s':float(vel[1:,1].mean()),
        'mean_planar_speed_m_s':float(vel[1:,:2].norm(dim=-1).mean()),
        'reference_mean_vx_m_s':float(refv[1:,0].mean()),'reference_mean_vy_m_s':float(refv[1:,1].mean()),
        'net_movement_direction_deg':direction,'final_body_yaw_deg':float(yaw[-1]*180/math.pi),
        'world_body_yaw_rms_deg':float(yaw[1:].square().mean().sqrt()*180/math.pi),
        'completed_without_local_termination':bool(alive[:,episode].all()) if alive is not None else None}
    values={'pos':pos.cpu().tolist(),'delta':delta.cpu().tolist(),'yaw':yaw.cpu().tolist(),
        'velocity':smooth(vel).cpu().tolist(),'ref_velocity':smooth(refv).cpu().tolist(),
        'ref_delta':(ref-ref[0]).cpu().tolist(),
        'alive':alive[:,episode].cpu().tolist() if alive is not None else [True]*len(pos)}
    return values,summary


class RecordedStateRenderer:
    """GPU FK and EGL graphics; no CPU forward dynamics during frame rendering."""
    def __init__(self,raw):
        import mujoco
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        class GpuTransformsRenderer(OffscreenRenderer):
            def _sync_data_fields(self,data,env_idx):
                fields=('qpos','qvel','xpos','xquat','xmat','xipos','ximat','geom_xpos','geom_xmat',
                        'site_xpos','site_xmat','cam_xpos','cam_xmat','light_xpos','light_xdir','subtree_com')
                for field in fields:
                    target=getattr(self._data,field)
                    if target.size:
                        value=getattr(data,field)[env_idx].cpu().numpy()
                        target[:]=value.reshape(target.shape)
        cfg=ViewerConfig(width=640,height=360,origin_type=ViewerConfig.OriginType.WORLD,
            distance=.48,elevation=-32,azimuth=90,max_extra_envs=0,
            enable_shadows=True,enable_reflections=False,fovy=45)
        self.renderer=GpuTransformsRenderer(raw.sim.mj_model,cfg,raw.scene)
        self.renderer.initialize();self.raw=raw;self.mujoco=mujoco
        # Visual-only ground texture, isolated in the renderer's model copy.
        self.renderer._model.vis.map.znear=.001

    def frame(self,trace,t,episode,values,agent=False):
        import torch
        raw=self.raw
        if agent:
            entity=raw.scene['robot'];ids=torch.tensor([0],device='cuda:0')
            root=torch.cat([trace[k][t,episode] for k in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w')])[None].clone()
            root[:,:3]+=raw.scene.env_origins[:1]
            entity.write_root_state_to_sim(root,env_ids=ids)
            entity.write_joint_state_to_sim(trace['joint_pos'][t,episode][None],trace['joint_vel'][t,episode][None],env_ids=ids)
            view_id=0
        else:
            raw.sim.data.qpos[:]=trace['qpos'][t]
            raw.sim.data.qvel[:]=trace['qvel'][t]
            view_id=episode
        raw.sim.forward()
        r=self.renderer;r._env_ids=(view_id,)
        origin=raw.scene.env_origins[view_id].cpu().tolist()
        position=[values['pos'][t][j]+origin[j] for j in range(3)]
        r._cam.lookat[:]=[position[0],position[1],.045]
        r.update(raw.sim.data)
        # World +X arrow drawn on the ground; it never changes with body yaw.
        start=[position[0]-.07,position[1]-.11,.003]
        end=[start[0]+.16,start[1],start[2]]
        scene=r.renderer.scene
        g=scene.geoms[scene.ngeom]
        self.mujoco.mjv_connector(g,self.mujoco.mjtGeom.mjGEOM_ARROW,.003,start,end)
        g.rgba[:]=[.9,.12,.12,1];scene.ngeom+=1
        return r.render()

    def close(self):self.renderer.close()


def draw_arrow(d,start,end,color,width=3):
    d.line([start,end],fill=color,width=width)
    theta=math.atan2(end[1]-start[1],end[0]-start[0])
    d.polygon([end,(end[0]-9*math.cos(theta-.45),end[1]-9*math.sin(theta-.45)),
                    (end[0]-9*math.cos(theta+.45),end[1]-9*math.sin(theta+.45))],fill=color)


def bounds_for(*values):
    points=[p for v in values for k in ('delta','ref_delta') for p in v[k]]
    xmin=min(p[0] for p in points);xmax=max(p[0] for p in points)
    ymin=min(p[1] for p in points);ymax=max(p[1] for p in points)
    span=max(xmax-xmin,ymax-ymin,.4)*1.2
    return ((xmin+xmax)/2,(ymin+ymax)/2,span)


def panel(frame,values,t,title,subtitle,bounds):
    from PIL import Image,ImageDraw,ImageFont
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fonts={n:ImageFont.truetype(font,n) for n in (15,17,19,24)}
    im=Image.new('RGB',(640,720),'#f5f7fa');d=ImageDraw.Draw(im)
    d.rectangle((0,0,639,79),fill='#142b43')
    d.text((16,7),title,font=fonts[24],fill='white')
    d.text((16,44),f'{subtitle}  |  t={t*.02:.2f}s  |  1×',font=fonts[17],fill='#c7d9e8')
    im.paste(Image.fromarray(frame),(0,80))
    d.text((16,445),'俯视轨迹：+X 向右，+Y 向上',font=fonts[17],fill='#263e53')
    left,top,size=16,478,232
    cx,cy,span=bounds
    def xy(p):return (left+size/2+(p[0]-cx)/span*size,top+size/2-(p[1]-cy)/span*size)
    d.rectangle((left,top,left+size,top+size),fill='white',outline='#b9c6d0')
    step=.1 if span<1 else .2
    for i in range(math.floor((cx-span/2)/step),math.ceil((cx+span/2)/step)+1):
        x=xy([i*step,cy])[0]
        if left<=x<=left+size:d.line((x,top,x,top+size),fill='#e4e9ee')
    for i in range(math.floor((cy-span/2)/step),math.ceil((cy+span/2)/step)+1):
        y=xy([cx,i*step])[1]
        if top<=y<=top+size:d.line((left,y,left+size,y),fill='#e4e9ee')
    ref=[xy(p) for p in values['ref_delta'][:t+1]]
    actual=[xy(p) for p in values['delta'][:t+1]]
    if len(ref)>1:d.line(ref,fill='#a4a8ad',width=2)
    if len(actual)>1:d.line(actual,fill='#166bc0',width=3)
    start=xy([0,0]);d.ellipse((start[0]-3,start[1]-3,start[0]+3,start[1]+3),fill='#222222')
    here=actual[-1];angle=values['yaw'][t]
    draw_arrow(d,here,(here[0]+20*math.cos(angle),here[1]-20*math.sin(angle)),'#1b9a65',3)
    draw_arrow(d,(left+10,top+20),(left+57,top+20),'#df3232',2)
    d.text((left+62,top+8),'+X',font=fonts[15],fill='#df3232')
    v=values['velocity'][t];rv=values['ref_velocity'][t];delta=values['delta'][t]
    lines=[f'实际 vx：{v[0]:+.3f} m/s',f'实际 vy：{v[1]:+.3f} m/s',
           f'参考 vx：{rv[0]:+.3f} m/s',f'机身朝向：{math.degrees(angle):+.1f}°',
           f'前向位移：{delta[0]:+.3f} m',f'横向偏移：{delta[1]:+.3f} m']
    for i,line in enumerate(lines):d.text((270,476+i*29),line,font=fonts[19],fill='#233e53')
    d.text((270,654),'蓝：实际轨迹  灰：参考轨迹',font=fonts[15],fill='#456178')
    d.text((270,678),'绿：机头方向；速度取近 0.2 秒均值',font=fonts[15],fill='#456178')
    if not values['alive'][t]:d.text((270,700),'本地环境已提前终止',font=fonts[15],fill='#bb2222')
    return im


def render(a):
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image
    torch=setup()
    from sesame_research.evaluation import Evaluator
    if read(a.output/'collection_status.json')['status']!='complete':raise RuntimeError('Collect traces first')
    assert not (a.output/'video_status.json').exists(),'Use a fresh video output or remove only failed render status after inspection'
    protocol=read(a.output/'protocol.json')
    protocol['policy_statistics_interval_s']=[0.,8.98]
    protocol['policy_endpoint_note']='Only frames 0..449 (pre-step states) are used. A legacy frame 450, if present, is post-timeout reset and is excluded.'
    protocol['render_source_sha256']=digest(Path(__file__))
    (a.output/'render_source.py').write_bytes(Path(__file__).read_bytes())
    write(a.output/'protocol.json',protocol)
    evaluator=Evaluator(a.test,91042,.0005)
    renderer=RecordedStateRenderer(evaluator.env.unwrapped)
    summaries={'policy':{},'agent':[],
        'policy_statistics_note':'Use pre-step states at 0..8.98 s; exclude any post-timeout auto-reset sample. Video duration is 9 s at 25 fps.',
        'agent_statistics_note':'Use original execution states at 0..6 s; their simulator had auto_reset=False.'}
    options={'fps':25,'codec':'libx264','quality':8,'macro_block_size':1,
             'ffmpeg_params':['-movflags','+faststart','-threads','2']}
    try:
        traces={m:load_trace(a.output/(m+'_trace.npz')) for m in ('supermimic','ppo')}
        videos=a.output/'videos';videos.mkdir(exist_ok=True)
        writer=imageio.get_writer(videos/'policy_comparison_all8.mp4',**options)
        try:
            for episode in range(8):
                values={}
                for method,trace in traces.items():
                    values[method],stats=analyze(trace,episode,reference=trace,state_limit=450)
                    summaries['policy'].setdefault(method,[]).append(stats)
                bounds=bounds_for(*values.values())
                preview=imageio.get_writer(videos/'policy_comparison_first_clip.mp4',**options) if episode==0 else None
                try:
                    for t in range(0,450,2):
                        canvas=Image.new('RGB',(1280,720))
                        for index,(method,name) in enumerate([('supermimic','SuperMimic'),('ppo','PPO')]):
                            v=values[method]
                            # Never display an auto-reset episode as continuing the failed one.
                            valid=[i for i in range(t+1) if v['alive'][i]]
                            shown=t if v['alive'][t] else valid[-1]
                            frame=renderer.frame(traces[method],shown,episode,v)
                            canvas.paste(panel(frame,v,t,f'{name} · 测试片段 {episode+1}/8','策略仿真实测状态回放',bounds),(index*640,0))
                        pixels=np.asarray(canvas);writer.append_data(pixels)
                        if preview:preview.append_data(pixels)
                        if episode==0 and t==224:canvas.save(a.output/'policy_preview.png')
                    print(f'RENDERED policy clip {episode+1}/8',flush=True)
                finally:
                    if preview:preview.close()
        finally:writer.close()
        del traces
        directories=sorted(a.agent.glob('candidates/*'))
        writer=imageio.get_writer(videos/'agent_candidates_all12.mp4',**options)
        try:
            for index,folder in enumerate(directories):
                report=read(folder/'report.json')
                nominal=load_trace(folder/'nominal_execution.npz')
                randomized=load_trace(folder/'randomized_execution.npz')
                vn,sn=analyze(nominal,0);vr,sr=analyze(randomized,0)
                summaries['agent'].append({'id':folder.name,'nominal':sn,'randomized_trial_0':sr,
                    'nominal_sha256':digest(folder/'nominal_execution.npz'),
                    'randomized_sha256':digest(folder/'randomized_execution.npz'),
                    'original_experimental_filter_passed':report['accepted']})
                assert digest(folder/'nominal_execution.npz')==report['nominal_trace_sha256']
                assert digest(folder/'randomized_execution.npz')==report['randomized_trace_sha256']
                bounds=bounds_for(vn,vr)
                preview=imageio.get_writer(videos/'agent_first_candidate.mp4',**options) if index==0 else None
                try:
                    for t in range(0,300,2):
                        canvas=Image.new('RGB',(1280,720))
                        for column,(trace,v,condition) in enumerate([(nominal,vn,'无扰动'),(randomized,vr,'扰动试验 1')]):
                            frame=renderer.frame(trace,t,0,v,agent=True)
                            title=f'Agent 候选 {index+1}/{len(directories)} · {condition}'
                            canvas.paste(panel(frame,v,t,title,'原关节 PD 执行记录回放',bounds),(column*640,0))
                        pixels=np.asarray(canvas);writer.append_data(pixels)
                        if preview:preview.append_data(pixels)
                        if index==0 and t==150:canvas.save(a.output/'agent_preview.png')
                    print(f'RENDERED Agent {index+1}/{len(directories)}',flush=True)
                finally:
                    if preview:preview.close()
        finally:writer.close()
        write(a.output/'motion_summary.json',summaries)
        write(a.output/'video_status.json',{'status':'complete','finished_unix':time.time(),
            'files':{str(p.relative_to(a.output)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in videos.glob('*.mp4')}})
    finally:renderer.close();evaluator.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','collect','render'],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',type=Path,default=ROOT/'runs/agent_reference_comparison_20261113')
    p.add_argument('--agent',type=Path,default=ROOT/'runs/trajectory_agent_refinement_20261113')
    p.add_argument('--test',type=Path,default=ROOT/'datasets/supermimic_research_sesame_20261112/test')
    a=p.parse_args();a.output=a.output.resolve()
    if a.stage in ('all','collect'):collect(a)
    if a.stage in ('all','render'):render(a)


if __name__=='__main__':main()
