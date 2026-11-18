"""CUDA math shared exactly by the physical task and differentiable model.

Small Sesame adaptation of SuperMimic, not the authors' G1 implementation.
No model, kinematics, normalization or numerical work is permitted on CPU.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DT = .02
JOINTS = ('R1','R2','L1','L2','R4','R3','L3','L4')
CONTEXT = ('mass_scale','robot_friction','effective_contact_friction','kp_scale','kd_scale',
           'effort_scale','speed_scale','target_tau_s','delay_control_steps')
SUCCESS = {'episode_seconds':9.0, 'velocity_rmse_m_s':.05, 'heading_rms_deg':10.0,
           'joint_position_rmse_rad':.25, 'foot_position_rmse_m':.02, 'root_position_rmse_m':.10,
           'validation_success_rate':.80, 'consecutive_evaluations':2}
REWARD = {'joint_position':1.0, 'joint_velocity':.25, 'root_position':.5,
          'root_orientation':.5, 'root_velocity':1.0, 'foot_position':.5,
          'action_change':-.02, 'joint_limit':-.1, 'termination':-2.0}


def quat_matrix(q):
    q = q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)


def quat_mul(a,b):
    aw,ax,ay,az=a.unbind(-1); bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw),-1)


def rotate(q,vector,inverse=False):
    u=q[...,1:]
    if inverse: u=-u
    u=u.expand_as(vector)
    t=2*torch.linalg.cross(u,vector,dim=-1)
    return vector+q[...,:1]*t+torch.linalg.cross(u,t,dim=-1)


def yaw(q):
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def angle_difference(a,b):
    d=a-b
    return torch.atan2(torch.sin(d),torch.cos(d))


class RobotMath(nn.Module):
    def __init__(self,device='cuda:0'):
        super().__init__()
        if not str(device).startswith('cuda') or not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        self.register_buffer('stand',torch.tensor([135,45,45,135,0,180,0,180],device=device)*math.pi/180)
        limits=torch.tensor([[45,180],[0,135],[0,135],[45,180],[0,180],[0,180],[0,180],[0,180]],device=device)*math.pi/180
        self.register_buffer('low',limits[:,0]); self.register_buffer('high',limits[:,1])
        xml=ET.parse(ROOT/'assets/isaac/sesame/import_source/sesame.urdf').getroot()
        joints={j.get('name'):j for j in xml.findall('joint')}
        def origin(name):
            o=joints[name].find('origin')
            return [float(v) for v in o.get('xyz').split()],[float(v) for v in o.get('rpy').split()]
        def fixed_rotation(rpy):
            # Convert fixed XML metadata; articulated FK itself runs on CUDA.
            r,p,y=rpy; cr,sr=math.cos(r),math.sin(r); cp,sp=math.cos(p),math.sin(p); cy,sy=math.cos(y),math.sin(y)
            return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                    [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
        for label,names in [('upper',('R1','R2','L1','L2')),('lower',('R3','R4','L3','L4'))]:
            positions,rotations=zip(*(origin(n) for n in names))
            self.register_buffer(label+'_offset',torch.tensor(positions,device=device))
            self.register_buffer(label+'_rotation',torch.tensor([fixed_rotation(r) for r in rotations],device=device))
        feet=[origin(n+'_foot_fixed')[0] for n in ('r3','r4','l3','l4')]
        self.register_buffer('foot_offset',torch.tensor(feet,device=device))
        self.register_buffer('delta_scale',torch.tensor([.02,.02,.01]+[.1]*4+[.2]*3+[3.]*3+[.2]*8+[5.]*8,device=device))
        self.register_buffer('state_scale',torch.tensor([1.,1.,.05]+[1.]*4+[.2]*3+[4.]*3+[.58]*8+[10.471975512]*8+[.58]*40+[1.]*8,device=device))
        center=torch.zeros(77,device=device); center[2]=.05; center[3]=1
        center[13:21]=self.stand
        center[29:69]=self.stand.repeat(5)
        self.register_buffer('state_center',center)
        self.register_buffer('context_scale',torch.tensor([1.,1.,1.,1.,1.,1.,1.,.04,2.],device=device))

    def features(self,state):
        value=(state-self.state_center)/self.state_scale
        return torch.cat((torch.zeros_like(value[...,:2]),value[...,2:]),-1).clamp(-20,20)

    def context_features(self,c):
        return c/self.context_scale

    def feet_base(self,q):
        def rz(angle):
            c,s=torch.cos(-angle),torch.sin(-angle); z=torch.zeros_like(c); o=torch.ones_like(c)
            return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(*angle.shape,3,3)
        ru=self.upper_rotation@rz(q[...,:4])
        pl=self.upper_offset+(ru@self.lower_offset[...,None]).squeeze(-1)
        rl=ru@self.lower_rotation@rz(q[...,[5,4,6,7]])
        return pl+(rl@self.foot_offset[...,None]).squeeze(-1)

    def feet_relative_world(self,state):
        return rotate(state[...,None,3:7],self.feet_base(state[...,13:21]))

    def observation(self,state,command=None):
        if command is None:
            command=torch.zeros_like(state[...,:3]); command[...,0]=.1
        gravity=torch.zeros_like(command); gravity[...,2]=-1
        vectors=torch.stack((state[...,7:10],state[...,10:13],gravity,command),-2)
        body=rotate(state[...,None,3:7],vectors,inverse=True)
        return torch.cat((body[...,0,:]*5,body[...,1,:]*.25,body[...,2,:],body[...,3,:]*5,
                          state[...,13:21]-self.stand,state[...,21:29]/10.471975512,state[...,69:77]),-1).clamp(-20,20)

    def controller(self,state,action,context):
        action=action.clamp(-1,1)
        desired=(self.stand+.58*action).clamp(self.low,self.high)
        nominal_delta=10.471975512*DT
        cmd=state[...,37:45]+(desired-state[...,37:45]).clamp(-nominal_delta*context[...,6,None],nominal_delta*context[...,6,None])
        history=torch.cat((cmd[...,None,:],state[...,45:69].reshape(*state.shape[:-1],3,8)[...,:2,:]),-2)
        delay=context[...,8].round().long().clamp(0,2)
        delayed=history.gather(-2,delay[...,None,None].expand(*delay.shape,1,8)).squeeze(-2)
        alpha=1-torch.exp(-DT/context[...,7,None].clamp_min(1e-9))
        filtered=state[...,29:37]+alpha*(delayed-state[...,29:37])
        nominal=state[...,29:37]+(desired-state[...,29:37]).clamp(-nominal_delta,nominal_delta)
        target=torch.where(context[...,7,None]>0,filtered,nominal)
        return torch.cat((target,cmd,history.flatten(-2),action),-1)

    def goal(self,state,reference_sequence):
        ref=reference_sequence
        rotation=quat_matrix(ref[...,3:7])[...,:,:2].flatten(-2)
        return torch.cat(((ref[...,:3]-state[...,None,:3])/.1,rotation,ref[...,7:10]*5,
                          (ref[...,13:21]-self.stand)/.58,ref[...,21:29]/10.471975512,
                          ref[...,29:41]/.1),-1).flatten(-2).clamp(-20,20)

    def termination_probability(self,state):
        upright=1-2*(state[...,4].square()+state[...,5].square())
        ph=torch.sigmoid((.023-state[...,2])/.002)
        pt=torch.sigmoid((.5-upright)/.03)
        pb=torch.sigmoid((state[...,:2].abs().amax(-1)-3.)/.05)
        return 1-(1-ph)*(1-pt)*(1-pb)

    def reward(self,previous,state,reference,termination):
        qerr=(state[...,13:21]-reference[...,13:21]).square().mean(-1)
        dqerr=(state[...,21:29]-reference[...,21:29]).square().mean(-1)
        perr=(state[...,:3]-reference[...,:3]).square().sum(-1)
        verr=(state[...,7:10]-reference[...,7:10]).square().sum(-1)
        dot=(state[...,3:7]*reference[...,3:7]).sum(-1).square().clamp(0,1)
        # sin(angle/2)^2 avoids an acos derivative singularity at exact tracking.
        orientation=1-dot
        ferr=(self.feet_relative_world(state)-reference[...,29:41].reshape(*state.shape[:-1],4,3)).square().sum(-1).mean(-1)
        arate=(state[...,69:77]-previous[...,69:77]).square().mean(-1)
        limit=(torch.relu(self.low-state[...,13:21])+torch.relu(state[...,13:21]-self.high)).square().mean(-1)/.1**2
        rate=(torch.exp(-qerr/.3**2)+.25*torch.exp(-dqerr/5**2)+.5*torch.exp(-perr/.1**2)
              +.5*torch.exp(-orientation/(math.sin(.35/2)**2))+torch.exp(-verr/.08**2)
              +.5*torch.exp(-ferr/.02**2)-.02*arate-.1*limit)
        return DT*rate-2*termination

    def metric_terms(self,previous,state,reference):
        h=angle_difference(yaw(state[:,3:7]),yaw(reference[:,3:7]))
        return torch.stack(((state[:,7:9]-reference[:,7:9]).square().sum(-1),h.square(),
                            (state[:,13:21]-reference[:,13:21]).square().mean(-1),
                            (self.feet_relative_world(state)-reference[:,29:41].reshape(-1,4,3)).square().sum(-1).mean(-1),
                            (state[:,:3]-reference[:,:3]).square().sum(-1),
                            ((state[:,21:29]-previous[:,21:29])/DT).square().mean(-1)),dim=-1)


class MotionLibrary:
    def __init__(self,root,split,device='cuda:0'):
        self.root=Path(root); self.split=split
        self.rows=[r for r in json.loads((self.root/'reference_index.json').read_text()) if r['split']==split]
        states=[]; feet=[]
        for row in self.rows:
            path=self.root/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: raise RuntimeError('Reference changed')
            with np.load(path,allow_pickle=False) as f:
                data={k:torch.as_tensor(f[k],device=device) for k in ('root_pos','root_quat','root_com_lin_vel','root_ang_vel','joint_pos','joint_vel','foot_pos')}
            states.append(torch.cat([data[k] for k in ('root_pos','root_quat','root_com_lin_vel','root_ang_vel','joint_pos','joint_vel')],-1))
            feet.append((data['foot_pos']-data['root_pos'][:,None,:]).flatten(-2))
        self.data=torch.cat((torch.stack(states),torch.stack(feet)),-1)
        self.count,self.frames=self.data.shape[:2]
        stand=torch.tensor([135,45,45,135,0,180,0,180],device=device)*math.pi/180
        self.goal_static=torch.cat((quat_matrix(self.data[...,3:7])[...,:,:2].flatten(-2),
            self.data[...,7:10]*5,(self.data[...,13:21]-stand)/.58,
            self.data[...,21:29]/10.471975512,self.data[...,29:41]/.1),-1)
        self.ahead=torch.arange(1,11,device=device)

    def get(self,ids,frames,offset):
        ref=self.data[ids,frames.clamp(0,self.frames-1)]
        return torch.cat((ref[...,:2]-offset,ref[...,2:]),-1)

    def future(self,ids,frames,offset,k=10):
        ahead=torch.arange(1,k+1,device=self.data.device)
        return self.get(ids[:,None],frames[:,None]+ahead,offset[:,None,:])

    def goal(self,state,ids,frames,offset):
        index=(frames[:,None]+self.ahead).clamp_max(self.frames-1)
        position=self.data[ids[:,None],index,:3]
        absolute=torch.cat((state[:,:2]+offset,state[:,2:3]),-1)
        delta=(position-absolute[:,None,:])/.1
        return torch.cat((delta,self.goal_static[ids[:,None],index]),-1).flatten(-2).clamp(-20,20)
