"""CUDA motion-program API and MimicAgent-inspired structural checks for Sesame."""
import math
import torch
from .motion_program import run_program
from .trajectories import angular_velocity, derivative


class MotionProgramRuntime:
    def __init__(self, engine, config):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
        self.engine=engine; self.config=config
        self.frames=round(config["duration_s"]*config["fps"])+1
        self.qhome=engine.robot.data.default_joint_pos[0].clone()
        feet=self.feet(self.qhome[None])[0]
        height=(engine.foot_radii-feet[:,2]).mean()
        self.home={"joint_pos":self.qhome,"feet_b":feet,"base_height":height,
                   "speed":config["target_speed_m_s"],"duration":config["duration_s"],
                   "foot_radius":engine.foot_radii.clone()}
        self.ik_error=0.

    def feet(self,q):
        if q.ndim!=2 or q.shape[1]!=8 or len(q)>self.frames*9:
            raise ValueError("feet expects [N,8] joint positions")
        n=len(q); pos=q.new_zeros(n,3); quat=q.new_zeros(n,4);quat[:,0]=1
        body=self.engine.fk(pos,quat,q,pos,pos,torch.zeros_like(q))[0]
        return body[:,self.engine.feet].clone()

    def ik(self, targets):
        """Unconstrained DLS IK: no joint clipping, target projection, or root correction.

        Residual is measured and rejected by validation, never silently accepted.
        Each frame starts at the same model home pose, selecting the same branch.
        """
        if targets.shape!=(self.frames,4,3): raise ValueError("ik expects [N,4,3] body-frame foot targets")
        n=len(targets); q=self.qhome.expand(n,-1).clone()
        eye=torch.eye(8,device=q.device); h=1e-4
        for _ in range(60):
            variants=torch.cat((q[:,None],q[:,None]+eye[None]*h),dim=1)
            f=self.feet(variants.reshape(-1,8)).reshape(n,9,12)
            jac=((f[:,1:]-f[:,:1])/h).transpose(1,2)
            residual=targets.reshape(n,12)-f[:,0]
            delta=torch.linalg.solve(jac.transpose(1,2)@jac+1e-6*eye,
                                      (jac.transpose(1,2)@residual[:,:,None]))[:,:,0]
            q=q+.7*delta
        self.ik_error=max(self.ik_error,float((self.feet(q)-targets).norm(dim=-1).max()))
        return q

    def metadata(self):
        from sesame_research.robot import URDF
        import xml.etree.ElementTree as ET
        tree=ET.parse(URDF)
        return {"joint_names":self.engine.joints,"foot_order":["right_front","right_rear","left_front","left_rear"],
                "home":{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in self.home.items()},
                "kinematic_joints":[ET.tostring(x,encoding="unicode") for x in tree.findall("joint")],
                "physical_constraints":self.engine.constraints(),
                "morphology":"Eight joints, two per leg. NOT a 12-DoF Go2; generic sagittal 3-joint IK is invalid.",
                "joint_mirror_pairs":[["R1","L1"],["R2","L2"],["R3","L3"],["R4","L4"]],
                "mirror_note":"Mirrored angles sum to pi, so symmetric offsets have opposite signs."}

    @torch.no_grad()
    def generate(self,source):
        n=self.frames; t=torch.arange(n,device="cuda:0",dtype=torch.float32)/self.config["fps"]
        def tensor(x):
            y=torch.tensor(x,device="cuda:0",dtype=torch.float32)
            if y.numel()>n*32: raise ValueError("Literal tensor too large")
            return y
        def stack(values,dim=-1):
            if len(values)>32: raise ValueError("Too many stacked columns")
            return torch.stack(values,dim=dim)
        api={k:getattr(torch,k) for k in ("sin","cos","sqrt","abs","exp","tanh","floor",
                                         "where","zeros_like","ones_like","amin","amax")}
        api.update(tensor=tensor,stack=stack,copy=lambda x:x.clone(),feet=self.feet,ik=self.ik)
        self.ik_error=0.
        result=run_program(source,t,{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in self.home.items()},api)
        mandatory={"root_pos_w","root_rpy","contacts"}
        if not isinstance(result,dict) or not mandatory<=set(result) or set(result)-mandatory not in ({"joint_pos"},{"foot_pos_b"}):
            raise ValueError("Return root_pos_w, root_rpy, contacts and exactly one of joint_pos / foot_pos_b")
        q=result.get("joint_pos")
        if q is None:q=self.ik(result["foot_pos_b"])
        for name,value,shape in [("root_pos_w",result["root_pos_w"],(n,3)),
                                ("root_rpy",result["root_rpy"],(n,3)),
                                ("joint_pos",q,(n,8)),("contacts",result["contacts"],(n,4))]:
            if not isinstance(value,torch.Tensor) or not value.is_cuda or value.shape!=shape:
                raise ValueError(f"{name} must be a CUDA tensor of shape {shape}")
            if not bool(torch.isfinite(value).all()): raise ValueError(f"Non-finite {name}")
        contacts=result["contacts"]
        if not bool(((contacts==0)|(contacts==1)).all()):raise ValueError("contacts must be boolean/0/1")
        roll,pitch,yaw=(result["root_rpy"]/2).unbind(-1)
        cr,sr=roll.cos(),roll.sin();cp,sp=pitch.cos(),pitch.sin();cy,sy=yaw.cos(),yaw.sin()
        quat=torch.stack((cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy),-1)
        pos=result["root_pos_w"];dt=1/self.config["fps"]
        ref={"root_pos_w":pos,"root_quat_w":quat,"root_lin_vel_w":derivative(pos,dt),
             "root_ang_vel_w":angular_velocity(quat,dt),"joint_pos":q,"joint_vel":derivative(q,dt)}
        return ref,contacts.bool(),{"generation_clipping":False,"automatic_root_correction":False,
                                   "source":"agent-authored phase motion program; no source-clip resampling",
                                   "ik_target_error_max_m":self.ik_error}

    @torch.no_grad()
    def check(self,ref,contacts,generation):
        base,body=self.engine.check_kinematics(ref)
        if body is None:return base,None
        gates=self.config["gates"];height=ref["root_pos_w"][:,2]
        gap=body[:,self.engine.feet,2]-self.engine.foot_radii
        ground=(gap.abs()<=.003)
        support=float(ground.any(-1).float().mean())
        # Paper's 0.1 m minimum is for Go2; Sesame standing root is ~0.047 m.
        # Reuse the existing Sesame minimum, record the scaled maximum explicitly.
        checks={"base_height_envelope":bool(((height>=gates["root_height_min_m"]) &
                        (height<=2*self.home["base_height"])).all()),
                "non_aerial_support_half_cycle":support>=.5,
                "ik_reachability":generation["ik_target_error_max_m"]<=.001,
                "declared_contacts_match_ground":bool((~contacts | ground).all())
                        and bool(contacts.any(-1).float().mean()>=.5),
                "fixed_forward_goal":bool((ref["root_pos_w"][:,0]-torch.arange(len(height),device=height.device)*.02*self.config["target_speed_m_s"]).abs().max()<1e-4)
                        and bool(ref["root_pos_w"][:,1].abs().max()<1e-4)}
        reasons=base["reasons"]+[k for k,v in checks.items() if not v]
        metrics=base.get("metrics",{}) | {"base_height_min_m":float(height.min()),"base_height_max_m":float(height.max()),
                 "base_height_allowed_m":[gates["root_height_min_m"],float(2*self.home["base_height"])],
                 "support_fraction":support,"ik_target_error_max_m":generation["ik_target_error_max_m"],
                 "foot_gap_min_m_by_leg":gap.amin(0).tolist(),"foot_gap_max_m_by_leg":gap.amax(0).tolist(),
                 "joint_min_rad":ref["joint_pos"].amin(0).tolist(),"joint_max_rad":ref["joint_pos"].amax(0).tolist()}
        phases=[]
        for i in range(4):
            a=i*(len(height)-1)//4;b=(i+1)*(len(height)-1)//4+1
            phases.append({"phase":[i/4,(i+1)/4],"root_start":ref["root_pos_w"][a].tolist(),
                "root_end":ref["root_pos_w"][b-1].tolist(),"foot_ground_fraction":ground[a:b].float().mean(0).tolist(),
                "declared_contact_fraction":contacts[a:b].float().mean(0).tolist()})
        return {"passed":not reasons,"reasons":reasons,"checks":checks,"metrics":metrics,"motion_readout":phases},body
