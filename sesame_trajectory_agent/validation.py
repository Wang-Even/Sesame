"""CUDA kinematic checks and free-running MuJoCo execution, without pose playback.

This evaluates one explicit controller (joint-position targets + native DC-motor
PD). Failure is not proof that no other controller could execute the motion.
"""
import math
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_mul
from sesame_research.task import make_cfg
from .trajectories import FIELDS, derivative, yaw
from .decisions import decide

FEET = ("r3_foot", "r4_foot", "l3_foot", "l4_foot")


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


class ExecutionEngine:
    def __init__(self, source, config, trials=1, randomized=False):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
        self.config = config; self.randomized = randomized; self.trials = trials
        cfg = make_cfg(source, trials, config["seed"] + (1000 if randomized else 0),
                       physics_dt_s=config.get("physics_dt_s",.002))
        # No reference command can teleport/reset a robot during verification.
        cfg.commands = {}; cfg.observations = {}; cfg.rewards = {}; cfg.terminations = {}
        cfg.metrics = {}; cfg.curriculum = {}; cfg.auto_reset = False
        cfg.scene.sensors = (ContactSensorCfg(name="agent_ground_contact",
            primary=ContactMatch(mode="geom", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce"),)
        if not randomized:
            cfg.events["foot_friction"].params["ranges"] = (.8,.8)
            cfg.events["base_com"].params["ranges"] = {i:(0.,0.) for i in range(3)}
            cfg.events["encoder_bias"].params["bias_range"] = (0.,0.)
        self.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        self.robot = self.env.scene["robot"]
        self.joints = list(self.robot.joint_names); self.bodies = list(self.robot.body_names)
        self.feet = [self.bodies.index(n) for n in FEET]
        self.contact = self.env.scene["agent_ground_contact"]
        names = [n.split("/")[-1] for n in self.contact.primary_names]
        # Foot marker bodies have no collision geometry. The actual foot spheres
        # belong to the shank bodies: match their named geoms, not marker bodies.
        foot_geoms = [n+"_collision" for n in FEET]
        self.contact_feet = [names.index(n) for n in foot_geoms]
        self.contact_nonfeet = [i for i,n in enumerate(names) if n not in foot_geoms]
        local_geom_ids,_ = self.robot.find_geoms(foot_geoms,preserve_order=True)
        global_geom_ids = self.robot.indexing.geom_ids[local_geom_ids].tolist()
        self.foot_radii = torch.as_tensor(self.env.sim.mj_model.geom_size[global_geom_ids,0],device="cuda:0",dtype=torch.float32)
        self.fk = ForwardKinematics(self.env.sim.mj_model, self.robot.indexing.body_ids,
            self.robot.indexing.joint_q_adr, torch.arange(len(self.bodies),device="cuda:0"), device="cuda:0")
        self.stall_torque = float(cfg.scene.entities["robot"].articulation.actuators[0].effort_limit)
        self.joint_noise = torch.zeros(trials,8,device="cuda:0")
        self.yaw_noise = torch.zeros(trials,device="cuda:0")
        self.velocity_noise = torch.zeros(trials,3,device="cuda:0")
        if randomized:
            g = torch.Generator(device="cuda:0").manual_seed(config["seed"] + 2000)
            self.joint_noise.uniform_(-.02,.02,generator=g)
            self.yaw_noise.uniform_(-5*math.pi/180,5*math.pi/180,generator=g)
            self.velocity_noise[:,:2].uniform_(-.01,.01,generator=g)

    def constraints(self):
        d = self.robot.data
        return {"joint_names":self.joints,"body_names":self.bodies,
                "joint_limits_rad":d.joint_pos_limits[0].tolist(),
                "controller":("analytic position/heading feedback on reference hip amplitudes, then native mjlab DC-motor PD and encoder bias" if self.config.get("execution_feedback") else "native mjlab DC-motor PD, q_target = q_reference[t+1] - encoder_bias"),
                "control_dt_s":self.env.step_dt,"physics_dt_s":self.env.physics_dt,
                "stall_torque_Nm":self.stall_torque,
                "execution_feedback":self.config.get("execution_feedback"),
                "mean_speed_measurement":self.config.get("mean_speed_measurement","control_samples"),
                "self_collision_checked":False,
                "contact_metrics":"ground contact; foot-origin velocity during detected contact, sampled at 50 Hz"}

    def context(self):
        # Save actual sampled model properties. Startup randomization is shared
        # by every candidate in this run, so proposals see the same trials.
        model = self.env.sim.model
        def to_list(x):
            return x.tolist() if isinstance(x,torch.Tensor) else str(x)
        return {"randomized":self.randomized,"seed":self.env.cfg.seed,
                "initial_joint_noise_rad":self.joint_noise.tolist(),
                "initial_yaw_noise_rad":self.yaw_noise.tolist(),
                "initial_velocity_noise_m_s":self.velocity_noise.tolist(),
                "encoder_bias_rad":self.robot.data.encoder_bias.tolist(),
                "geom_friction":to_list(model.geom_friction),
                "body_ipos":to_list(model.body_ipos)}

    @torch.no_grad()
    def check_kinematics(self, ref):
        d = self.robot.data; gate = self.config["gates"]
        finite = all(bool(torch.isfinite(x).all()) for x in ref.values())
        if not finite: return {"passed":False,"reasons":["nonfinite_reference"]},None
        norm_error = (ref["root_quat_w"].norm(dim=-1)-1).abs().max()
        body_pos,*_ = self.fk(*(ref[k] for k in ("root_pos_w","root_quat_w","joint_pos","root_lin_vel_w","root_ang_vel_w","joint_vel")))
        stats = {"joint_velocity_max_rad_s":scalar(ref["joint_vel"].abs().max()),
                 "joint_acceleration_max_rad_s2":scalar(derivative(ref["joint_vel"],.02).abs().max()),
                 "foot_surface_min_z_m":scalar((body_pos[:,self.feet,2]-self.foot_radii).min()),
                 "quaternion_norm_max_error":scalar(norm_error)}
        reasons = []
        if norm_error > 1e-4: reasons.append("reference_quaternion")
        for k in ("joint_velocity_max_rad_s","joint_acceleration_max_rad_s2"):
            if stats[k] > gate[k]: reasons.append(f"reference_exceeds:{k}")
        if stats["foot_surface_min_z_m"] < gate["foot_surface_min_z_m"]:
            reasons.append("reference_foot_sphere_below_ground_tolerance")
        return {"passed":not reasons,"reasons":reasons,"metrics":stats},body_pos

    def snapshot(self):
        d = self.robot.data
        return {"root_pos_w":d.root_link_pos_w-self.env.scene.env_origins,
                "root_quat_w":d.root_link_quat_w,"root_lin_vel_w":d.root_link_lin_vel_w,
                "root_ang_vel_w":d.root_link_ang_vel_w,"joint_pos":d.joint_pos,
                "joint_vel":d.joint_vel,
                "body_pos_w":d.body_link_pos_w-self.env.scene.env_origins[:,None,:],
                "body_lin_vel_w":d.body_link_lin_vel_w}

    @torch.no_grad()
    def execute(self, ref, ref_body):
        env = self.env; d = self.robot.data; n = self.trials; gate = self.config["gates"]
        env.reset()
        ids = torch.arange(n, device="cuda:0")
        root = torch.cat([ref[k][0].expand(n,-1) for k in ("root_pos_w","root_quat_w","root_lin_vel_w","root_ang_vel_w")],dim=-1).clone()
        root[:,:3] += env.scene.env_origins
        noise_q = root.new_zeros(n,4); noise_q[:,0] = (self.yaw_noise/2).cos(); noise_q[:,3] = (self.yaw_noise/2).sin()
        root[:,3:7] = quat_mul(noise_q, root[:,3:7]); root[:,7:10] += self.velocity_noise
        q = ref["joint_pos"][0].expand(n,-1) + self.joint_noise
        # The only pose write in a rollout is this initialization.
        self.robot.write_root_state_to_sim(root,env_ids=ids)
        self.robot.write_joint_state_to_sim(q,ref["joint_vel"][0].expand(n,-1),env_ids=ids)
        self.robot.reset(env_ids=ids); env.action_manager.reset(ids)
        env.sim.forward(); env.sim.sense()
        snapshots = {k:[v.clone()] for k,v in self.snapshot().items()}
        actions=[]; targets=[]; torques=[]; contacts=[]; forces=[]
        steps = len(ref["joint_pos"])-1
        feedback=self.config.get("execution_feedback")
        hip_ids=torch.tensor([0,2,4,6],device="cuda:0")
        hip_center=ref["joint_pos"][:-1,hip_ids].mean(0)
        previous_pos=snapshots["root_pos_w"][0].clone()
        filtered_velocity=root[:,7:10].clone()
        for t in range(steps):
            target=ref["joint_pos"][t+1].expand(n,-1).clone()
            if feedback:
                # Analytic execution teacher. This changes motor commands only;
                # no reference/root state is rewritten and no command is clipped.
                pos=d.root_link_pos_w-env.scene.env_origins
                if t:
                    alpha=feedback["velocity_filter_alpha"]
                    filtered_velocity=(1-alpha)*filtered_velocity+alpha*(pos-previous_pos)/env.step_dt
                previous_pos=pos.clone()
                angle=yaw(d.root_link_quat_w)
                advance=1+feedback["position_gain"]*(ref["root_pos_w"][t,0]-pos[:,0])+feedback["velocity_gain"]*(self.config["target_speed_m_s"]-filtered_velocity[:,0])
                turn=feedback["heading_gain"]*(angle+feedback["lateral_gain"]*pos[:,1])+feedback["yaw_rate_gain"]*d.root_link_ang_vel_w[:,2]
                side=target.new_tensor([-1.,-1.,1.,1.])
                scale=advance[:,None]+turn[:,None]*side
                target[:,hip_ids]=hip_center+(target[:,hip_ids]-hip_center)*scale
            action = (target-d.default_joint_pos)/.58
            _,_,terminated,truncated,_ = env.step(action)
            expected_target = target - d.encoder_bias
            if not torch.allclose(d.joint_pos_target,expected_target,atol=1e-6,rtol=0):
                raise RuntimeError("Action-to-joint-target mapping changed; cannot certify this controller")
            if bool((terminated|truncated).any()):
                raise RuntimeError("Unexpected reset/termination in free-running verification")
            for k,v in self.snapshot().items(): snapshots[k].append(v.clone())
            actions.append(action.clone()); targets.append(d.joint_pos_target.clone())
            torques.append(d.qfrc_actuator.clone())
            contacts.append(self.contact.data.found.clone())
            forces.append(self.contact.data.force.clone())
            if not all(bool(torch.isfinite(v[-1]).all()) for v in snapshots.values()): break
        trace = {k:torch.stack(v) for k,v in snapshots.items()}
        trace.update(action=torch.stack(actions), executed_joint_target=torch.stack(targets),
                     joint_torque_Nm=torch.stack(torques), contact_found=torch.stack(contacts),
                     contact_force_N=torch.stack(forces))
        T = len(actions); completed = T == steps
        state_finite = torch.stack([torch.isfinite(x).reshape(len(x),n,-1).all(0).all(-1) for x in trace.values()]).all(0)
        actual = trace["root_pos_w"][1:]
        qactual = trace["root_quat_w"][1:]
        angle = yaw(qactual); angle = torch.atan2(angle.sin(),angle.cos())
        tilt = torch.acos((1-2*(qactual[:,:,1].square()+qactual[:,:,2].square())).clamp(-1,1))
        velocity = trace["root_lin_vel_w"][1:,:,:2]
        target_v = velocity.new_tensor([self.config["target_speed_m_s"],0.])
        foot_velocity = trace["body_lin_vel_w"][1:,:,self.feet,:2]
        contact_on = (trace["contact_found"]>0) & (trace["contact_force_N"].norm(dim=-1)>.01)
        foot_contact = contact_on[:,:,self.contact_feet]
        contact_count = foot_contact.sum((0,2))
        slip = ((foot_velocity.square().sum(-1)*foot_contact).sum((0,2))/contact_count.clamp_min(1)).sqrt()
        qtrace = trace["joint_pos"]
        if T > 1:
            rate = ((trace["action"][1:]-trace["action"][:-1])/.02).square().mean((0,2)).sqrt()
        else: rate = velocity.new_full((n,),float("nan"))
        displacement_velocity=(trace["root_pos_w"][-1]-trace["root_pos_w"][0])/(T*env.step_dt)
        mean_speed=(displacement_velocity[:,0] if self.config.get("mean_speed_measurement")=="displacement" else velocity[:,:,0].mean(0))
        arrays = {
            "velocity_rmse_m_s":(velocity-target_v).square().sum(-1).mean(0).sqrt(),
            "mean_forward_speed_m_s":mean_speed,
            "mean_speed_error_m_s":(mean_speed-self.config["target_speed_m_s"]).abs(),
            "control_sample_mean_forward_speed_m_s":velocity[:,:,0].mean(0),
            "displacement_mean_forward_speed_m_s":displacement_velocity[:,0],
            "heading_rms_deg":angle.square().mean(0).sqrt()*180/math.pi,
            "heading_max_deg":angle.abs().amax(0)*180/math.pi,
            "lateral_displacement_max_m":actual[:,:,1].abs().amax(0),
            "joint_tracking_rmse_rad":(qtrace[1:]-ref["joint_pos"][1:T+1,None]).square().mean((0,2)).sqrt(),
            "root_tracking_rmse_m":(actual-ref["root_pos_w"][1:T+1,None]).square().sum(-1).mean(0).sqrt(),
            "contact_foot_speed_rms_m_s":slip,
            "nonfoot_contact_fraction":contact_on[:,:,self.contact_nonfeet].any(-1).float().mean(0),
            "torque_near_stall_fraction":(trace["joint_torque_Nm"].abs()>=.95*self.stall_torque).float().mean((0,2)),
            "joint_torque_peak_Nm":trace["joint_torque_Nm"].abs().amax((0,2)),
            "action_rate_rms_per_s":rate,
        }
        # Include the initialized state in fall detection; do not hide a failure
        # by averaging only over an early surviving prefix.
        root_low = trace["root_pos_w"][:,:,2].amin(0) < gate["root_height_min_m"]
        init_q = trace["root_quat_w"][0]
        init_tilt = torch.acos((1-2*(init_q[:,1].square()+init_q[:,2].square())).clamp(-1,1))
        fell = root_low | (tilt.amax(0)*180/math.pi>gate["tilt_max_deg"]) | (init_tilt*180/math.pi>gate["tilt_max_deg"])
        results=[]
        for i in range(n):
            m = {k:scalar(v[i]) for k,v in arrays.items()}
            m.update(finite=bool(state_finite[i]),completed=completed,fell=bool(fell[i]),steps=T,
                     contact_observations=int(contact_count[i]))
            if not contact_count[i]: m["contact_foot_speed_rms_m_s"] = None
            reasons = decide(m,gate,self.config.get("diagnostic_only_metrics",[]))
            results.append({"trial":i,"randomized":self.randomized,"passed":not reasons,"reasons":reasons,"metrics":m})
        return results, trace

    def close(self):
        self.env.close()
