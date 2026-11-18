"""Fixed full-clip evaluation; early failure receives zero for remaining frames."""
import math
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from sesame_official.task import make_cfg
from supermimic.tasks.task_spec import build_actor_obs_td


class Evaluator:
    def __init__(self, motion_file, seed=91042):
        import numpy as np
        with np.load(motion_file) as data:
            n = len(data["motion_ends"])
        cfg = make_cfg(motion_file, n, seed)
        cfg.commands["motion"].pose_range = {}
        cfg.commands["motion"].velocity_range = {}
        cfg.commands["motion"].joint_position_range = (0.,0.)
        cfg.events["foot_friction"].params["ranges"] = (.8,.8)
        cfg.events["base_com"].params["ranges"] = {i:(0.,0.) for i in range(3)}
        cfg.events["encoder_bias"].params["bias_range"] = (0.,0.)
        self.env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device="cuda:0"))

    @torch.no_grad()
    def run(self, actor, representation, sim_context, history_length, steps=450, dynamics=None, loss_weights=None):
        # Evaluation does not consume training RNG or update normalization.
        rng = torch.cuda.get_rng_state()
        mode = actor.training
        actor.eval()
        env = self.env
        raw = env.unwrapped
        env.reset()
        c = raw.command_manager.get_term("motion")
        starts = torch.cat([c.motion.motion_ends.new_zeros(1),c.motion.motion_ends[:-1]])
        ids = torch.arange(env.num_envs,device="cuda:0")
        c.time_steps[:] = starts
        c._write_reference_state_to_sim(ids,c.body_pos_w[:,0],c.body_quat_w[:,0],
            c.body_lin_vel_w[:,0],c.body_ang_vel_w[:,0],c.joint_pos,c.joint_vel)
        raw.sim.forward(); c.update_relative_body_poses()
        raw.episode_length_buf.zero_()
        raw.action_manager.reset(ids)
        obs = env.get_observations()
        history = obs["actor"].unsqueeze(1).repeat(1,history_length,1)
        alive = torch.ones(env.num_envs,device="cuda:0",dtype=torch.bool)
        total = torch.zeros(env.num_envs,device="cuda:0")
        survived = torch.zeros_like(total)
        sums = torch.zeros(env.num_envs,5,device="cuda:0")
        prediction_sum = torch.zeros((),device="cuda:0")
        prediction_count = torch.zeros((),device="cuda:0")
        feet = [c.cfg.body_names.index(n) for n in ("r3_foot","r4_foot","l3_foot","l4_foot")]
        for t in range(steps):
            state = obs["full_state"]
            aux = representation.extract_aux_state(state)
            _, goal = representation.compute_wm_goal(c.motion,c.time_steps,aux)
            td = build_actor_obs_td({"motion":goal}, history)
            actions = actor(td)
            if dynamics is not None:
                if t%8==0:
                    predicted_rep=representation.represent(state)
                    predicted_aux=aux
                    prediction_valid=alive.clone()
                    prediction_context=sim_context.read(raw)
                predicted_aux,predicted_rep=dynamics(predicted_rep,actions,predicted_aux,prediction_context)
            # Log current state against current reference before auto-reset.
            ref = c.motion.body_state(c.time_steps)
            d = raw.scene["robot"].data
            rp = d.root_link_pos_w - raw.scene.env_origins
            rq = d.root_link_quat_w
            def yaw(q):
                w,x,y,z = q.unbind(-1)
                return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            angle = yaw(rq)-yaw(ref[1][:,0])
            angle = torch.atan2(angle.sin(),angle.cos())
            footpos = c.robot_body_pos_w[:,feet]-raw.scene.env_origins[:,None,:]
            metric = torch.stack([
                (d.root_link_lin_vel_w[:,:2]-ref[2][:,0,:2]).square().sum(-1),
                angle.square(), (d.joint_pos-c.joint_pos).square().mean(-1),
                (footpos-ref[0][:,feet]).square().sum(-1).mean(-1),
                (rp-ref[0][:,0]).square().sum(-1)],dim=-1)
            sums += metric * alive[:,None]
            survived += alive
            obs, reward, done, extras = env.step(actions)
            total += reward * alive
            # The fixed 450-step horizon finishes before the 451-frame clip-end timeout.
            alive &= ~raw.termination_manager.terminated
            if dynamics is not None:
                prediction_valid &= alive & ~done.bool()
                if t%8==7:
                    truth=obs["full_state"]
                    pred_blocks=representation.loss_blocks(predicted_rep,predicted_aux)
                    true_blocks=representation.loss_blocks(representation.represent(truth),representation.extract_aux_state(truth))
                    squared=torch.cat([(pred_blocks[k]-true_blocks[k]).square()*loss_weights[k] for k in pred_blocks],dim=-1).mean(-1)
                    prediction_sum += squared[prediction_valid].sum()
                    prediction_count += prediction_valid.sum()
            history = torch.cat([history[:,1:],obs["actor"].unsqueeze(1)],dim=1)
        rms = (sums/survived.clamp_min(1)[:,None]).sqrt()
        limits = torch.tensor([.05,10*math.pi/180,.25,.02,.10],device="cuda:0")
        success = alive & (rms<=limits).all(-1) & (survived==steps)
        result = {"episodes":env.num_envs, "steps_per_episode":steps,
            "normalized_return":float((total/(steps*.02*5.)).mean()),
            "return":float(total.mean()), "success_rate":float(success.float().mean()),
            "fall_rate":float((~alive).float().mean()),
            "velocity_rmse_m_s":float(rms[:,0].mean()),
            "heading_rms_deg":float(rms[:,1].mean()*180/math.pi),
            "joint_position_rmse_rad":float(rms[:,2].mean()),
            "foot_position_rmse_m":float(rms[:,3].mean()),
            "root_position_rmse_m":float(rms[:,4].mean()),
            "episode_returns":total.tolist()}
        if dynamics is not None:
            result["prediction_error_8_step"] = (float(prediction_sum/prediction_count)
                                                 if prediction_count.item()>0 else None)
            result["prediction_windows_8_step"] = int(prediction_count.item())
        actor.train(mode)
        torch.cuda.set_rng_state(rng)
        return result

    def close(self):
        self.env.close()
