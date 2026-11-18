"""Reference tracking with identical physical rewards for PPO and SuperMimic."""
import torch
from .dr_environment import SesameDREnv
from .imitation_core import CONTEXT, quat_mul


class SesameImitationEnv(SesameDREnv):
    def __init__(self,cfg,robot_math,library,**kwargs):
        self.imitation_ready=False
        self.math=robot_math
        self.library=library
        self.evaluation=False
        super().__init__(cfg,**kwargs)
        self.motion_ids=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.phase=torch.zeros_like(self.motion_ids)
        self.offset=torch.zeros((self.num_envs,2),device=self.device)
        self.reference_rng=torch.Generator(device=self.device).manual_seed(int(cfg.seed)+7001)
        self.imitation_ready=True

    def set_library(self,library,evaluation=False):
        self.library=library
        self.evaluation=evaluation

    def reset(self,seed=None,options=None):
        if seed is not None and self.imitation_ready: self.reference_rng.manual_seed(int(seed)+7001)
        return super().reset(seed=seed,options=options)

    def state(self):
        root=self.robot.data.root_state_w.clone()
        root[:,:3]-=self.scene.env_origins
        return torch.cat((root,self.robot.data.joint_pos[:,self.firmware_joint_ids],
                          self.robot.data.joint_vel[:,self.firmware_joint_ids],self._targets,
                          self._command_targets,self._target_history.flatten(1),self._actions),-1)

    def context(self):
        return torch.stack([self.domain_parameters[k] for k in CONTEXT],-1)

    def _pre_physics_step(self,actions):
        if self.imitation_ready:
            self.transition_state=self.state().clone()
            self.transition_context=self.context().clone()
            self.transition_motion_ids=self.motion_ids.clone()
            self.transition_phase=self.phase.clone()
            self.transition_offset=self.offset.clone()
        super()._pre_physics_step(actions)

    def _get_dones(self):
        terminated,timeout=super()._get_dones()
        if self.imitation_ready:
            timeout=(self.phase+1>=self.library.frames-1)&~terminated
        return terminated,timeout

    def _get_rewards(self):
        if not self.imitation_ready: return super()._get_rewards()
        self.next_state=self.state().clone()
        self.next_phase=self.phase+1
        ref=self.library.get(self.motion_ids,self.next_phase,self.offset)
        reward=self.math.reward(self.transition_state,self.next_state,ref,self.reset_terminated.float())
        self.transition_metrics=self.math.metric_terms(self.transition_state,self.next_state,ref)
        self.transition_terminated=self.reset_terminated.clone()
        self.transition_truncated=self.reset_time_outs.clone()
        self.transition_fell=self._fell.clone()
        self.transition_outside=self._outside.clone()
        self.transition_invalid=self._invalid.clone()
        self._episode_return.add_(reward)
        self.phase.add_(1)
        return reward

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if not self.imitation_ready: return
        ids=self._ids(env_ids); count=len(ids)
        if self.evaluation:
            self.motion_ids[ids]=ids%self.library.count
            self.phase[ids]=0
        else:
            self.motion_ids[ids]=torch.randint(self.library.count,(count,),device=self.device,generator=self.reference_rng)
            # Uniform reference-state initialization; deliberately no adaptive
            # reset curriculum in either arm of this small-scale comparison.
            self.phase[ids]=torch.randint(self.library.frames-50,(count,),device=self.device,generator=self.reference_rng)
        original=self.library.data[self.motion_ids[ids],self.phase[ids]]
        self.offset[ids]=original[:,:2]
        ref=self.library.get(self.motion_ids[ids],self.phase[ids],self.offset[ids])
        root=ref[:,:13].clone()
        root[:,:3]+=self.scene.env_origins[ids]
        angle=self.domain_parameters['initial_yaw_rad'][ids]
        yawq=torch.zeros((count,4),device=self.device); yawq[:,0]=torch.cos(angle/2); yawq[:,3]=torch.sin(angle/2)
        root[:,3:7]=quat_mul(yawq,root[:,3:7])
        root[:,7:9]+=self.domain_parameters['initial_xy_velocity_m_s'][ids]
        q=(ref[:,13:21]+self.domain_parameters['initial_joint_offset_rad'][ids]).clamp(self._joint_low,self._joint_high)
        dq=ref[:,21:29]
        simq=self.robot.data.default_joint_pos[ids].clone(); simdq=torch.zeros_like(simq)
        simq[:,self.firmware_joint_ids]=q; simdq[:,self.firmware_joint_ids]=dq
        self.robot.write_root_pose_to_sim(root[:,:7],ids)
        self.robot.write_root_velocity_to_sim(root[:,7:],ids)
        self.robot.write_joint_state_to_sim(simq,simdq,env_ids=ids)
        self._targets[ids]=q; self._command_targets[ids]=q; self._target_history[ids]=q[:,None,:]
        self.robot.set_joint_position_target(q,joint_ids=self.firmware_joint_ids,env_ids=ids)
        self._actions[ids]=0; self._previous_actions[ids]=0
        self._episode_start_x[ids]=root[:,0]; self._episode_start_y[ids]=root[:,1]
