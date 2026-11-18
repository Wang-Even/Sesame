"""Small differentiable-dynamics learner and matched PPO implementation.

Uses analytic observation/reward/actuator heads. The learned physical state
transition is differentiated through 32 imagined steps; this is not Dyna-PPO.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .imitation_core import CONTEXT


def mlp(widths,zero_last=False):
    layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        layer=nn.Linear(a,b)
        layers.append(layer)
        if i<len(widths)-2: layers.append(nn.ELU())
    if zero_last:
        nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=mlp([400,256,128,32])
        self.decoder=mlp([360+32,256,128,8])
        nn.init.normal_(self.decoder[-1].weight,std=.01); nn.init.zeros_(self.decoder[-1].bias)
        self.log_std=nn.Parameter(torch.full((8,),-.69314718056))

    def forward(self,history,goal):
        return self.decoder(torch.cat((history.flatten(-2),torch.tanh(self.encoder(goal))),-1))

    def action(self,history,goal): return torch.tanh(self(history,goal))

    def sample(self,history,goal):
        mean=self(history,goal); std=self.log_std.clamp(-3,1).exp()
        z=mean+torch.randn_like(mean)*std
        action=z.tanh()
        return action,z,self.logprob(mean,z)

    def logprob(self,mean,z):
        logstd=self.log_std.clamp(-3,1)
        lp=-.5*((z-mean)/logstd.exp()).square()-logstd-.9189385332046727
        logjac=2*(.69314718056-z-torch.nn.functional.softplus(-2*z))
        return (lp-logjac).sum(-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__(); self.net=mlp([77+400+9,128,128,1])
    def forward(self,state_features,goal,context_features):
        return self.net(torch.cat((state_features,goal,context_features),-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self,robot_math):
        super().__init__()
        # RobotMath contains buffers only, no trainable weights.
        self.math=robot_math
        self.net=mlp([77+8+9,256,256,256,29],zero_last=True)

    def forward(self,state,action,context):
        control=self.math.controller(state,action,context)
        delta=self.net(torch.cat((self.math.features(state),action.clamp(-1,1),self.math.context_features(context)),-1))
        physical=state[...,:29]+delta*self.math.delta_scale
        q=physical[...,3:7]
        q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        return torch.cat((physical[...,:3],q,physical[...,7:],control),-1)


class OfflineReplay:
    """Load only the frozen TRAIN/VALIDATION split, never test for model fitting."""
    def __init__(self,root,split,device,max_episodes=None):
        from .motion_dataset import STATE_KEYS
        root=Path(root)
        rows=json.loads((root/'splits'/f'{split}.json').read_text())['episodes']
        if max_episodes is not None: rows=rows[:max_episodes]
        self.states=torch.zeros((len(rows),501,77),device=device)
        self.actions=torch.zeros((len(rows),500,8),device=device)
        self.contexts=torch.empty((len(rows),9),device=device)
        self.lengths=torch.tensor([r['transitions'] for r in rows],device=device,dtype=torch.long)
        self.failed=torch.zeros(len(rows),device=device,dtype=torch.bool)
        for i,row in enumerate(rows):
            with np.load(root/row['path'],allow_pickle=False) as f:
                pieces=[torch.as_tensor(f[k],device=device).reshape(row['transitions']+1,-1) for k in STATE_KEYS]
                state=torch.cat(pieces,-1)
                self.states[i,:len(state)]=state
                self.states[i,len(state):]=state[-1]
                self.actions[i,:row['transitions']]=torch.as_tensor(f['executed_action'],device=device)
                self.failed[i]=torch.as_tensor(f['terminated'],device=device).any()
            meta=json.loads((root/row['metadata_path']).read_text())
            self.contexts[i]=torch.tensor([meta['domain_parameters'][k] for k in CONTEXT],device=device)
        self.transitions=int(self.lengths.sum().item())
        self.failure_ids=self.failed.nonzero().squeeze(-1)
        self.device=device

    def sample(self,batch,horizon=8):
        weights=(self.lengths-horizon+1).clamp_min(1).float()
        ids=torch.multinomial(weights,batch,replacement=True)
        starts=(torch.rand(batch,device=self.device)*weights[ids]).long()
        # Ensure the sparse terminal events receive model supervision too.
        nf=batch//10
        if len(self.failure_ids) and nf:
            ids[:nf]=self.failure_ids[torch.randint(len(self.failure_ids),(nf,),device=self.device)]
            starts[:nf]=(self.lengths[ids[:nf]]-horizon).clamp_min(0)
        idx=starts[:,None]+torch.arange(horizon,device=self.device)
        valid=idx<self.lengths[ids,None]
        safe=idx.clamp_max(499)
        target=self.states[ids[:,None],(idx+1).clamp_max(500)]
        return {'state':self.states[ids,starts], 'actions':self.actions[ids[:,None],safe],
                'target':target,'context':self.contexts[ids], 'mask':valid.float()}


class OnlineReplay:
    """Time-major GPU ring with explicit pre-reset next states and episode ids."""
    def __init__(self,num_envs,device,capacity=512):
        self.n=num_envs; self.capacity=capacity; self.device=device; self.size=0; self.position=0
        self.buffers={}

    @torch.no_grad()
    def add(self,**values):
        if not self.buffers:
            self.buffers={k:torch.empty((self.capacity,*v.shape),dtype=v.dtype,device=self.device) for k,v in values.items()}
        for k,v in values.items(): self.buffers[k][self.position].copy_(v)
        self.position=(self.position+1)%self.capacity; self.size=min(self.size+1,self.capacity)

    def _indices(self,batch,horizon=1):
        count=self.size-horizon+1
        if count<1: raise RuntimeError('Insufficient replay')
        start=torch.randint(count,(batch,),device=self.device)
        oldest=(self.position-self.size)%self.capacity
        time=(start+oldest)%self.capacity
        env=torch.randint(self.n,(batch,),device=self.device)
        return time,env

    def sample(self,batch,horizon=8):
        time,env=self._indices(batch,horizon)
        idx=(time[:,None]+torch.arange(horizon,device=self.device))%self.capacity
        b=self.buffers
        epi=b['episode'][idx,env[:,None]]
        mask=(epi==epi[:,:1]).float()
        return {'state':b['state'][time,env], 'actions':b['action'][idx,env[:,None]],
                'target':b['next_state'][idx,env[:,None]],'context':b['context'][time,env], 'mask':mask}

    def starts(self,batch,last_frame,horizon=32):
        # Reference clip boundaries are known. Never imagine a cyclic wrap.
        candidates=[]
        while sum(len(v[0]) for v in candidates)<batch:
            time,env=self._indices(batch)
            keep=self.buffers['phase'][time,env]+horizon<=last_frame
            candidates.append((time[keep],env[keep]))
            if len(candidates)>50: raise RuntimeError('No eligible imagination starts')
        time=torch.cat([x[0] for x in candidates])[:batch]
        env=torch.cat([x[1] for x in candidates])[:batch]
        return {k:v[time,env].clone() for k,v in self.buffers.items()}


def fit_world(model,optimizer,replay,batch=256,horizon=8):
    data=replay.sample(batch,horizon)
    prediction=data['state']
    errors=[]
    for t in range(horizon):
        prediction=model(prediction,data['actions'][:,t],data['context'])
        residual=(prediction[:,:29]-data['target'][:,t,:29])/model.math.delta_scale
        # Robust to contact impulses without silently discarding failures.
        per=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none').mean(-1)
        errors.append(per*data['mask'][:,t])
    loss=torch.stack(errors,1).sum()/data['mask'].sum().clamp_min(1)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite world-model loss')
    optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def world_error(model,replay,batch=256,horizon=8):
    data=replay.sample(batch,horizon); p=data['state']; error=[]
    for t in range(horizon):
        p=model(p,data['actions'][:,t],data['context'])
        error.append(((p[:,:29]-data['target'][:,t,:29])/model.math.delta_scale).square().mean(-1)*data['mask'][:,t])
    e=torch.stack(error,1)
    valid=data['mask'][:,-1].bool()
    return {'normalized_rmse':float((e.sum()/data['mask'].sum().clamp_min(1)).sqrt().item()),
            'root_position_rmse_m':float((p[valid,:3]-data['target'][valid,-1,:3]).square().mean().sqrt().item()),
            'joint_position_rmse_rad':float((p[valid,13:21]-data['target'][valid,-1,13:21]).square().mean().sqrt().item())}


def ppo_update(actor,critic,actor_opt,critic_opt,robot_math,rollout,gamma=.99,lam=.95):
    # All values include the same reference goals and privileged dynamics context.
    with torch.no_grad():
        rewards=rollout['reward']; values=rollout['value']; nxt=rollout['next_value']; done=rollout['done']
        adv=torch.zeros_like(rewards); gae=torch.zeros_like(rewards[0])
        for t in reversed(range(len(rewards))):
            delta=rewards[t]+gamma*(~done[t]).float()*nxt[t]-values[t]
            gae=delta+gamma*lam*(~done[t]).float()*gae; adv[t]=gae
        returns=adv+values
        advantage=(adv-adv.mean())/adv.std().clamp_min(1e-6)
        data={k:v.flatten(0,1) for k,v in rollout.items()}
        returns=returns.flatten(); advantage=advantage.flatten()
    actor_losses=[]; value_losses=[]; kls=[]
    total=len(returns); mini=min(total,1024)
    for epoch in range(5):
        permutation=torch.randperm(total,device=returns.device)
        for ids in permutation.split(mini):
            mean=actor(data['history'][ids],data['goal'][ids])
            lp=actor.logprob(mean,data['z'][ids]); ratio=(lp-data['logprob'][ids]).exp()
            surrogate=torch.minimum(ratio*advantage[ids],ratio.clamp(.8,1.2)*advantage[ids])
            entropy=(actor.log_std.clamp(-3,1)+1.4189385332).sum()
            aloss=-surrogate.mean()-.005*entropy
            pred=critic(robot_math.features(data['state'][ids]),data['goal'][ids],robot_math.context_features(data['context'][ids]))
            clipped=data['value'][ids]+(pred-data['value'][ids]).clamp(-.2,.2)
            vloss=.5*torch.maximum((pred-returns[ids]).square(),(clipped-returns[ids]).square()).mean()
            actor_opt.zero_grad(set_to_none=True); aloss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),1.0); actor_opt.step()
            critic_opt.zero_grad(set_to_none=True); vloss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1.0); critic_opt.step()
            actor_losses.append(aloss.detach()); value_losses.append(vloss.detach())
            kls.append(((ratio-1)-(lp-data['logprob'][ids])).mean().detach())
    return {'actor_loss':float(torch.stack(actor_losses).mean()),'critic_loss':float(torch.stack(value_losses).mean()),
            'approx_kl':float(torch.stack(kls).mean()),'action_std':float(actor.log_std.exp().mean())}


def supermimic_update(actor,critic,world,actor_opt,critic_opt,replay,library,robot_math,batch=256,horizon=32):
    initial=replay.starts(batch,library.frames-1,horizon)
    state=initial['state']; history=initial['history']; context=initial['context']
    mids=initial['motion']; phase=initial['phase']; offset=initial['offset']
    for p in world.parameters(): p.requires_grad_(False)
    for p in critic.parameters(): p.requires_grad_(False)
    rewards=[]; dones=[]; values=[]; states=[]; goals=[]
    survival=torch.ones(batch,device=state.device)
    objective=torch.zeros_like(survival)
    for t in range(horizon):
        goal=library.goal(state,mids,phase+t,offset)
        action=actor.action(history,goal)
        states.append(state.detach()); goals.append(goal.detach())
        values.append(critic(robot_math.features(state),goal,robot_math.context_features(context)).detach())
        nxt=world(state,action,context)
        failure=robot_math.termination_probability(nxt)
        terminal=torch.where(phase+t+1>=library.frames-1,torch.ones_like(failure),failure)
        ref=library.get(mids,phase+t+1,offset)
        # Reaching the reference end stops bootstrapping but is not a fall.
        reward=robot_math.reward(state,nxt,ref,failure)
        objective=objective+(.99**t)*survival*reward
        survival=survival*(1-terminal)
        rewards.append(reward.detach()); dones.append(terminal.detach())
        state=nxt
        obs=robot_math.observation(state)
        history=torch.cat((history[:,1:],obs[:,None]),1)
    goal=library.goal(state,mids,phase+horizon,offset)
    bootstrap=critic(robot_math.features(state),goal,robot_math.context_features(context))
    objective=objective+(.99**horizon)*survival*bootstrap
    loss=-objective.mean()
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite imagined objective')
    actor_opt.zero_grad(set_to_none=True); loss.backward()
    norm=nn.utils.clip_grad_norm_(actor.parameters(),1.0)
    for p in actor.parameters():
        if p.grad is not None: p.grad.clamp_(-.1,.1)
    actor_opt.step()
    for p in world.parameters(): p.requires_grad_(True)
    for p in critic.parameters(): p.requires_grad_(True)
    with torch.no_grad():
        targets=[]; continuation=bootstrap.detach()
        for t in reversed(range(horizon)):
            vnext=bootstrap.detach() if t==horizon-1 else values[t+1]
            continuation=rewards[t]+.99*(1-dones[t])*(.05*vnext+.95*continuation)
            targets.append(continuation)
        target=torch.stack(list(reversed(targets))).flatten()
        state_flat=torch.stack(states).flatten(0,1); goal_flat=torch.stack(goals).flatten(0,1)
        ctx=context.expand(horizon,-1,-1).flatten(0,1)
    losses=[]
    for _ in range(5):
        order=torch.randperm(len(target),device=target.device)
        for ids in order.chunk(4):
            value=critic(robot_math.features(state_flat[ids]),goal_flat[ids],robot_math.context_features(ctx[ids]))
            vl=(value-target[ids]).square().mean()
            critic_opt.zero_grad(set_to_none=True); vl.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1.0); critic_opt.step()
            losses.append(vl.detach())
    return {'actor_loss':float(loss.detach()),'critic_loss':float(torch.stack(losses).mean()),
            'actor_grad_norm':float(norm),'imagined_reward_mean':float(torch.stack(rewards).mean()),
            'imagined_survival_end':float(survival.detach().mean()),'imagined_transitions':batch*horizon}
