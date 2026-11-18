"""Portable NPZ dataset reader; all sampling/tensor computations stay on CUDA.

Training uses split='train'. Validation/test are explicit alternatives. Files
contain raw float32 physical states, not PPO running-normalized observations.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch


STATE_KEYS = ('state_root_state', 'state_joint_pos', 'state_joint_vel',
              'state_applied_target', 'state_command_target', 'state_target_history', 'state_previous_action')
CONTEXT_KEYS = ('mass_scale', 'robot_friction', 'effective_contact_friction', 'kp_scale', 'kd_scale',
                'effort_scale', 'speed_scale', 'target_tau_s', 'delay_control_steps')


class SesameMotionDataset:
    """Uniformly sample valid windows; no window crosses an episode boundary.

    State width=77, action width=8, context width=9. Contact/foot data remain
    in source files for auxiliary losses and diagnostics; not packed into state.
    A trainer must recompute motion-tracking rewards and supply reference goals.
    """
    def __init__(self, root, split='train', horizon=8, device='cuda:0', seed=42, cache_episodes=16):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU numerical fallback')
        if split not in ('train','validation','test') or horizon<1 or cache_episodes<1:
            raise ValueError('Invalid split, horizon or cache capacity')
        self.root=Path(root)
        self.horizon=horizon
        self.device=device
        self.cache_capacity=cache_episodes
        self.cache=OrderedDict()
        index=json.loads((self.root/'splits'/f'{split}.json').read_text())
        self.rows=[r for r in index['episodes'] if r['transitions']>=horizon]
        if not self.rows: raise ValueError('No sufficiently long episodes')
        self.window_counts=torch.tensor([r['transitions']-horizon+1 for r in self.rows],dtype=torch.float32,device=device)
        self.rng=torch.Generator(device=device).manual_seed(seed)

    def _episode(self,index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row=self.rows[index]
        with np.load(self.root/row['path'],allow_pickle=False) as file:
            tensors={k:torch.as_tensor(file[k],device=self.device) for k in (*STATE_KEYS,
                'executed_action','command_world','observation','next_observation','terminated','truncated')}
        state=torch.cat([tensors.pop(k).reshape(row['transitions']+1,-1) for k in STATE_KEYS],dim=-1)
        metadata=json.loads((self.root/row['metadata_path']).read_text())
        tensors['state']=state
        tensors['context']=torch.tensor([metadata['domain_parameters'][k] for k in CONTEXT_KEYS],device=self.device)
        self.cache[index]=tensors
        if len(self.cache)>self.cache_capacity: self.cache.popitem(last=False)
        return tensors

    def sample(self,batch_size):
        if batch_size<1: raise ValueError('batch_size must be positive')
        ids=torch.multinomial(self.window_counts,batch_size,replacement=True,generator=self.rng)
        starts=(torch.rand(batch_size,device=self.device,generator=self.rng)*self.window_counts[ids]).long()
        result=[]
        for index,start in zip(ids.tolist(),starts.tolist()):
            episode=self._episode(index)
            row={k:(v if k=='context' else v[start:start+self.horizon+(k=='state')]) for k,v in episode.items()}
            if bool(row['terminated'][:-1].any() | row['truncated'][:-1].any()):
                raise RuntimeError('Invalid window crosses a terminal/reset')
            result.append(row)
        return {**{k:torch.stack([r[k] for r in result]) for k in result[0]},
                'episode_ids':[self.rows[i]['id'] for i in ids.tolist()], 'start_indices':starts}


class SesameReferenceDataset:
    """Sample a current reference frame and K strictly future frames, no wrap.

    Robot retargeting and reference-to-policy goal construction are the future
    trainer's responsibility. These files already use Sesame joint/body order.
    """
    def __init__(self, root, split='train', future_frames=10, device='cuda:0', seed=42):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        if split not in ('train','validation','test') or future_frames<1:
            raise ValueError('Invalid split or future frame count')
        self.root=Path(root)
        self.device=device
        self.horizon=future_frames
        self.rows=[r for r in json.loads((self.root/'reference_index.json').read_text())
                   if r['split']==split and r['frames']>future_frames]
        if not self.rows: raise ValueError('No eligible reference motions')
        self.rng=torch.Generator(device=device).manual_seed(seed)
        self.counts=torch.tensor([r['frames']-future_frames for r in self.rows],device=device,dtype=torch.float32)
        self.cache={}

    def sample(self,batch_size):
        if batch_size<1: raise ValueError('batch_size must be positive')
        ids=torch.multinomial(self.counts,batch_size,replacement=True,generator=self.rng)
        starts=(torch.rand(batch_size,device=self.device,generator=self.rng)*self.counts[ids]).long()
        current,future=[],[]
        for index,start in zip(ids.tolist(),starts.tolist()):
            if index not in self.cache:
                with np.load(self.root/self.rows[index]['path'],allow_pickle=False) as f:
                    self.cache[index]={k:torch.as_tensor(f[k],device=self.device) for k in f.files if k!='fps'}
            episode=self.cache[index]
            current.append({k:v[start] for k,v in episode.items()})
            future.append({k:v[start+1:start+1+self.horizon] for k,v in episode.items()})
        return {'current':{k:torch.stack([r[k] for r in current]) for k in current[0]},
                'future':{k:torch.stack([r[k] for r in future]) for k in future[0]},
                'motion_ids':[self.rows[i]['id'] for i in ids.tolist()], 'frame_indices':starts}
