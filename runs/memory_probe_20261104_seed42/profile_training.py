"""Instrument the frozen PPO launcher without modifying algorithm or sources."""
import json
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
sys.path.insert(0, '/home/ywwang/sesame/scripts')
import train_ppo


def phase(name, **extra):
    row = dict(phase=name, pid=os.getpid(), timestamp=time.time(), **extra)
    tmp = OUT / 'phase.tmp'
    tmp.write_text(json.dumps(row) + '\n')
    tmp.replace(OUT / 'phase.json')
    with (OUT / 'phase_events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


original_config = train_ppo.training_config
original_evaluate = train_ppo.evaluate


def profiled_config(args):
    import torch
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    original_learn = OnPolicyRunner.learn

    def learn(self, *a, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        phase('training', allocated_bytes=torch.cuda.memory_allocated(),
              reserved_bytes=torch.cuda.memory_reserved())
        start = time.monotonic()
        try:
            return original_learn(self, *a, **kw)
        finally:
            torch.cuda.synchronize()
            record = dict(pid=os.getpid(), phase='training', seconds=time.monotonic()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          final_allocated_bytes=torch.cuda.memory_allocated(),
                          final_reserved_bytes=torch.cuda.memory_reserved())
            (OUT / 'torch_memory.json').write_text(json.dumps(record, indent=2)+'\n')
            phase('checkpoint_and_reload', **record_without_phase(record))

    OnPolicyRunner.learn = learn
    return original_config(args)


def record_without_phase(record):
    return {k: v for k, v in record.items() if k not in ('phase', 'pid')}


def evaluate(env, wrapper, policy, args, label, torch):
    phase('evaluation_' + label)
    return original_evaluate(env, wrapper, policy, args, label, torch)


if __name__ == '__main__':
    phase('initialization')
    train_ppo.training_config = profiled_config
    train_ppo.evaluate = evaluate
    code = train_ppo.main()
    phase('completed', exit_code=code)
    raise SystemExit(code)
