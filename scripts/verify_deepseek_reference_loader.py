"""Verify the completed DeepSeek collection using the official CUDA loader."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/supermimic_official_5efbf114/src')]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.runtime/research_cache'))
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    read = lambda p: json.loads(p.read_text())
    status = read(run/'status.json')
    assert status['status'] == 'complete' and status['accepted_count'] == 32
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4)
    from sesame_trajectory_agent.validation import ExecutionEngine
    from mjlab.tasks.tracking.mdp.commands import MotionLoader
    started = time.monotonic()
    protocol = read(run/'protocol.json')
    engine = ExecutionEngine(ROOT/protocol['config']['source'], protocol['config'])
    try:
        loader = MotionLoader(str(run/'accepted/train'), engine.fk, device='cuda:0')
        states = loader.body_state(loader.motion_ends-1)
        assert len(loader.motion_ends) == 32
        assert all(bool(torch.isfinite(s).all()) for s in states)
        rounded = defaultdict(list)
        for entry in read(run/'accepted/manifest.json')['entries']:
            path = Path(entry['validation']).parent/'candidate_reference.npz'
            with np.load(path, allow_pickle=False) as data:
                digest = hashlib.sha256()
                for key in ('root_pos_w','root_quat_w','joint_pos'):
                    value = torch.as_tensor(data[key], device='cuda:0')
                    quantized = torch.round(value*100000).to(torch.int32)
                    digest.update(key.encode())
                    digest.update(quantized.cpu().numpy().tobytes())
                rounded[digest.hexdigest()].append(entry['id'])
        result = dict(status='complete', device='cuda:0', official_motion_loader=True,
                      clips=32, end_frame_states_finite=True,
                      rounded_pose_resolution=1e-5,
                      rounded_pose_unique_clips=len(rounded),
                      rounded_pose_duplicate_groups=[v for v in rounded.values() if len(v)>1],
                      rounded_pose_note='Diagnostic diversity audit only; original admission rules unchanged. Meter/radian pose channels rounded on CUDA; no model or numerical computation on CPU.',
                      elapsed_s=time.monotonic()-started)
        (run/'loader_verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
