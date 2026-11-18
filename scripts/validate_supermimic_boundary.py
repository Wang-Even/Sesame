#!/usr/bin/env python3
"""CUDA regression: reference completion must not receive a fall penalty."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    from sesame_rl.imitation_core import RobotMath, MotionLibrary
    from sesame_rl.supermimic_small import Actor, Critic, WorldModel, supermimic_update
    study = args.original_study.resolve()
    previous_source = study / 'supermimic_seed42/sources/sesame_rl/supermimic_small.py'
    current_source = ROOT / 'sesame_rl/supermimic_small.py'
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = definitions(previous_source), definitions(current_source)
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    assert changed == ['supermimic_update'], changed
    # Every PPO helper and its update function are exactly unchanged.
    assert all(old[k] == new[k] for k in ('mlp', 'Actor', 'Critic', 'ppo_update'))
    with torch.device('cuda:0'):
        saved = torch.load(study / 'supermimic_seed42/final.pt', map_location='cuda:0', weights_only=False)
        robot = RobotMath(); library = MotionLibrary(Path(saved['protocol']['dataset']), 'train')
        actor = Actor(); actor.load_state_dict(saved['actor'])
        critic = Critic(); critic.load_state_dict(saved['critic'])
        world = WorldModel(robot); world.load_state_dict(saved['world'])
        batch, horizon = 8, 32
        ids = torch.zeros(batch, dtype=torch.long)
        phase = torch.full((batch,), library.frames - 1 - horizon, dtype=torch.long)
        offset = library.data[ids, phase, :2].clone()
        physical = library.get(ids, phase, offset)[:, :29]
        q = physical[:, 13:21]
        state = torch.cat((physical, q, q, q.repeat(1, 3), torch.zeros(batch, 8)), -1)
        history = robot.observation(state)[:, None].repeat(1, 10, 1)
        context = torch.tensor([1., 1., 1., 1., 1., 1., 1., .02, 0.]).repeat(batch, 1)
        class EndReplay:
            def starts(self, requested, last_frame, requested_horizon):
                assert (requested, requested_horizon, last_frame) == (batch, horizon, 450)
                return {'state': state.clone(), 'history': history.clone(), 'context': context.clone(),
                        'motion': ids.clone(), 'phase': phase.clone(), 'offset': offset.clone()}
        # Force a no-failure scenario to isolate successful reference completion.
        robot.termination_probability = lambda x: x[..., 2] * 0
        observed = []
        original_reward = robot.reward
        def observe_reward(previous, nxt, ref, failure):
            observed.append(failure.detach().clone())
            return original_reward(previous, nxt, ref, failure)
        robot.reward = observe_reward
        actor_opt = torch.optim.Adam(actor.parameters(), lr=.0002)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=.0002)
        metrics = supermimic_update(actor, critic, world, actor_opt, critic_opt, EndReplay(),
                                    library, robot, batch, horizon)
        failure_max = float(torch.stack(observed).abs().max())
        assert failure_max == 0, 'Normal reference completion was charged a fall penalty'
        assert metrics['imagined_survival_end'] == 0, 'Value bootstrapping must stop at reference end'
        assert len(observed) == horizon
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    report = {'passed': True, 'cuda': True, 'horizon': horizon, 'last_reference_frame': 450,
              'normal_completion_failure_penalty_input_max_abs': failure_max,
              'continuation_probability_at_reference_end': metrics['imagined_survival_end'],
              'only_changed_definition': changed, 'ppo_implementation_unchanged': True,
              'old_source_sha256': hashlib.sha256(previous_source.read_bytes()).hexdigest(),
              'corrected_source_sha256': hashlib.sha256(current_source.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
