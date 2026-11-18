#!/usr/bin/env python3
"""CUDA-only numerical checks on the short-run checkpoints before formal runs."""
import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    from sesame_rl.imitation_core import RobotMath, MotionLibrary
    from sesame_rl.supermimic_small import Actor, WorldModel, OfflineReplay
    study = args.study_dir.resolve()
    report = {'cuda': True, 'checks': {}}
    models = {}
    with torch.device('cuda:0'):
        robot = RobotMath()
        for algorithm in ('supermimic', 'ppo'):
            run = study / f'{algorithm}_seed42'
            result = json.loads((run / 'result.json').read_text())
            if result['status'] != 'completed':
                raise RuntimeError(f'{algorithm} smoke not complete')
            initial = torch.load(run / 'initial.pt', map_location='cuda:0', weights_only=False)
            saved = torch.load(run / 'final.pt', map_location='cuda:0', weights_only=False)
            actor = Actor(); actor.load_state_dict(saved['actor'])
            change = sum(float((saved['actor'][k] - initial['actor'][k]).square().sum()) for k in saved['actor'])
            if change <= 0:
                raise AssertionError('Actor did not learn')
            h = torch.randn(32, 10, 36); g = torch.randn(32, 400)
            action, z, lp = actor.sample(h, g)
            exact = torch.distributions.TransformedDistribution(
                torch.distributions.Normal(actor(h, g), actor.log_std.clamp(-3, 1).exp()),
                torch.distributions.TanhTransform()).log_prob(action).sum(-1)
            logprob_error = float((exact - lp).abs().max())
            if logprob_error > 1e-4:
                raise AssertionError(f'Tanh log probability mismatch: {logprob_error}')
            buffer = io.BytesIO(); torch.save(actor.state_dict(), buffer); buffer.seek(0)
            loaded = Actor(); loaded.load_state_dict(torch.load(buffer, map_location='cuda:0', weights_only=True))
            reload_error = float((loaded.action(h, g) - actor.action(h, g)).abs().max())
            if reload_error != 0:
                raise AssertionError('Checkpoint inference changed after reload')
            report['checks'][algorithm] = {'actor_squared_parameter_change': change,
                                         'transformed_log_probability_max_abs': logprob_error,
                                         'checkpoint_reload_action_max_abs': reload_error,
                                         'initial_actor_sha256': saved['initial_actor_sha256']}
            models[algorithm] = saved
        if models['supermimic']['initial_actor_sha256'] != models['ppo']['initial_actor_sha256']:
            raise AssertionError('Unmatched actor initial weights')
        world = WorldModel(robot); world.load_state_dict(models['supermimic']['world'])
        for p in world.parameters():
            p.requires_grad_(False)
        replay = OfflineReplay(Path(models['supermimic']['protocol']['dataset']), 'train', 'cuda:0', 16)
        data = replay.sample(32, 1)
        # Float64 on CUDA avoids finite-difference cancellation in this probe;
        # actual training and the checkpoint inference tests above use float32.
        world = world.double()
        data['state'] = data['state'].double()
        data['context'] = data['context'].double()
        a = data['actions'][:, 0].double().clamp(-.8, .8).detach().requires_grad_(True)
        direction = torch.randn_like(a); direction = direction / direction.norm()
        weight = torch.randn(32, 29) / robot.delta_scale
        def value(action):
            return (world(data['state'], action, data['context'])[:, :29] * weight).mean()
        gradient = torch.autograd.grad(value(a), a)[0]
        analytic = float((gradient * direction).sum())
        eps = .0001
        numeric = float((value(a.detach() + eps * direction) - value(a.detach() - eps * direction)) / (2 * eps))
        relative = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-5)
        if not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise AssertionError('No finite action-to-state gradient through world model')
        if relative > .001 and abs(analytic - numeric) > 1e-8:
            raise AssertionError(f'World derivative mismatch: {analytic}, {numeric}')
        report['checks']['world_action_derivative'] = {'autograd': analytic, 'central_difference': numeric,
                                                      'relative_error': relative, 'gradient_norm': float(gradient.norm())}
    report['passed'] = True
    (study / 'numerical_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
