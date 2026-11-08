#!/usr/bin/env python3
"""CUDA-only heading reward semantics, accounting and reset-boundary probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_ppo import disable_git_imports, write_json
from train_ppo_study import require_runtime, validate_frozen_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/heading_reward_probe.json')
    args = parser.parse_args()
    report_path = args.report.resolve()
    if not report_path.is_relative_to(ROOT):
        parser.error('Report must remain inside the Sesame workspace')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = ROOT/'.runtime/heading_probe'
    for key, sub in {'XDG_CACHE_HOME':'cache', 'XDG_CONFIG_HOME':'config',
                     'XDG_DATA_HOME':'data', 'CUDA_CACHE_PATH':'cuda_cache'}.items():
        directory = runtime/sub
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    sys.dont_write_bytecode = True
    report = {'passed':False, 'scope':'16-env CUDA heading formula and real-physics episode accounting',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    app = env = None
    code = 1
    try:
        require_runtime()
        report['frozen_sources'] = validate_frozen_sources()
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU computation fallback')
        torch.set_num_threads(2)
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        app = AppLauncher(headless=True, device='cuda:0', kit_args=(
            f'--portable-root {runtime/"kit"} --/log/file={runtime/"kit.log"} '
            '--/app/settings/persistent=false --/renderer/multiGpu/enabled=false'
        )).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime/'warp_cache')
        from sesame_rl.heading_environment import (
            SesameHeadingEnv, make_heading_env_cfg, heading_reward_rate,
        )
        device = 'cuda:0'
        with torch.inference_mode():
            yaw = torch.tensor([0, math.pi/2, -math.pi/2, math.pi, -math.pi,
                                math.pi+1e-4, -math.pi+1e-4, 2*math.pi+math.pi/2], device=device)
            command = torch.zeros((len(yaw),3), device=device)
            command[:,0] = 0.1
            expected = torch.tensor([0,-0.5,-0.5,-1,-1,-1,-1,-0.5], device=device)
            rates = heading_reward_rate(yaw, command, 0.5)
            assert torch.allclose(rates, expected, atol=1e-6, rtol=0), rates
            assert bool((rates <= 0).all())
            rotated = command.clone(); rotated[:,0] = 0; rotated[:,1] = 0.1
            assert torch.allclose(heading_reward_rate(yaw+math.pi/2, rotated, 0.5), rates, atol=1e-6, rtol=0)
            zero_xy = torch.zeros_like(command); zero_xy[:,2] = 0.2
            assert torch.equal(heading_reward_rate(yaw, zero_xy, 0.5), torch.zeros_like(yaw))
            assert torch.equal(heading_reward_rate(yaw, command, 0.0), torch.zeros_like(yaw))
            report['analytical_cuda_checks'] = {
                'passed':True, 'yaw_rad':yaw.tolist(), 'actual_reward_rates':rates.tolist(),
                'expected_reward_rates':expected.tolist(), 'periodic_wrap_checked':True,
                'rotated_command_direction_checked':True, 'zero_xy_command_disabled':True,
                'zero_weight_disabled':True,
            }

        class AuditedHeadingEnv(SesameHeadingEnv):
            def __init__(self, *a, **kw):
                self._audit_ready = False
                super().__init__(*a, **kw)
                self._audit_ready = True
                self.last_audit = None

            def _get_rewards(self):
                if not self._audit_ready:
                    return super()._get_rewards()
                old_return = self._episode_return.clone()
                old_sums = {k:v.clone() for k,v in self._reward_sums.items()}
                # Independent projected-forward dot command calculation, without
                # calling the heading helper or using the yaw method under test.
                q = self.robot.data.root_quat_w
                forward = torch.stack((1-2*(q[:,2].square()+q[:,3].square()),
                                       2*(q[:,1]*q[:,2]+q[:,0]*q[:,3])), dim=1)
                xy = self._commands_w[:,:2]
                active = xy.square().sum(1)>0
                cosine = (forward*xy).sum(1)/(forward.norm(dim=1)*xy.norm(dim=1)).clamp_min(1e-12)
                heading = torch.where(active, -self.cfg.heading_penalty_scale*(1-cosine), 0.0)*self.step_dt
                value = super()._get_rewards()
                increments = {k:v-old_sums.get(k, torch.zeros_like(v)) for k,v in self._reward_sums.items()}
                assert torch.allclose(increments['heading'], heading, atol=1e-6, rtol=1e-5)
                assert torch.allclose(value, torch.stack(list(increments.values())).sum(0), atol=2e-6, rtol=1e-5)
                assert torch.allclose(self._episode_return, old_return+value, atol=2e-6, rtol=1e-5)
                self.last_audit = {'heading':heading.clone(), 'reward':value.clone(),
                                   'heading_component_error':(increments['heading']-heading).abs().max()}
                return value

        env = AuditedHeadingEnv(make_heading_env_cfg(num_envs=16, seed=42, episode_length_s=1.0))
        report['reward_specification'] = env.heading_reward_metadata()
        report['environment_source_sha256'] = hashlib.sha256((ROOT/'sesame_rl/heading_environment.py').read_bytes()).hexdigest()
        with torch.inference_mode():
            observations, _ = env.reset(seed=7042)
            assert observations['policy'].shape == (16,36) and observations['policy'].is_cuda
            assert env.cfg.action_space == 8 and env.cfg.domain_randomization
            report['initial_domain_readback'] = env.verify_domain_parameters()
            # Explicit yaw overrides give sizeable penalties during the real
            # rollout; these probe poses are not a change to the training reset.
            state = env.robot.data.root_state_w.clone()
            angles = torch.linspace(-math.pi, math.pi, 16, device=device)
            state[:,3] = torch.cos(angles/2); state[:,4:6] = 0; state[:,6] = torch.sin(angles/2)
            state[:,7:] = 0
            env.robot.write_root_pose_to_sim(state[:,:7])
            env.robot.write_root_velocity_to_sim(state[:,7:])
            env.episode_records.clear()
            totals = torch.zeros(16, device=device)
            heading_totals = torch.zeros_like(totals)
            counts = torch.zeros(16, dtype=torch.long, device=device)
            error_max = torch.zeros((), device=device)
            completed = []
            actions = torch.zeros((16,8), device=device)

            def step_and_check():
                nonlocal error_max
                obs, reward, terminated, truncated, info = env.step(actions)
                assert obs['policy'].shape == (16,36) and obs['policy'].is_cuda
                assert bool(torch.isfinite(obs['policy']).all()) and bool(torch.isfinite(reward).all())
                assert torch.equal(reward, env.last_audit['reward'])
                totals.add_(reward); heading_totals.add_(env.last_audit['heading']); counts.add_(1)
                error_max = torch.maximum(error_max, env.last_audit['heading_component_error'])
                ids = torch.nonzero(terminated|truncated, as_tuple=False).flatten()
                if len(ids):
                    records = list(env.episode_records)[-len(ids):]
                    assert [r['env_id'] for r in records] == ids.tolist()
                    actual = torch.tensor([r['return'] for r in records], device=device)
                    assert torch.allclose(actual, totals[ids], atol=2e-5, rtol=1e-5)
                    durations = counts[ids].float()*env.step_dt
                    expected_rate = (heading_totals[ids]/durations).mean()
                    assert torch.allclose(info['log']['RewardRate/heading'], expected_rate, atol=2e-6, rtol=1e-5)
                    assert torch.allclose(info['log']['Episode/return'], totals[ids].mean(), atol=2e-5, rtol=1e-5)
                    completed.extend({'env_id':r['env_id'], 'duration_s':r['duration_s'],
                                      'fell':r['fell'], 'timeout':r['timeout'],
                                      'return':r['return']} for r in records)
                    totals[ids]=0; heading_totals[ids]=0; counts[ids]=0
                assert torch.allclose(env._episode_return, totals, atol=2e-5, rtol=1e-5)
                assert torch.allclose(env._reward_sums['heading'], heading_totals, atol=2e-6, rtol=1e-5)
                return terminated, truncated

            for _ in range(66):
                step_and_check()
            assert len(completed)>=16 and any(r['timeout'] for r in completed)
            # An upside-down airborne state reliably exercises a true failure,
            # independently of the time-limit resets tested above.
            ids = torch.tensor([0], device=device)
            upside = env.robot.data.root_state_w[ids].clone()
            upside[:,2] = 0.20
            upside[:,3:7] = torch.tensor([0, math.sqrt(.5), math.sqrt(.5), 0], device=device)
            upside[:,7:] = 0
            env.robot.write_root_pose_to_sim(upside[:,:7], env_ids=ids)
            env.robot.write_root_velocity_to_sim(upside[:,7:], env_ids=ids)
            terminated, _ = step_and_check()
            assert bool(terminated[0]), 'Forced upside-down state must terminate'
            env.cfg.heading_penalty_scale = 0.0
            step_and_check()
            assert torch.equal(env.last_audit['heading'], torch.zeros_like(totals))
            env.cfg.heading_penalty_scale = 0.5
            subset = torch.tensor([1,7], device=device)
            untouched = torch.tensor([i for i in range(16) if i not in (1,7)], device=device)
            old_returns = env._episode_return[untouched].clone()
            old_heading = env._reward_sums['heading'][untouched].clone()
            env._reset_idx(subset)
            assert torch.equal(env._episode_return[subset], torch.zeros(2,device=device))
            assert torch.equal(env._reward_sums['heading'][subset], torch.zeros(2,device=device))
            assert torch.equal(env._episode_return[untouched], old_returns)
            assert torch.equal(env._reward_sums['heading'][untouched], old_heading)
            report['physics_accounting'] = {
                'passed':True, 'control_steps':68, 'observation_shape':[16,36], 'action_shape':[16,8],
                'full_dr_enabled':True, 'completed_episodes':completed,
                'heading_max_absolute_accounting_error':error_max.item(),
                'returned_reward_equals_sum_of_component_increments':True,
                'episode_return_includes_heading_once':True, 'reward_rate_log_checked':True,
                'timeout_and_failure_terminal_samples_checked':True,
                'automatic_reset_clears_accumulators':True,
                'subset_reset_preserves_other_accumulators':True,
                'zero_scale_real_step_checked':True,
            }
            report['final_domain_readback'] = env.verify_domain_parameters()
        report['passed'] = True
        code = 0
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(report_path, report)
        print('HEADING_REWARD_PROBE='+json.dumps({'passed':report['passed'], 'report':str(report_path)}), flush=True)
        if app is not None:
            def force_exit():
                report['shutdown'] = {'completed':False, 'forced_process_exit':True}
                write_json(report_path, report)
                os._exit(code)
            timer = threading.Timer(15, force_exit); timer.daemon=True; timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            report['shutdown'] = {'completed':True, 'forced_process_exit':False}
            write_json(report_path, report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
