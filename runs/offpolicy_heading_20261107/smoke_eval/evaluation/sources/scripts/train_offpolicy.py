#!/usr/bin/env python3
"""CUDA-only SAC / branching Double-DQN with the frozen heading PPO reward.

Formal runs have the same 2,048,000 real-transition budget as heading PPO.
The five-bin DQN action restriction and different update rules are explicit.
No test-based selection, reward scaling, or modification of frozen sources.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import threading
import time
import traceback

from train_ppo import ROOT, disable_git_imports, jsonable, write_json
from train_heading_ppo import require_runtime, domain_readback

REFERENCE = ROOT / 'runs/ppo_heading_20261105/protocol.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=('sac', 'dqn'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--total-transitions', type=int, default=2048000)
    p.add_argument('--warmup-transitions', type=int, default=65536)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--replay-capacity', type=int, default=262144)
    p.add_argument('--updates-per-vector-step', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    for key in ('num_envs', 'total_transitions', 'warmup_transitions', 'batch_size',
                'replay_capacity', 'updates_per_vector_step'):
        if getattr(args, key) <= 0:
            p.error(key+' must be positive')
    if args.total_transitions % args.num_envs:
        p.error('Transition budget must be divisible by num-envs')
    if args.replay_capacity < max(args.batch_size, args.num_envs):
        p.error('Replay capacity too small')
    if args.warmup_transitions >= args.total_transitions:
        p.error('Warmup must leave steps for learning')
    if not args.smoke:
        actual = (args.num_envs, args.total_transitions, args.warmup_transitions,
                  args.batch_size, args.replay_capacity, args.updates_per_vector_step)
        if actual != (256, 2048000, 65536, 512, 262144, 2) or args.seed not in (42, 43, 44):
            p.error('Formal protocol fixed; custom diagnostic sizes require --smoke')
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(ROOT / 'runs'):
        p.error('run-dir must be inside Sesame runs')
    if (args.run_dir / 'result.json').exists():
        p.error('Existing result; choose a new directory')
    return args


def snapshot_sources(run):
    if sha256(REFERENCE) != REFERENCE.with_suffix('.sha256').read_text().strip():
        raise RuntimeError('Reference heading training protocol changed')
    reference = json.loads(REFERENCE.read_text())
    checks = {}
    for rel, expected in reference['frozen_sources'].items():
        actual = sha256(ROOT / rel)
        if actual != expected:
            raise RuntimeError('Frozen source changed: '+rel)
        checks[rel] = actual
    paths = list(checks) + ['sesame_rl/offpolicy.py', 'sesame_rl/offpolicy_environment.py',
                          'scripts/train_offpolicy.py']
    records = []
    for rel in paths:
        src, dst = ROOT / rel, run / 'sources' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append({'path': rel, 'sha256': sha256(src), 'bytes': src.stat().st_size})
    return records, checks


def main():
    args = parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'result.json').exists():
        raise RuntimeError('Result appeared before lock; refusing overwrite')
    result = dict(status='running', pid=os.getpid(), variant=args.algorithm,
                  algorithm='SAC' if args.algorithm == 'sac' else 'BranchingDoubleDQN',
                  train_seed=args.seed, seed=args.seed, smoke=args.smoke,
                  run_dir=str(run), args=vars(args), reward_version='absolute_heading_v1',
                  heading_penalty_scale=0.5, domain_randomization=True,
                  fresh_from_scratch=True, checkpoint_selection='final only',
                  reward_scale=1.0, git_operations='disabled',
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(run / 'result.json', result)
    runtime = ROOT / '.runtime/offpolicy_heading'
    for key, rel in {'XDG_CACHE_HOME':'cache', 'XDG_CONFIG_HOME':'config',
                     'XDG_DATA_HOME':'data', 'CUDA_CACHE_PATH':'cuda_cache'}.items():
        path = runtime / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    app = env = agent = None
    exit_code = 1
    started = time.monotonic()
    try:
        require_runtime()
        result['sources'], result['frozen_sources'] = snapshot_sources(run)
        result['robot_asset_sha256'] = sha256(ROOT / 'assets/isaac/sesame/sesame.usd')
        disable_git_imports()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
        torch.cuda.set_device('cuda:0')
        torch.set_num_threads(4)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        from isaaclab.app import AppLauncher
        sys.argv = [sys.argv[0]]
        kit_args = f'--portable-root {runtime / "kit"} --/log/file={run / "isaac.log"} --/app/settings/persistent=false --/app/telemetry/enabled=false --/renderer/multiGpu/enabled=false'
        app = AppLauncher(headless=True, device='cuda:0', kit_args=kit_args).app
        import warp as wp
        wp.config.kernel_cache_dir = str(runtime / 'warp_cache')
        from sesame_rl.heading_environment import make_heading_env_cfg
        from sesame_rl.offpolicy_environment import SesameOffPolicyEnv
        from sesame_rl.offpolicy import default_config, make_agent, ReplayBuffer
        cfg = make_heading_env_cfg(num_envs=args.num_envs, seed=args.seed, device='cuda:0',
                                   episode_length_s=10.0, heading_penalty_scale=0.5)
        env = SesameOffPolicyEnv(cfg)
        env.set_domain_randomization(True)
        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict['policy']
        if list(obs.shape) != [args.num_envs, 36] or not obs.is_cuda:
            raise RuntimeError('Expected CUDA [num_envs,36] observation')
        write_json(run / 'environment_config.json', cfg.to_dict())
        result['reward_config'] = env.heading_reward_metadata()
        write_json(run / 'reward_config.json', result['reward_config'])
        domains, validation = domain_readback(env, torch)
        write_json(run / 'domain_samples_initial.json', domains)
        write_json(run / 'domain_validation_initial.json', validation)
        config = default_config(args.algorithm)
        config['seed'] = args.seed
        write_json(run / 'offpolicy_config.json', config)
        training_config = dict(vars(args), device='cuda:0', reward_scale=1.0,
                               dqn_epsilon_initial=1.0, dqn_epsilon_final=0.05,
                               dqn_epsilon_decay_fraction=0.5,
                               observation_normalization='fixed physical scaling from environment',
                               timeout_bootstrap='pre-reset observation; true terminations masked',
                               replay_checkpointed=False)
        write_json(run / 'training_config.json', training_config)
        torch.cuda.reset_peak_memory_stats()
        agent = make_agent(args.algorithm, device='cuda:0', config=config)
        replay = ReplayBuffer(args.replay_capacity, device='cuda:0', seed=args.seed+10001)
        warmup_rng = torch.Generator(device='cuda:0').manual_seed(args.seed+20001)
        result['gpu'] = torch.cuda.get_device_name(0)
        result['versions'] = {}
        for package in ('torch', 'isaaclab', 'isaaclab-rl', 'gymnasium'):
            try:
                result['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result['versions'][package] = 'not in distribution metadata'
        write_json(run / 'result.json', result)
        env.episode_records.clear()
        recent_episodes = deque(maxlen=256)
        counters = dict(completed_episodes=0, true_terminations=0, timeouts=0,
                        timeout_action_snapshot_checks=0, nonterminal_snapshot_checks=0)
        losses = defaultdict(list)
        update_count = 0
        vector_steps = args.total_transitions // args.num_envs
        learning_started = time.monotonic()
        window_started = learning_started
        for step in range(1, vector_steps+1):
            before = (step-1)*args.num_envs
            epsilon = max(0.05, 1.0-0.95*before/(args.total_transitions*0.5))
            with torch.no_grad():
                if before < args.warmup_transitions:
                    if args.algorithm == 'sac':
                        actions = torch.rand((args.num_envs,8), device='cuda:0', generator=warmup_rng)*2-1
                    else:
                        actions = agent.act(obs, deterministic=False, epsilon=1.0)
                else:
                    actions = agent.act(obs, deterministic=False, epsilon=epsilon if args.algorithm=='dqn' else 0.0)
                if not bool(torch.isfinite(actions).all()):
                    raise RuntimeError('Nonfinite policy action')
                returned, reward, terminated, timeout, info = env.step(actions)
                returned_obs = returned['policy']
                snapshot = env.transition_next_obs
                if snapshot is None or snapshot.shape != obs.shape:
                    raise RuntimeError('Missing pre-reset observation snapshot')
                # A zero next state is harmless only for true termination. Never
                # bootstrap through an autoreset, and never multiply NaN by 0.
                replay_next = torch.where(terminated[:, None], torch.zeros_like(snapshot), snapshot)
                if not all(bool(torch.isfinite(x).all()) for x in (obs, reward, replay_next, returned_obs)):
                    raise RuntimeError('Nonfinite replay/returned observation or reward')
                done = terminated | timeout
                if args.smoke or step % 500 == 0:
                    continuing = ~done
                    if bool(continuing.any()):
                        if not torch.equal(snapshot[continuing], returned_obs[continuing]):
                            raise RuntimeError('Continuing next observation differs from snapshot')
                        counters['nonterminal_snapshot_checks'] += int(continuing.sum().item())
                if bool(timeout.any()):
                    if not torch.equal(snapshot[timeout, -8:], actions[timeout].clamp(-1,1)):
                        raise RuntimeError('Timeout snapshot did not retain final action')
                    if not bool((returned_obs[timeout, -8:] == 0).all()):
                        raise RuntimeError('Returned timeout observation is not reset')
                    counters['timeout_action_snapshot_checks'] += int(timeout.sum().item())
                replay.add(obs, actions, reward, replay_next, terminated)
                obs = returned_obs
                counters['true_terminations'] += int(terminated.sum().item())
                counters['timeouts'] += int(timeout.sum().item())
                episodes = list(env.episode_records)
                if len(episodes) == env.episode_records.maxlen:
                    raise RuntimeError('Episode diagnostics buffer overflow')
                env.episode_records.clear()
                if episodes:
                    recent_episodes.extend(episodes)
                    counters['completed_episodes'] += len(episodes)
                    with (run / 'training_episodes.jsonl').open('a') as out:
                        for ep in episodes:
                            out.write(json.dumps(dict(environment_transitions=step*args.num_envs, **ep), allow_nan=False)+'\n')
            if before >= args.warmup_transitions and len(replay) >= args.batch_size:
                for _ in range(args.updates_per_vector_step):
                    log = agent.update(replay.sample(args.batch_size))
                    update_count += 1
                    for key, value in log.items():
                        value = float(value)
                        if not math.isfinite(value):
                            raise RuntimeError('Nonfinite optimizer diagnostic: '+key)
                        losses[key].append(value)
            if step % 32 == 0 or step == vector_steps:
                diagnostics = defaultdict(list)
                for ep in recent_episodes:
                    for key, value in ep.get('study_metrics', {}).items():
                        if isinstance(value, (int, float)):
                            diagnostics[key].append(float(value))
                row = dict(vector_step=step, environment_transitions=step*args.num_envs,
                           optimizer_updates=update_count, replay_size=len(replay),
                           mean_episode_return=statistics.mean(ep['return'] for ep in recent_episodes) if recent_episodes else None,
                           mean_episode_length_steps=statistics.mean(ep['duration_s']/0.02 for ep in recent_episodes) if recent_episodes else None,
                           recent_fall_rate=statistics.mean(float(ep['fell']) for ep in recent_episodes) if recent_episodes else None,
                           losses={key:statistics.mean(values) for key,values in losses.items()},
                           diagnostics={key:statistics.mean(values) for key,values in diagnostics.items()},
                           epsilon=epsilon if args.algorithm=='dqn' else None,
                           window_wall_seconds=time.monotonic()-window_started,
                           training_wall_seconds=time.monotonic()-learning_started,
                           **counters)
                with (run / 'training.jsonl').open('a') as out:
                    out.write(json.dumps(row, allow_nan=False)+'\n')
                write_json(run / 'progress.json', row)
                print(f"{args.algorithm} seed={args.seed} transitions={row['environment_transitions']}/{args.total_transitions} updates={update_count} return={row['mean_episode_return']}", flush=True)
                losses.clear()
                window_started = time.monotonic()
            if step % 2000 == 0 and step < vector_steps:
                torch.save(agent.checkpoint(extra=dict(environment_transitions=step*args.num_envs, seed=args.seed)), run / 'latest.pt')
        torch.cuda.synchronize()
        result['training_wall_seconds'] = time.monotonic()-learning_started
        result['training_torch_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                              scope='Torch only, includes replay; excludes non-Torch simulator allocations')
        metadata = dict(environment_transitions=args.total_transitions, seed=args.seed,
                        reward_version='absolute_heading_v1', heading_penalty_scale=0.5,
                        source_run_dir=str(run), optimizer_updates=update_count)
        final = run / 'final.pt'
        torch.save(agent.checkpoint(extra=metadata), final)
        restored = make_agent(args.algorithm, device='cuda:0', config=config)
        restored.load_checkpoint(torch.load(final, map_location='cuda:0', weights_only=False), load_optimizers=False)
        with torch.no_grad():
            expected = agent.act(obs, deterministic=True)
            actual = restored.act(obs, deterministic=True)
            reload_error = float((actual-expected).abs().max().item())
            if reload_error != 0 or not bool(torch.isfinite(actual).all()):
                raise RuntimeError('Checkpoint reload does not reproduce deterministic actions')
        domains, validation = domain_readback(env, torch)
        write_json(run / 'domain_samples_after_training.json', domains)
        write_json(run / 'domain_validation_after_training.json', validation)
        if counters['timeout_action_snapshot_checks'] != counters['timeouts']:
            raise RuntimeError('Not all timeout transitions validated')
        if args.smoke and counters['timeouts'] == 0:
            raise RuntimeError('Smoke must exercise timeout bootstrap; extend diagnostic rollout')
        result.update(status='completed', checkpoint=str(final), checkpoint_sha256=sha256(final),
                      environment_transitions_total=args.total_transitions, optimizer_updates=update_count,
                      reload_max_abs_action_error=reload_error, transition_validation=counters,
                      completed_training_episodes=counters['completed_episodes'],
                      wall_seconds_before_shutdown=time.monotonic()-started)
        exit_code = 0
    except BaseException as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        result.update(status='interrupted' if exit_code==130 else 'failed', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(run / 'result.json', result)
        print('SESAME_OFFPOLICY_RESULT='+json.dumps(jsonable(result), allow_nan=False), flush=True)
        if app is not None:
            def timeout_exit():
                result['shutdown'] = dict(completed=False, forced_process_exit=True, timeout_seconds=15)
                write_json(run / 'result.json', result)
                os._exit(exit_code)
            timer = threading.Timer(15.0, timeout_exit)
            timer.daemon = True
            timer.start()
            if env is not None:
                env.close()
            app.close(wait_for_replicator=False)
            timer.cancel()
            result['shutdown'] = dict(completed=True, forced_process_exit=False)
            write_json(run / 'result.json', result)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
