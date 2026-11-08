#!/usr/bin/env python3
"""Small GPU-only numerical checks for Sesame SAC/branching-DQN and replay.

Does not launch a simulator or train an experimental policy. The parent runner
uses this to validate actual gradient updates, targets and checkpoint recovery
before launching real Isaac training. Refuses CPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Keep probe outputs inside the Sesame workspace")
    if (output / "result.json").exists():
        parser.error("Preserve existing probe results; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from sesame_rl.offpolicy import ReplayBuffer, default_config, load_policy, make_agent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU numerical fallback")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(97142)
    torch.cuda.manual_seed_all(97142)
    started = time.monotonic()
    checks = {}

    # Ring insertions larger than capacity retain precisely the latest data;
    # replay owns detached storage independent of mutable simulator buffers.
    replay = ReplayBuffer(7, device="cuda:0", seed=1)
    obs10 = torch.arange(10, device="cuda:0", dtype=torch.float32)[:, None].expand(-1, 36).clone()
    reward10 = torch.arange(10, device="cuda:0", dtype=torch.float32)
    replay.add(obs10, torch.zeros((10, 8), device="cuda:0"), reward10,
               obs10 + 1, torch.zeros(10, device="cuda:0", dtype=torch.bool))
    assert len(replay) == 7 and replay.position == 3 and replay.total_added == 10
    assert torch.equal(replay.storage["rewards"].sort().values, torch.arange(3, 10, device="cuda:0"))
    obs10.zero_()
    assert bool((replay.storage["obs"][:, 0] >= 3).all())
    restored_replay = ReplayBuffer(7, device="cuda:0", seed=99)
    restored_replay.load_state_dict(replay.state_dict())
    sample, restored_sample = replay.sample(32), restored_replay.sample(32)
    assert all(torch.equal(sample[key], restored_sample[key]) for key in sample)
    checks["replay_ring_ownership_and_rng_restore"] = True

    # Controlled Double-DQN target: online Q=0, target V=2 on every branch.
    # A continuing/time-truncated transition bootstraps once (mean branches),
    # a genuine terminal transition returns its reward without bootstrap.
    diagnostic = make_agent("dqn", config={"seed": 97142, "hidden_dims": [32, 32]})
    with torch.no_grad():
        for parameter in diagnostic.q.parameters():
            parameter.zero_()
        for parameter in diagnostic.target_q.parameters():
            parameter.zero_()
        diagnostic.target_q.value.bias.fill_(2.0)
    test_batch = {
        "obs": torch.zeros((2, 36), device="cuda:0"),
        "actions": torch.zeros((2, 8), device="cuda:0"),
        "rewards": torch.tensor([0.3, 0.4], device="cuda:0"),
        "next_obs": torch.ones((2, 36), device="cuda:0"),
        "terminated": torch.tensor([False, True], device="cuda:0"),
    }
    diagnostic_metrics = diagnostic.update(test_batch)
    expected_target_mean = (0.3 + 0.99 * 2.0 + 0.4) / 2
    assert math.isclose(diagnostic_metrics["target_q_mean"], expected_target_mean, abs_tol=1e-6)
    checks["dqn_mean_branch_target_and_terminal_mask"] = {
        "expected": expected_target_mean, "actual": diagnostic_metrics["target_q_mean"]}

    # A terminal NaN must fail rather than silently use 0*NaN masking.
    invalid_batch = {key: value.clone() for key, value in test_batch.items()}
    invalid_batch["next_obs"][1, 0] = float("nan")
    try:
        diagnostic.update(invalid_batch)
    except ValueError as error:
        assert "Nonfinite next_obs" in str(error)
    else:
        raise AssertionError("Terminal next_obs NaN was not rejected")
    checks["terminal_nan_rejected"] = True
    del diagnostic

    # Exercise both real optimizer paths and verify complete agent-state
    # restoration produces the same next stochastic update and deterministic
    # action. All tensor/model operations below execute on the CUDA device.
    observations = torch.randn((64, 36), device="cuda:0") * 0.2
    next_observations = observations + torch.randn_like(observations) * 0.02
    for algorithm in ("sac", "dqn"):
        config = default_config(algorithm)
        config.update(seed=97143, hidden_dims=[32, 32])
        agent = make_agent(algorithm, config=config)
        actions = agent.act(observations, deterministic=False, epsilon=0.4)
        assert actions.shape == (64, 8) and bool((actions.abs() <= 1.0).all())
        batch = {"obs": observations, "actions": actions,
                 "rewards": torch.full((64,), 0.04, device="cuda:0"),
                 "next_obs": next_observations,
                 "terminated": torch.arange(64, device="cuda:0") % 7 == 0}
        module = agent.actor if algorithm == "sac" else agent.q
        before = [parameter.detach().clone() for parameter in module.parameters()]
        for _ in range(3):
            metrics = agent.update(batch)
        assert all(math.isfinite(float(value)) for value in metrics.values())
        assert any(not torch.equal(saved, now) for saved, now in zip(before, module.parameters(), strict=True))
        if algorithm == "sac":
            assert agent.alpha.item() > 0 and agent.alpha.item() != config["initial_temperature"]
        else:
            assert bool(torch.isin(actions, agent.action_bins).all())
        path = output / f"{algorithm}_probe.pt"
        torch.save(agent.checkpoint({"probe": True}), path)
        deterministic_before = agent.act(observations, deterministic=True)
        inference_agent = load_policy(path)
        deterministic_after = inference_agent.act(observations, deterministic=True)
        reload_error = float((deterministic_before - deterministic_after).abs().max().item())
        assert reload_error == 0.0
        # Inference construction changes global initialization RNG but updates
        # use an agent-local restored generator, preserving stochastic learning.
        expected_metrics = agent.update(batch)
        expected_action = agent.act(observations, deterministic=True)
        resumed = make_agent(algorithm, config=config)
        resumed.load_checkpoint(torch.load(path, map_location="cuda:0", weights_only=True))
        resumed_metrics = resumed.update(batch)
        resumed_action = resumed.act(observations, deterministic=True)
        resume_error = float((expected_action - resumed_action).abs().max().item())
        assert resume_error < 1e-6, (algorithm, resume_error)
        assert all(math.isclose(expected_metrics[key], resumed_metrics[key], rel_tol=1e-6, abs_tol=1e-7)
                   for key in expected_metrics), (algorithm, expected_metrics, resumed_metrics)
        checks[algorithm] = {"finite_actual_gradient_updates": True, "update_count": resumed.updates,
                             "deterministic_reload_max_abs_action_error": reload_error,
                             "resumed_update_max_abs_action_error": resume_error,
                             "metrics": metrics}
        del agent, resumed, inference_agent
    result = {"status": "completed", "passed": True, "device": "cuda:0", "model_computation_on_cpu": False,
              "simulator_started": False, "checks": checks,
              "offpolicy_source_sha256": hashlib.sha256((ROOT / "sesame_rl/offpolicy.py").read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - started}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
