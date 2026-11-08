#!/usr/bin/env python3
"""Independent CUDA-only world-model checks; no simulator, CPU model or Git."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from train_ppo import ROOT, disable_git_imports


def main():
    output = ROOT / "reports/dyna_world_model_probe.json"
    expected_python = Path("/home/ubuntu/miniforge3/envs/env_isaaclab/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError("Run this probe with the env_isaaclab Python executable")
    if Path(os.environ.get("CONDA_PREFIX", "")).resolve() != expected_python.parent.parent:
        raise RuntimeError("The env_isaaclab activation hooks are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing probe report: {output}")
    result = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Synthetic known-linear-transition CUDA probe; no Isaac Sim and no claim about real robot model accuracy",
              "git_operations": "disabled", "cpu_model_computation": False, "tests": {}}
    started = time.monotonic()
    exit_code = 1
    try:
        disable_git_imports()
        import torch
        from sesame_rl.dyna_world_model import DynaWorldModel
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        data_rng = torch.Generator(device=device).manual_seed(9107)

        def cuda_rng_state():
            # Torch's state serialization API returns bytes on CPU; comparison
            # is explicitly transferred to CUDA, never computed on CPU.
            return torch.cuda.get_rng_state(device).to(device)

        def equal(first, second):
            assert first.is_cuda and second.is_cuda
            return bool(torch.equal(first, second))

        def generate(n):
            history = torch.randn((n, 4, 36), device=device, generator=data_rng) * 0.15
            history[:, :, 6:9] = 0.0
            history[:, :, 8] = -1.0
            history[:, :, 9:12] = 0.0
            history[:, :, 9] = 0.5
            history[:, :, 28:] = torch.rand((n, 4, 8), device=device, generator=data_rng) - 0.5
            actions = torch.rand((n, 8), device=device, generator=data_rng) - 0.5
            delta = torch.zeros((n, 28), device=device)
            delta[:, :6] = 0.08 * actions[:, :6] - 0.10 * history[:, -1, :6]
            delta[:, 12:20] = 0.05 * actions - 0.10 * history[:, -1, 12:20]
            delta[:, 20:28] = 0.07 * actions - 0.15 * history[:, -1, 20:28]
            next_obs = torch.cat((history[:, -1, :28] + delta, actions), dim=1)
            reward = 0.025 + 0.004 * actions.mean(dim=1) + 0.002 * history[:, -1, 0]
            dones = torch.zeros(n, dtype=torch.bool, device=device)
            ids = torch.arange(n, device=device) % 256
            return history, actions, next_obs, reward, dones, ids

        original_rng = cuda_rng_state()
        config = {"capacity": 8192, "fit_steps": 24, "batch_size": 512,
                  "validation_batch_size": 819, "min_fit_calls": 2}
        model = DynaWorldModel(device, seed=132, config=config)
        assert equal(original_rng, cuda_rng_state())
        result["tests"]["initialization_preserves_global_cuda_rng"] = True
        data = generate(8192)
        added = model.add(*data)
        expected_train = int((data[-1] % 10 != 0).sum().item())
        expected_validation = int((data[-1] % 10 == 0).sum().item())
        assert added["train_added"] == expected_train and added["validation_added"] == expected_validation
        assert bool(model.input_moments.count == expected_train)

        # Validation-only additions cannot affect optimizer inputs or moments.
        moments_before = {name: value.clone() for name, value in model.input_moments.state_dict().items()}
        targets_before = {name: value.clone() for name, value in model.target_moments.state_dict().items()}
        train_x = model.train_replay.x[:model.train_replay.size].clone()
        train_y = model.train_replay.y[:model.train_replay.size].clone()
        train_size, train_position = model.train_replay.size, model.train_replay.position
        held_out = list(generate(256))
        held_out[-1].fill_(10)
        validation_add = model.add(*held_out)
        assert validation_add["train_added"] == 0 and validation_add["validation_added"] == 256
        assert model.train_replay.size == train_size and model.train_replay.position == train_position
        assert equal(train_x, model.train_replay.x[:train_size]) and equal(train_y, model.train_replay.y[:train_size])
        assert all(equal(value, model.input_moments.state_dict()[name]) for name, value in moments_before.items())
        assert all(equal(value, model.target_moments.state_dict()[name]) for name, value in targets_before.items())
        result["tests"]["validation_excluded_from_training_replay_and_normalizers"] = True

        # Huge post-reset jumps and failure rewards must never enter the model.
        terminal = list(generate(32))
        terminal[2].fill_(1000.0)
        terminal[3].fill_(-2.0)
        terminal[4].fill_(True)
        counts_before = (model.real_train_samples_seen, model.real_validation_samples_seen,
                         model.train_replay.size, model.validation_replay.size)
        excluded = model.add(*terminal)
        assert excluded == {"train_added": 0, "validation_added": 0, "excluded_done_or_nonfinite": 32}
        assert counts_before == (model.real_train_samples_seen, model.real_validation_samples_seen,
                                 model.train_replay.size, model.validation_replay.size)
        assert all(equal(value, model.input_moments.state_dict()[name]) for name, value in moments_before.items())
        assert all(equal(value, model.target_moments.state_dict()[name]) for name, value in targets_before.items())
        result["tests"]["terminal_reset_jumps_excluded"] = True

        fits = []
        for _ in range(12):
            fits.append(model.fit())
        initial_error = fits[0]["validation_pre_delta_mse_normalized"]
        final_error = fits[-1]["validation_post_delta_mse_normalized"]
        assert final_error < initial_error * 0.5, (initial_error, final_error)
        assert fits[-1]["validation_to_persistence_mse_ratio"] < 0.5
        assert fits[-1]["validation_reward_mae"] < 0.003
        assert fits[-1]["ready"]
        result["tests"]["known_linear_transition_learned_and_heldout_error_decreased"] = True
        assert all(parameter.is_cuda for parameter in model.models.parameters())
        assert all(value.is_cuda for value in model.input_moments.buffers())
        assert all(value.is_cuda for value in model.target_moments.buffers())
        assert all(value.is_cuda for entry in model.optimizer.state.values() for value in entry.values()
                   if isinstance(value, torch.Tensor))
        result["tests"]["model_normalizers_and_optimizer_tensors_on_cuda"] = True

        starts = model.sample_starts(1024)
        actions = torch.rand((1024, 8), device=device, generator=data_rng) - 0.5
        prediction = model.imagine(starts, actions)
        assert all(value.is_cuda for value in prediction)
        assert equal(prediction[0][:, 28:], actions.clamp(-1, 1))
        accepted = int(prediction[2].sum().item())
        assert accepted > 0
        assert bool((prediction[1] <= 0.057 + 1e-7).all())
        large_actions = actions * 5.0
        clipped_prediction = model.imagine(starts, large_actions)
        assert equal(clipped_prediction[0][:, 28:], large_actions.clamp(-1, 1))
        result["tests"]["next_observation_action_slots_exact_and_reward_upper_bound"] = True
        ood = starts.clone()
        ood[:, :, 0] = 100.0
        rejected = model.imagine(ood, actions)
        assert not bool(rejected[2].any())
        assert bool(torch.isfinite(rejected[0]).all()) and bool(torch.isfinite(rejected[1]).all())
        result["tests"]["out_of_support_inputs_rejected_with_finite_fallback"] = True
        assert equal(original_rng, cuda_rng_state())
        result["tests"]["fit_add_imagine_and_sample_preserve_global_cuda_rng"] = True

        # Real torch serialization, in-memory only; no extra checkpoint artifact.
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = DynaWorldModel(device, seed=999, config=config)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
        restored_prediction = restored.imagine(starts, actions)
        assert all(equal(first, second) for first, second in zip(prediction, restored_prediction))
        assert restored.train_replay.size == restored.validation_replay.size == 0
        assert restored.fit_calls == model.fit_calls
        assert equal(restored.input_moments.count, model.input_moments.count)
        assert equal(restored.target_moments.count, model.target_moments.count)
        assert equal(restored.generator.get_state().to(device), model.generator.get_state().to(device))
        assert all(value.is_cuda for entry in restored.optimizer.state.values() for value in entry.values()
                   if isinstance(value, torch.Tensor))
        assert equal(original_rng, cuda_rng_state())
        result["tests"]["serialized_model_optimizer_normalizers_rng_reload_exact"] = True
        result.update(status="passed", gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      effective_model_config=model.config, fits=fits,
                      initial_validation_normalized_delta_mse=initial_error,
                      final_validation_normalized_delta_mse=final_error,
                      heldout_error_final_over_initial=final_error / initial_error,
                      imagination_candidates=1024, imagination_accepted=accepted,
                      train_samples_seen=model.real_train_samples_seen,
                      validation_samples_seen=model.real_validation_samples_seen,
                      excluded_terminal_samples=32, replay_saved=False,
                      reload_prediction_max_absolute_error=0.0)
        exit_code = 0
    except Exception as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    result["source_sha256"] = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in ("sesame_rl/dyna_world_model.py", "scripts/probe_dyna_world_model.py")
    }
    result["wall_seconds"] = time.monotonic() - started
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output), "tests": result["tests"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
