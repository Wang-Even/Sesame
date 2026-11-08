"""Small CUDA-only one-step world model for the local Sesame Dyna-PPO study.

Only REAL, nonterminal transitions may be passed to ``add``. History is oldest
to newest; its last frame is the current raw 36-D policy observation. The caller
must fill every history slot with the reset observation after each done. Raw
here means before the actor's observation normalizer, retaining environment
feature scaling. This model changes neither the environment nor its reward.

Environment IDs divisible by ten are permanently held out from model fitting,
normalizer updates, and imagination starts. Validation calibrates acceptance;
it is not the study's independent policy evaluation. There is no terminal model:
the caller must restrict imagination to ONE step and retain real PPO updates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DynaWorldModelConfig:
    members: int = 3
    hidden_size: int = 128
    capacity: int = 65536  # Total across train and validation replay.
    batch_size: int = 512
    fit_steps: int = 24  # Each step updates all three independently sampled members.
    validation_batch_size: int = 2048
    learning_rate: float = 1.0e-3
    grad_norm: float = 10.0
    min_train_samples: int = 2048
    min_validation_samples: int = 256
    min_fit_calls: int = 5
    input_std_floor: float = 0.05
    delta_std_floor: float = 0.01
    reward_std_floor: float = 0.005
    max_validation_persistence_ratio: float = 0.95
    max_validation_reward_mae: float = 0.01
    input_max_sigma: float = 6.0
    output_max_sigma: float = 8.0
    uncertainty_quantile: float = 0.95
    uncertainty_min: float = 0.05
    uncertainty_max: float = 0.50
    reward_max: float = 0.057  # 0.02 * (1.5 + 1.0 + 0.35); no invented lower bound.
    min_predicted_upright: float = 0.60
    gravity_norm_tolerance: float = 0.15


class _Moments(nn.Module):
    """Streaming train-only moments; all reductions and buffers live on CUDA."""

    def __init__(self, width: int, device: torch.device):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("mean", torch.zeros(width, device=device))
        self.register_buffer("m2", torch.zeros(width, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        if values.shape[0] == 0:
            return
        variance, mean = torch.var_mean(values, dim=0, unbiased=False)
        n = values.shape[0]
        total = self.count + n
        difference = mean - self.mean
        self.m2.add_(variance * n + difference.square() * (self.count * n / total).float())
        self.mean.add_(difference * (n / total).float())
        self.count.copy_(total)

    def scale(self, floor: torch.Tensor | float):
        variance = self.m2 / self.count.clamp_min(1).float()
        return torch.maximum(variance.clamp_min(0).sqrt(), torch.as_tensor(floor, device=self.mean.device))


class _Replay:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.x = torch.empty((capacity, 152), device=device)
        self.y = torch.empty((capacity, 29), device=device)

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor):
        x, y = x[-self.capacity:], y[-self.capacity:]
        n = x.shape[0]
        if not n:
            return
        indices = (torch.arange(n, device=x.device) + self.position) % self.capacity
        self.x[indices], self.y[indices] = x, y
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)


class DynaWorldModel:
    """History/action -> ensemble of next-observation deltas and physical reward.

    ``fit`` returns JSON-compatible scalar metadata. ``imagine`` returns CUDA
    tensors (next_obs, reward, valid_mask, uncertainty), shapes (N,36), (N,),
    (N,), (N,). Invalid predictions use a finite persistence fallback and zero
    reward, but MUST still be excluded using valid_mask. Uncertainty is ensemble
    RMS disagreement in train-standardized delta/reward units, not probability.
    ``state_dict`` includes optimizer, moments, RNG and counters, but no replay.
    """

    def __init__(self, device: str | torch.device, seed: int, config: dict | DynaWorldModelConfig | None = None):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Dyna world-model computation requires CUDA; no CPU fallback")
        self.device = torch.device("cuda", self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.cfg = config if isinstance(config, DynaWorldModelConfig) else DynaWorldModelConfig(**(config or {}))
        self.config, self.seed = asdict(self.cfg), int(seed)
        if self.cfg.members != 3 or self.cfg.capacity < 4096 or self.cfg.fit_steps < 1 or self.cfg.batch_size < 1:
            raise ValueError("Require three members, capacity >=4096, positive steps and batch size")
        if self.cfg.reward_max != 0.057 or not 0 < self.cfg.uncertainty_quantile < 1:
            raise ValueError("Invalid physical reward maximum or uncertainty quantile")
        if not 0 < self.cfg.uncertainty_min <= self.cfg.uncertainty_max:
            raise ValueError("Invalid uncertainty limits")
        # Preserve the PPO CUDA RNG stream; neither model initialization nor
        # bootstrap/replay draws consume policy-training randomness.
        with torch.inference_mode(False), torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device.index]), torch.device(self.device):
            torch.cuda.manual_seed(self.seed)
            self.models = nn.ModuleList([
                nn.Sequential(nn.Linear(152, self.cfg.hidden_size), nn.ELU(),
                              nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size), nn.ELU(),
                              nn.Linear(self.cfg.hidden_size, 29))
                for _ in range(self.cfg.members)
            ]).to(device=self.device, dtype=torch.float32)
            self.input_moments = _Moments(152, self.device)
            self.target_moments = _Moments(29, self.device)
            self.target_floor = torch.full((29,), self.cfg.delta_std_floor, device=self.device)
            self.target_floor[-1] = self.cfg.reward_std_floor
            self._ready = torch.zeros((), dtype=torch.bool, device=self.device)
            self._uncertainty_limit = torch.full((), self.cfg.uncertainty_min, device=self.device)
            validation_capacity = self.cfg.capacity // 10
            self.train_replay = _Replay(self.cfg.capacity - validation_capacity, self.device)
            self.validation_replay = _Replay(validation_capacity, self.device)
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=self.cfg.learning_rate, capturable=True)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed + 7919)
        self.fit_calls = self.real_train_samples_seen = self.real_validation_samples_seen = 0
        self.last_metrics: dict[str, Any] = {}

    def _check(self, value: torch.Tensor, shape: tuple, name: str):
        if not isinstance(value, torch.Tensor) or value.device != self.device or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape} on {self.device}")

    def _scales(self):
        return (self.input_moments.scale(self.cfg.input_std_floor), self.target_moments.scale(self.target_floor))

    def _predict(self, x: torch.Tensor, input_scale: torch.Tensor):
        normalized = (x - self.input_moments.mean) / input_scale
        return torch.stack([model(normalized) for model in self.models], dim=0)

    @staticmethod
    def _uncertainty(predictions: torch.Tensor):
        return predictions.var(dim=0, unbiased=False).mean(dim=-1).sqrt()

    @torch.no_grad()
    def _validation(self, x, target, input_scale, target_scale):
        predictions = self._predict(x, input_scale)
        estimate = predictions.mean(dim=0) * target_scale + self.target_moments.mean
        delta_mse = ((estimate[:, :28] - target[:, :28]) / target_scale[:28]).square().mean()
        persistence_mse = (target[:, :28] / target_scale[:28]).square().mean()
        return {
            "delta_mse_normalized": delta_mse,
            "persistence_delta_mse_normalized": persistence_mse,
            "to_persistence_mse_ratio": delta_mse / persistence_mse.clamp_min(1e-8),
            "obs_rmse_raw": (estimate[:, :28] - target[:, :28]).square().mean().sqrt(),
            "reward_mae": (estimate[:, 28] - target[:, 28]).abs().mean(),
            "reward_mean_baseline_mae": (target[:, 28] - self.target_moments.mean[28]).abs().mean(),
            "uncertainty_quantile": torch.quantile(self._uncertainty(predictions), self.cfg.uncertainty_quantile),
        }

    @torch.no_grad()
    def add(self, history, actions, next_obs, rewards, dones, env_ids) -> dict[str, int]:
        n = history.shape[0]
        for value, shape, name in ((history, (n, 4, 36), "history"), (actions, (n, 8), "actions"),
                                   (next_obs, (n, 36), "next_obs"), (rewards, (n,), "rewards"),
                                   (dones, (n,), "dones"), (env_ids, (n,), "env_ids")):
            self._check(value, shape, name)
        if env_ids.dtype not in (torch.int32, torch.int64) or bool((env_ids < 0).any()):
            raise ValueError("env_ids must be nonnegative int32/int64 identifiers")
        clipped = actions.float().clamp(-1, 1)
        x = torch.cat((history.float().flatten(1), clipped), dim=1)
        y = torch.cat((next_obs[:, :28].float() - history[:, -1, :28].float(), rewards.float()[:, None]), dim=1)
        finite = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1) & torch.isfinite(next_obs).all(dim=1)
        keep = ~dones.bool() & finite
        if not torch.allclose(next_obs[keep, 28:], clipped[keep], atol=1e-5, rtol=0):
            raise ValueError("Nonterminal next observation action slots must equal clipped current actions")
        if bool((rewards[keep] > self.cfg.reward_max + 1e-5).any()):
            raise ValueError("Real reward exceeds the unchanged environment's physical maximum")
        training, validation = keep & (env_ids % 10 != 0), keep & (env_ids % 10 == 0)
        tx, ty = x[training], y[training]
        self.input_moments.update(tx)
        self.target_moments.update(ty)
        self.train_replay.add(tx, ty)
        self.validation_replay.add(x[validation], y[validation])
        nt, nv = tx.shape[0], int(validation.sum().item())
        self.real_train_samples_seen += nt
        self.real_validation_samples_seen += nv
        return {"train_added": nt, "validation_added": nv, "excluded_done_or_nonfinite": n - nt - nv}

    def fit(self) -> dict[str, Any]:
        """Fit real training replay, then assess a disjoint real validation set."""
        metrics = {"train_size": self.train_replay.size, "validation_size": self.validation_replay.size,
                   "real_train_samples_seen": self.real_train_samples_seen,
                   "real_validation_samples_seen": self.real_validation_samples_seen,
                   "model_fit_calls": self.fit_calls, "ready_before_fit": bool(self._ready.item()), "ready": False}
        if self.train_replay.size < self.cfg.min_train_samples or self.validation_replay.size < self.cfg.min_validation_samples:
            self._ready.fill_(False)
            self.last_metrics = metrics
            return dict(metrics)
        with torch.inference_mode(False), torch.enable_grad():
            xs, ys = self._scales()
            size = min(self.cfg.validation_batch_size, self.validation_replay.size)
            val_indices = torch.randperm(self.validation_replay.size, device=self.device, generator=self.generator)[:size]
            val_x, val_y = self.validation_replay.x[val_indices], self.validation_replay.y[val_indices]
            self.models.eval()
            before = self._validation(val_x, val_y, xs, ys)
            losses = torch.zeros((self.cfg.fit_steps, 3), device=self.device)
            self.models.train()
            for step in range(self.cfg.fit_steps):
                indices = torch.randint(self.train_replay.size, (3, self.cfg.batch_size), device=self.device, generator=self.generator)
                x = (self.train_replay.x[indices] - self.input_moments.mean) / xs
                target = (self.train_replay.y[indices] - self.target_moments.mean) / ys
                prediction = torch.stack([model(x[i]) for i, model in enumerate(self.models)])
                errors = (prediction - target).square()
                # Equal dynamics/reward task weighting: reward is not diluted
                # to one twenty-ninth of the objective by observation width.
                dynamics_loss, reward_loss = errors[..., :28].mean(), errors[..., 28].mean()
                loss = dynamics_loss + reward_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite CUDA world-model loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.models.parameters(), self.cfg.grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                losses[step] = torch.stack((dynamics_loss.detach(), reward_loss.detach(), norm.detach()))
            self.fit_calls += 1
            self.models.eval()
            with torch.no_grad():
                # Same held-out transitions and normalizers before/after fit.
                after = self._validation(val_x, val_y, xs, ys)
                self._uncertainty_limit.copy_(after["uncertainty_quantile"].clamp(self.cfg.uncertainty_min, self.cfg.uncertainty_max))
                self._ready.copy_((after["to_persistence_mse_ratio"] <= self.cfg.max_validation_persistence_ratio)
                                  & (after["reward_mae"] <= self.cfg.max_validation_reward_mae)
                                  & (self.fit_calls >= self.cfg.min_fit_calls))
                scalars = {
                    "train_dynamics_loss": losses[:, 0].mean(), "train_reward_loss": losses[:, 1].mean(),
                    "train_gradient_norm_before_clip": losses[:, 2].mean(),
                    "persistence_delta_mse_normalized": after["persistence_delta_mse_normalized"],
                    "uncertainty_threshold": self._uncertainty_limit,
                }
                scalars.update({"validation_" + key: value for key, value in after.items()})
                scalars.update({"validation_pre_" + key: value for key, value in before.items()})
                scalars.update({"validation_post_" + key: value for key, value in after.items()})
                if not all(bool(torch.isfinite(value)) for value in scalars.values()):
                    raise RuntimeError("Nonfinite world-model validation metadata")
                metrics.update({key: float(value.item()) for key, value in scalars.items()})
                metrics.update(model_fit_calls=self.fit_calls, fit_steps=self.cfg.fit_steps,
                               validation_batch_size=size, ready=bool(self._ready.item()))
        self.last_metrics = metrics
        return dict(metrics)

    @torch.no_grad()
    def sample_starts(self, batch: int) -> torch.Tensor:
        if batch < 1 or not self.train_replay.size:
            raise ValueError("Positive batch and nonempty real training replay required")
        ids = torch.randint(self.train_replay.size, (batch,), device=self.device, generator=self.generator)
        return self.train_replay.x[ids, :144].reshape(batch, 4, 36).clone()

    @torch.no_grad()
    def imagine(self, history, raw_actions):
        n = history.shape[0]
        self._check(history, (n, 4, 36), "history")
        self._check(raw_actions, (n, 8), "raw_actions")
        finite_input = torch.isfinite(history).all(dim=(1, 2)) & torch.isfinite(raw_actions).all(dim=1)
        safe_history = torch.nan_to_num(history.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        actions = torch.nan_to_num(raw_actions.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        x = torch.cat((safe_history.flatten(1), actions), dim=1)
        xs, ys = self._scales()
        predictions = self._predict(x, xs)
        normalized_estimate = predictions.mean(dim=0)
        estimate = normalized_estimate * ys + self.target_moments.mean
        next_state = safe_history[:, -1, :28] + estimate[:, :28]
        next_obs = torch.cat((next_state, actions), dim=1)
        uncertainty = self._uncertainty(predictions)
        gravity_norm = next_state[:, 6:9].norm(dim=1)
        valid = (self._ready & finite_input & torch.isfinite(estimate).all(dim=1)
                 & (((x - self.input_moments.mean) / xs).abs().amax(dim=1) <= self.cfg.input_max_sigma)
                 & (normalized_estimate.abs().amax(dim=1) <= self.cfg.output_max_sigma)
                 & (uncertainty <= self._uncertainty_limit)
                 & (next_state.abs().amax(dim=1) <= 20.0)
                 & (-next_state[:, 8] >= self.cfg.min_predicted_upright)
                 & ((gravity_norm - 1.0).abs() <= self.cfg.gravity_norm_tolerance))
        fallback = torch.cat((safe_history[:, -1, :28].clamp(-20, 20), actions), dim=1)
        next_obs = torch.where(valid[:, None], next_obs, fallback)
        reward = torch.where(valid, estimate[:, 28].clamp(max=self.cfg.reward_max), 0.0)
        return next_obs, reward, valid, torch.nan_to_num(uncertainty, nan=1e6, posinf=1e6, neginf=1e6)

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "config": self.config, "seed": self.seed,
                "models": self.models.state_dict(), "optimizer": self.optimizer.state_dict(),
                "input_moments": self.input_moments.state_dict(), "target_moments": self.target_moments.state_dict(),
                "rng_state": self.generator.get_state(), "fit_calls": self.fit_calls,
                "real_train_samples_seen": self.real_train_samples_seen,
                "real_validation_samples_seen": self.real_validation_samples_seen,
                "ready": self._ready.clone(), "uncertainty_limit": self._uncertainty_limit.clone(),
                "last_metrics": dict(self.last_metrics), "replay_saved": False}

    def load_state_dict(self, state: dict[str, Any]):
        if state["version"] != 1 or state["config"] != self.config:
            raise ValueError("World-model checkpoint version/configuration mismatch")
        self.models.load_state_dict(state["models"])
        self.optimizer.load_state_dict(state["optimizer"])
        for entry in self.optimizer.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to(self.device)
        self.input_moments.load_state_dict(state["input_moments"])
        self.target_moments.load_state_dict(state["target_moments"])
        # Torch serializes generator state as CPU bytes, not model computation.
        self.generator.set_state(state["rng_state"].cpu())
        self.seed, self.fit_calls = int(state["seed"]), int(state["fit_calls"])
        self.real_train_samples_seen = int(state["real_train_samples_seen"])
        self.real_validation_samples_seen = int(state["real_validation_samples_seen"])
        self._ready.copy_(state["ready"])
        self._uncertainty_limit.copy_(state["uncertainty_limit"])
        self.last_metrics = dict(state["last_metrics"])
        self.train_replay.position = self.train_replay.size = 0
        self.validation_replay.position = self.validation_replay.size = 0
        self.models.eval()
