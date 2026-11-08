"""CUDA-only SAC and branching dueling Double-DQN for the Sesame walking task.

Both consume the environment's already physically scaled 36-D observation and
the unchanged, time-integrated heading reward. SAC has continuous 8-D actions.
The local DQN adaptation uses one dueling Q branch per joint and five action
bins per branch; it is NOT a flat DQN over the continuous action space or a full
reproduction of the BDQ paper (uniform replay, no prioritized replay).

References:
  SAC + automatic temperature: https://arxiv.org/abs/1812.05905
  Action branching / mean branch TD target: https://arxiv.org/abs/1711.08946

Replay receives the true next state BEFORE an auto-reset. Only genuine
termination suppresses bootstrap; time-limit truncation must use the captured
terminal observation and terminated=False. Reward scaling is exactly one.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


FORMAT_VERSION = 1
POLICY_VERSION = "sesame_heading_offpolicy_v1"
OBSERVATION_PREPROCESSING = "environment_physical_scaling_only"


def _cuda_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Sesame off-policy models and replay require CUDA; CPU fallback is prohibited")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def default_config(algorithm: str) -> dict[str, Any]:
    if algorithm not in ("sac", "dqn"):
        raise ValueError(f"Unknown algorithm: {algorithm}")
    config = {
        "seed": 42,
        "observation_dim": 36,
        "action_dim": 8,
        "hidden_dims": [256, 256],
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "target_tau": 0.005,
        "max_grad_norm": 10.0,
        "reward_scale": 1.0,
    }
    if algorithm == "sac":
        config.update(initial_temperature=0.02, target_entropy=-8.0,
                      log_std_min=-5.0, log_std_max=2.0, initial_log_std=-1.0)
    else:
        config.update(action_bins=[-1.0, -0.5, 0.0, 0.5, 1.0],
                      td_loss="smooth_l1", branch_target_aggregation="mean")
    return config


def _config(algorithm: str, supplied: dict | None) -> dict[str, Any]:
    defaults = default_config(algorithm)
    supplied = {} if supplied is None else copy.deepcopy(supplied)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {algorithm} configuration fields: {sorted(unknown)}")
    config = {**defaults, **supplied}
    if config["observation_dim"] != 36 or config["action_dim"] != 8:
        raise ValueError("This experiment fixes observation/action dimensions at 36/8")
    if config["reward_scale"] != 1.0:
        raise ValueError("The comparison must preserve the original environment reward scale")
    if not config["hidden_dims"] or any(not isinstance(n, int) or n <= 0 for n in config["hidden_dims"]):
        raise ValueError("hidden_dims must be positive integers")
    for key in ("learning_rate", "target_tau", "max_grad_norm"):
        if not math.isfinite(float(config[key])) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < config["gamma"] < 1 or config["target_tau"] > 1:
        raise ValueError("gamma must be in (0,1); target_tau must be in (0,1]")
    if algorithm == "sac":
        if not math.isfinite(float(config["initial_temperature"])) or config["initial_temperature"] <= 0:
            raise ValueError("initial_temperature must be positive")
        if not config["log_std_min"] < config["initial_log_std"] < config["log_std_max"]:
            raise ValueError("initial_log_std must lie inside the log_std bounds")
        if not math.isfinite(float(config["target_entropy"])):
            raise ValueError("target_entropy must be finite")
    else:
        if config["action_bins"] != [-1.0, -0.5, 0.0, 0.5, 1.0]:
            raise ValueError("This DQN experiment fixes five action bins per joint")
        if config["td_loss"] != "smooth_l1" or config["branch_target_aggregation"] != "mean":
            raise ValueError("Unsupported branching DQN loss or target aggregation")
    config["seed"] = int(config["seed"])
    return config


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    if output_dim is not None:
        layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _tensor(tensor: torch.Tensor, device: torch.device, shape: tuple[int, ...], label: str) -> None:
    if tensor.device != device or tuple(tensor.shape) != shape:
        raise ValueError(f"{label} must have shape {shape} on {device}; got {tuple(tensor.shape)} on {tensor.device}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Nonfinite {label}")


class ReplayBuffer:
    """Uniform GPU ring buffer; storage does not alias simulator tensors."""

    def __init__(self, capacity: int, obs_dim: int = 36, action_dim: int = 8,
                 device: str | torch.device = "cuda:0", seed: int = 42):
        self.device = _cuda_device(device)
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity, self.obs_dim, self.action_dim = capacity, obs_dim, action_dim
        self.position = self.size = self.total_added = 0
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode(False):
            self.storage = {
                "obs": torch.empty((capacity, obs_dim), device=self.device),
                "actions": torch.empty((capacity, action_dim), device=self.device),
                "rewards": torch.empty((capacity,), device=self.device),
                "next_obs": torch.empty((capacity, obs_dim), device=self.device),
                "terminated": torch.empty((capacity,), device=self.device, dtype=torch.bool),
            }

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
            next_obs: torch.Tensor, terminated: torch.Tensor) -> None:
        count = obs.shape[0]
        if count < 1:
            return
        rewards, terminated = rewards.reshape(-1), terminated.reshape(-1)
        for key, value, shape in (("obs", obs, (count, self.obs_dim)),
                                  ("actions", actions, (count, self.action_dim)),
                                  ("rewards", rewards, (count,)),
                                  ("next_obs", next_obs, (count, self.obs_dim)),
                                  ("terminated", terminated, (count,))):
            _tensor(value, self.device, shape, key)
        if terminated.dtype != torch.bool:
            raise ValueError("Replay terminated must be a bool genuine-termination mask")
        if bool((actions.abs() > 1.000001).any()):
            raise ValueError("Replay actions must be normalized/clipped to [-1,1]")
        values = {"obs": obs, "actions": actions, "rewards": rewards,
                  "next_obs": next_obs, "terminated": terminated}
        # A large insertion keeps its final capacity entries and correctly
        # advances the ring cursor as if every preceding entry had been added.
        kept = min(count, self.capacity)
        start = (self.position + count - kept) % self.capacity
        ids = (torch.arange(kept, device=self.device) + start) % self.capacity
        for key, value in values.items():
            self.storage[key][ids] = value[-kept:].detach()
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_added += count

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size == 0 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Cannot sample an empty replay or invalid batch size")
        # Sampling with replacement is the intended uniform replay law.
        ids = torch.randint(self.size, (batch_size,), device=self.device, generator=self.generator)
        return {key: value[ids] for key, value in self.storage.items()}

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "obs_dim": self.obs_dim, "action_dim": self.action_dim,
                "position": self.position, "size": self.size, "total_added": self.total_added,
                "generator_state": self.generator.get_state(),
                "storage": {key: value[:self.size].clone() for key, value in self.storage.items()}}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state["capacity"], state["obs_dim"], state["action_dim"]) != (self.capacity, self.obs_dim, self.action_dim):
            raise ValueError("Replay checkpoint shape mismatch")
        if not 0 <= state["size"] <= self.capacity or not 0 <= state["position"] < self.capacity:
            raise ValueError("Invalid replay checkpoint cursor")
        for key in self.storage:
            self.storage[key][:state["size"]].copy_(state["storage"][key].to(self.device))
        self.position, self.size, self.total_added = state["position"], state["size"], state["total_added"]
        self.generator.set_state(state["generator_state"].cpu())


class SquashedGaussianActor(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.action_dim = config["action_dim"]
        self.log_std_min, self.log_std_max = config["log_std_min"], config["log_std_max"]
        self.net = _mlp(config["observation_dim"], config["hidden_dims"], 2 * self.action_dim)
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.action_dim:].fill_(config["initial_log_std"])

    def distribution_parameters(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, obs: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(obs)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        pre_tanh = mean + log_std.exp() * noise
        gaussian_log_prob = -0.5 * noise.square() - log_std - 0.5 * math.log(2.0 * math.pi)
        # Stable equivalent of log(1 - tanh(pre_tanh)^2), avoiding saturation NaNs.
        log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = (gaussian_log_prob - log_jacobian).sum(dim=-1)
        return pre_tanh.tanh(), log_prob

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(obs)
        return mean.tanh()


class TwinQ(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = config["observation_dim"] + config["action_dim"]
        self.q1 = _mlp(dim, config["hidden_dims"], 1)
        self.q2 = _mlp(dim, config["hidden_dims"], 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((obs, actions), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class BranchingDuelingQ(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.branches, self.bins = config["action_dim"], len(config["action_bins"])
        self.trunk = _mlp(config["observation_dim"], config["hidden_dims"])
        width = config["hidden_dims"][-1]
        self.value = nn.Linear(width, 1)
        self.advantages = nn.Linear(width, self.branches * self.bins)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(obs)
        value = self.value(shared).unsqueeze(-1)
        advantage = self.advantages(shared).view(-1, self.branches, self.bins)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class _BaseAgent:
    algorithm: str

    def _setup(self, device: str | torch.device, config: dict | None) -> None:
        self.device = _cuda_device(device)
        self.config = _config(self.algorithm, config)
        self.updates = 0
        self.generator = torch.Generator(device=self.device).manual_seed(self.config["seed"] + 310003)

    def _obs(self, obs: torch.Tensor) -> None:
        if obs.ndim != 2 or obs.shape[1] != self.config["observation_dim"] or obs.device != self.device:
            raise ValueError(f"Expected CUDA observation [N,36] on {self.device}")

    def _batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        obs, actions = batch["obs"], batch["actions"]
        next_obs = batch["next_obs"]
        rewards, terminated = batch["rewards"].reshape(-1), batch["terminated"].reshape(-1)
        self._obs(obs)
        count = obs.shape[0]
        for key, value, shape in (("obs", obs, (count, 36)), ("actions", actions, (count, 8)),
                                  ("next_obs", next_obs, (count, 36)), ("rewards", rewards, (count,)),
                                  ("terminated", terminated, (count,))):
            _tensor(value, self.device, shape, key)
        if terminated.dtype != torch.bool:
            raise ValueError("terminated must be bool; do not pass an auto-reset done mask")
        return obs, actions, rewards, next_obs, terminated

    @torch.no_grad()
    def _polyak(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.config["target_tau"]
        for target_parameter, parameter in zip(target.parameters(), source.parameters(), strict=True):
            target_parameter.lerp_(parameter, tau)

    def _step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, parameters) -> float:
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite {self.algorithm} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, self.config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        return float(norm.item())

    def _checkpoint_header(self, extra: dict | None) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION, "policy_version": POLICY_VERSION,
            "algorithm": self.algorithm, "algorithm_description": self.description,
            "config": copy.deepcopy(self.config), "updates": self.updates,
            "observation_preprocessing": OBSERVATION_PREPROCESSING,
            "reward_version": "absolute_heading_v1", "reward_scale": 1.0,
            "generator_state": self.generator.get_state(),
            "torch_cpu_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "replay_saved": False, "environment_saved": False,
            "resume_scope": "Agent learning/optimizer state and RNG; replay and simulator require separate restoration",
            "extra": copy.deepcopy(extra or {}),
        }

    def _load_header(self, checkpoint: dict, restore_rng: bool) -> None:
        for key, expected in (("format_version", FORMAT_VERSION), ("policy_version", POLICY_VERSION),
                              ("algorithm", self.algorithm),
                              ("observation_preprocessing", OBSERVATION_PREPROCESSING)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint {key} mismatch")
        if checkpoint["config"] != self.config:
            raise ValueError("Checkpoint configuration differs from constructed agent")
        self.updates = int(checkpoint["updates"])
        self.generator.set_state(checkpoint["generator_state"].cpu())
        if restore_rng:
            torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
            torch.cuda.set_rng_state(checkpoint["torch_cuda_rng_state"].cpu(), self.device)


class SACAgent(_BaseAgent):
    algorithm = "sac"
    description = "SAC with squashed Gaussian actor, twin Q, Polyak targets and learned temperature"

    def __init__(self, device: str | torch.device = "cuda:0", config: dict | None = None):
        self._setup(device, config)
        # Allocate AND initialize all network parameters on GPU, never on CPU.
        with torch.device(self.device):
            self.actor = SquashedGaussianActor(self.config)
            self.critic = TwinQ(self.config)
            self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
            self.log_alpha = nn.Parameter(torch.tensor(math.log(self.config["initial_temperature"])))
        lr = self.config["learning_rate"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False, epsilon: float = 0.0) -> torch.Tensor:
        self._obs(obs)
        return self.actor(obs) if deterministic else self.actor.sample(obs, self.generator)[0]

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, actions, rewards, next_obs, terminated = self._batch(batch)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs, self.generator)
            target_q1, target_q2 = self.target_critic(next_obs, next_actions)
            soft_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rewards + self.config["gamma"] * (~terminated).float() * soft_value
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_grad_norm = self._step(critic_loss, self.critic_optimizer, self.critic.parameters())
        # Preserve the differentiable dQ/da path but do not accumulate critic
        # parameter gradients during actor optimization.
        self.critic.requires_grad_(False)
        try:
            sampled_actions, log_prob = self.actor.sample(obs, self.generator)
            policy_q1, policy_q2 = self.critic(obs, sampled_actions)
            actor_loss = (self.alpha.detach() * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
            actor_grad_norm = self._step(actor_loss, self.actor_optimizer, self.actor.parameters())
        finally:
            self.critic.requires_grad_(True)
        # Standard log-temperature surrogate; log_prob is detached so this
        # optimizer cannot change actor parameters or reuse the actor graph.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.config["target_entropy"])).mean()
        self._step(alpha_loss, self.alpha_optimizer, [self.log_alpha])
        self._polyak(self.target_critic, self.critic)
        self.updates += 1
        metrics = {
            "critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()), "alpha": float(self.alpha.detach().item()),
            "entropy": float(-log_prob.detach().mean().item()),
            "q_mean": float(torch.minimum(q1, q2).detach().mean().item()),
            "target_q_mean": float(target.mean().item()),
            "critic_grad_norm": critic_grad_norm, "actor_grad_norm": actor_grad_norm,
            "updates": self.updates,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite SAC update metrics")
        return metrics

    def checkpoint(self, extra: dict | None = None) -> dict[str, Any]:
        checkpoint = self._checkpoint_header(extra)
        checkpoint.update(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                          target_critic=self.target_critic.state_dict(), log_alpha=self.log_alpha.detach().clone(),
                          actor_optimizer=self.actor_optimizer.state_dict(),
                          critic_optimizer=self.critic_optimizer.state_dict(),
                          alpha_optimizer=self.alpha_optimizer.state_dict())
        return checkpoint

    def load_checkpoint(self, checkpoint: dict, load_optimizers: bool = True, restore_rng: bool | None = None) -> None:
        self._load_header(checkpoint, load_optimizers if restore_rng is None else restore_rng)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


class BranchingDQNAgent(_BaseAgent):
    algorithm = "dqn"
    description = "Local branching dueling Double-DQN: eight branches, five bins per joint, uniform replay"

    def __init__(self, device: str | torch.device = "cuda:0", config: dict | None = None):
        self._setup(device, config)
        with torch.device(self.device):
            self.q = BranchingDuelingQ(self.config)
            self.target_q = copy.deepcopy(self.q).requires_grad_(False)
            self.action_bins = torch.tensor(self.config["action_bins"])
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.config["learning_rate"])

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False, epsilon: float = 0.0) -> torch.Tensor:
        self._obs(obs)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0,1]")
        indices = self.q(obs).argmax(dim=-1)
        if not deterministic and epsilon > 0:
            random_indices = torch.randint(len(self.action_bins), indices.shape, device=self.device,
                                           generator=self.generator)
            # epsilon-greedy over whole joint-action vectors, uniformly sampling
            # all eight bins independently when an environment explores.
            explore = torch.rand((obs.shape[0], 1), device=self.device, generator=self.generator) < epsilon
            indices = torch.where(explore, random_indices, indices)
        return self.action_bins[indices]

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, actions, rewards, next_obs, terminated = self._batch(batch)
        indices = ((actions + 1.0) * 2.0).round().long()
        if bool(((indices < 0) | (indices >= len(self.action_bins))).any()):
            raise ValueError("DQN replay contains out-of-range action bins")
        if not torch.allclose(self.action_bins[indices], actions, atol=1e-6, rtol=0):
            raise ValueError("DQN replay contains actions outside its fixed discretization")
        with torch.no_grad():
            next_indices = self.q(next_obs).argmax(dim=-1, keepdim=True)
            next_branch_q = self.target_q(next_obs).gather(-1, next_indices).squeeze(-1)
            # All branches share the same scalar environment reward; mean the
            # next-state branch values instead of counting reward eight times.
            next_value = next_branch_q.mean(dim=-1)
            target = rewards + self.config["gamma"] * (~terminated).float() * next_value
        selected_branch_q = self.q(obs).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        expanded_target = target[:, None].expand_as(selected_branch_q)
        # Average losses AFTER forming each branch's error (no cancellation).
        loss = F.smooth_l1_loss(selected_branch_q, expanded_target)
        grad_norm = self._step(loss, self.optimizer, self.q.parameters())
        self._polyak(self.target_q, self.q)
        self.updates += 1
        metrics = {
            "q_loss": float(loss.item()), "q_mean": float(selected_branch_q.detach().mean().item()),
            "target_q_mean": float(target.mean().item()),
            "td_absolute_error": float((selected_branch_q.detach() - expanded_target).abs().mean().item()),
            "grad_norm": grad_norm, "updates": self.updates,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite DQN update metrics")
        return metrics

    def checkpoint(self, extra: dict | None = None) -> dict[str, Any]:
        checkpoint = self._checkpoint_header(extra)
        checkpoint.update(q=self.q.state_dict(), target_q=self.target_q.state_dict(),
                          optimizer=self.optimizer.state_dict())
        return checkpoint

    def load_checkpoint(self, checkpoint: dict, load_optimizers: bool = True, restore_rng: bool | None = None) -> None:
        self._load_header(checkpoint, load_optimizers if restore_rng is None else restore_rng)
        self.q.load_state_dict(checkpoint["q"])
        self.target_q.load_state_dict(checkpoint["target_q"])
        if load_optimizers:
            self.optimizer.load_state_dict(checkpoint["optimizer"])


def make_agent(algorithm: str, device: str | torch.device = "cuda:0", config: dict | None = None):
    if algorithm == "sac":
        return SACAgent(device, config)
    if algorithm == "dqn":
        return BranchingDQNAgent(device, config)
    raise ValueError(f"Unknown off-policy algorithm: {algorithm}")


def load_policy(checkpoint_path: str | Path, device: str | torch.device = "cuda:0"):
    """Load a trusted project checkpoint for deterministic CUDA evaluation."""
    device = _cuda_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent = make_agent(checkpoint["algorithm"], device=device, config=checkpoint["config"])
    agent.load_checkpoint(checkpoint, load_optimizers=False, restore_rng=False)
    return agent


load_inference_policy = load_policy
