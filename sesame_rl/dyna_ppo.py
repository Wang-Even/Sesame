"""CUDA-only local Dyna augmentation of the installed RSL-RL PPO.

Real PPO is unchanged. Auxiliary data uses fresh current-policy actions from
real replay states, a one-step continuing learned model, and detached TD targets.
The real environment remains the only source of policy normalization statistics
and world-model training transitions. This is not a paper-specific reproduction.
"""
from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from sesame_rl.dyna_world_model import DynaWorldModel


DEFAULT_CONFIG = {
    "warmup_updates": 40, "imagined_batch_size": 512, "imagined_weight": 0.1,
    "imagined_epochs": 1, "imagined_minibatches": 4, "min_imagined_samples": 64,
    "world_model": {},
}


class DynaPPO(PPO):
    def __init__(self, policy, *args, dyna_config=None, dyna_seed=42, **kwargs):
        super().__init__(policy, *args, **kwargs)
        if torch.device(self.device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DynaPPO requires CUDA")
        if policy.is_recurrent or self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError("This experiment supports only plain single-GPU feed-forward PPO")
        supplied = dyna_config or {}
        if set(supplied) - set(DEFAULT_CONFIG):
            raise ValueError("Unknown Dyna configuration keys")
        self.dyna_config = copy.deepcopy({**DEFAULT_CONFIG, **supplied})
        self.dyna_seed = int(dyna_seed)
        for key in ("warmup_updates", "imagined_batch_size", "imagined_epochs", "imagined_minibatches", "min_imagined_samples"):
            if not isinstance(self.dyna_config[key], int) or self.dyna_config[key] < 1:
                raise ValueError(f"Invalid {key}")
        if not 0 < self.dyna_config["imagined_weight"] <= 1:
            raise ValueError("imagined_weight must be in (0,1]")
        self.world_model = DynaWorldModel(self.device, self.dyna_seed + 100003, self.dyna_config["world_model"])
        self.imagined_generator = torch.Generator(device=self.device).manual_seed(self.dyna_seed + 200003)
        self.history = None
        self.env_ids = None
        self.updates = self.real_transitions_seen = self.real_optimizer_steps = 0
        self.imagined_generated = self.imagined_accepted = self.imagined_optimizer_steps = 0
        self.imagined_update_calls = self.model_optimizer_steps = 0
        self.last_metrics = {}

    def act(self, obs):
        current = obs["policy"]
        if not current.is_cuda or current.shape[-1] != 36:
            raise ValueError("Expected 36-D CUDA raw policy observations")
        if self.history is None:
            # Allocate ordinary tensors even when the runner collects under
            # inference_mode; replay/model training must not inherit that mode.
            with torch.inference_mode(False):
                self.history = current.clone().unsqueeze(1).repeat(1, 4, 1)
                self.env_ids = torch.arange(current.shape[0], device=current.device)
        elif not torch.equal(self.history[:, -1], current):
            raise RuntimeError("History does not match the current real observation")
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        # This precedes PPO's timeout-bootstrap reward adjustment and transition
        # clearing. The model sees only unmodified rewards and raw actions.
        self.world_model.add(self.history, self.transition.actions, obs["policy"],
                             rewards.flatten(), dones.flatten(), self.env_ids)
        self.real_transitions_seen += obs["policy"].shape[0]
        with torch.no_grad():
            self.history[:, :-1].copy_(self.history[:, 1:].clone())
            self.history[:, -1].copy_(obs["policy"])
            ended = dones.flatten().bool()
            self.history[ended] = obs["policy"][ended, None, :].expand(-1, 4, -1)
        super().process_env_step(obs, rewards, dones, extras)

    @staticmethod
    def _obs(values):
        return TensorDict({"policy": values}, batch_size=[values.shape[0]], device=values.device)

    def _imagined_update(self):
        cfg = self.dyna_config
        metrics = {"imagined_candidates": 0.0, "imagined_accepted": 0.0,
                   "imagined_acceptance_fraction": 0.0, "imagined_optimizer_steps": 0.0,
                   "imagined_surrogate": 0.0, "imagined_value_loss": 0.0,
                   "imagined_kl": 0.0, "imagined_normalization_unchanged": 1.0}
        if self.updates <= cfg["warmup_updates"] or not self.world_model.last_metrics.get("ready", False):
            return metrics
        normalization = {k: v.detach().clone() for k, v in self.policy.state_dict().items() if "obs_normalizer" in k}
        with torch.no_grad():
            history = self.world_model.sample_starts(cfg["imagined_batch_size"])
            observations = self._obs(history[:, -1].clone())
            actions = self.policy.act(observations).clone()
            old_log_prob = self.policy.get_actions_log_prob(actions).clone()
            old_values = self.policy.evaluate(observations).flatten().clone()
            old_mean, old_std = self.policy.action_mean.clone(), self.policy.action_std.clone()
            next_obs, rewards, valid, uncertainty = self.world_model.imagine(history, actions)
            next_values = self.policy.evaluate(self._obs(next_obs)).flatten()
            # Horizon 1: bootstrapped continuing TD return; no fabricated done.
            returns = rewards + self.gamma * next_values
            advantages = returns - old_values
            valid &= torch.isfinite(returns) & torch.isfinite(advantages) & torch.isfinite(old_log_prob)
            count = int(valid.sum().item())
            self.imagined_generated += actions.shape[0]
            self.imagined_accepted += count
            metrics.update(imagined_candidates=float(actions.shape[0]), imagined_accepted=float(count),
                           imagined_acceptance_fraction=count/actions.shape[0],
                           imagined_uncertainty_mean=float(uncertainty.mean().item()))
            if count < cfg["min_imagined_samples"]:
                return metrics
            observations = self._obs(observations["policy"][valid].clone())
            actions, old_log_prob, old_values = actions[valid], old_log_prob[valid], old_values[valid]
            old_mean, old_std = old_mean[valid], old_std[valid]
            returns, advantages = returns[valid], advantages[valid]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        totals = torch.zeros(3, device=self.device)
        steps = 0
        stop = False
        for _ in range(cfg["imagined_epochs"]):
            order = torch.randperm(count, device=self.device)
            for ids in torch.tensor_split(order, cfg["imagined_minibatches"]):
                if ids.numel() == 0:
                    continue
                batch = self._obs(observations["policy"][ids])
                self.policy.act(batch)
                log_prob = self.policy.get_actions_log_prob(actions[ids])
                values = self.policy.evaluate(batch).flatten()
                entropy = self.policy.entropy
                with torch.no_grad():
                    mu, sigma = self.policy.action_mean, self.policy.action_std
                    kl = (torch.log(sigma / old_std[ids]) +
                          (old_std[ids].square() + (old_mean[ids] - mu).square()) /
                          (2.0 * sigma.square()) - 0.5).sum(-1).mean()
                    if not bool(torch.isfinite(kl)):
                        raise RuntimeError("Nonfinite imagined policy KL")
                    if self.desired_kl and float(kl.item()) > 1.5 * self.desired_kl:
                        stop = True
                        break
                ratio = (log_prob - old_log_prob[ids]).exp()
                surrogate = torch.maximum(-advantages[ids] * ratio,
                                          -advantages[ids] * ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
                clipped_value = old_values[ids] + (values-old_values[ids]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values-returns[ids]).square(),
                                           (clipped_value-returns[ids]).square()).mean()
                loss = cfg["imagined_weight"] * (surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite imagined PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                totals += torch.stack((surrogate.detach(), value_loss.detach(), kl.detach()))
                steps += 1
            if stop:
                break
        self.imagined_optimizer_steps += steps
        self.imagined_update_calls += int(steps > 0)
        for key, value in normalization.items():
            if not torch.equal(value, self.policy.state_dict()[key]):
                raise RuntimeError("Imagined data changed a policy observation normalizer")
        metrics["imagined_optimizer_steps"] = float(steps)
        if steps:
            for key, value in zip(("imagined_surrogate", "imagined_value_loss", "imagined_kl"), totals / steps):
                metrics[key] = float(value.item())
        return metrics

    def update(self):
        losses = super().update()
        self.updates += 1
        self.real_optimizer_steps += self.num_learning_epochs * self.num_mini_batches
        model_metrics = self.world_model.fit()
        self.model_optimizer_steps = self.world_model.fit_calls * self.world_model.cfg.fit_steps
        # Preserve the RNG used by future real simulator rollouts. Imagination
        # sampling and its discarded actor samples use their own CUDA stream.
        index = torch.device(self.device).index
        if index is None:
            index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            torch.cuda.set_rng_state(self.imagined_generator.get_state(), index)
            imagined = self._imagined_update()
            self.imagined_generator.set_state(torch.cuda.get_rng_state(index))
        losses.update({"wm_"+k: float(v) for k, v in model_metrics.items() if isinstance(v, (int, float, bool))})
        losses.update(imagined)
        losses["imagined_optimizer_steps_total"] = float(self.imagined_optimizer_steps)
        losses["imagined_accepted_total"] = float(self.imagined_accepted)
        losses["world_model_optimizer_steps_total"] = float(self.model_optimizer_steps)
        if not all(math.isfinite(float(v)) for v in losses.values()):
            raise RuntimeError("Nonfinite Dyna log metric")
        self.last_metrics = dict(losses)
        return losses

    def training_summary(self):
        return {"real_transitions_seen": self.real_transitions_seen, "real_updates": self.updates,
                "real_optimizer_steps": self.real_optimizer_steps, "model_optimizer_steps": self.model_optimizer_steps,
                "model_member_optimizer_steps": self.model_optimizer_steps * 3,
                "imagined_generated": self.imagined_generated, "imagined_accepted": self.imagined_accepted,
                "imagined_optimizer_steps": self.imagined_optimizer_steps, "imagined_update_calls": self.imagined_update_calls,
                "model_fit_calls": self.world_model.fit_calls, "imagined_horizon": 1,
                "history_frames": 4, "replay_saved": False}

    def dyna_state_dict(self):
        return {"version": 1, "config": copy.deepcopy(self.dyna_config), "seed": self.dyna_seed,
                "world_model": self.world_model.state_dict(), "rng_state": self.imagined_generator.get_state(),
                **self.training_summary()}

    def load_dyna_state_dict(self, state):
        if state["version"] != 1 or state["config"] != self.dyna_config:
            raise ValueError("Dyna checkpoint/configuration mismatch")
        self.world_model.load_state_dict(state["world_model"])
        self.imagined_generator.set_state(state["rng_state"].cpu())
        self.dyna_seed = state["seed"]
        self.updates = state["real_updates"]
        for key in ("real_transitions_seen", "real_optimizer_steps", "model_optimizer_steps", "imagined_generated",
                    "imagined_accepted", "imagined_optimizer_steps", "imagined_update_calls"):
            setattr(self, key, state[key])
        self.history = self.env_ids = None
