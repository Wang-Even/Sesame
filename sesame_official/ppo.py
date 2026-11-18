"""PPO-Sim with the same official actor/critic architecture as SuperMimic.

PPO optimization is rsl_rl 5.4.0; this file adapts its observation/distribution
interface. The upstream SuperMimic PpoRunner is PPO-WM, not PPO-Sim.
"""
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from supermimic.runners.supermimic_runner import SuperMimicRunner, build_actor
from supermimic.models import build_mlp


class OfficialActorDistribution(MLPModel):
    def __init__(self, core, action_dim):
        torch.nn.Module.__init__(self)
        self.core = core
        self.distribution = GaussianDistribution(action_dim, init_std=.1, std_type="log")

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("Non-recurrent PPO batches expected")
        mean = self.core(obs)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def update_normalization(self, obs):
        self.core.update_normalization(obs)


class OfficialCritic(MLPModel):
    def __init__(self, dim, hidden_dims):
        torch.nn.Module.__init__(self)
        self.core = build_mlp(dim, hidden_dims, output_dim=1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        return self.core(obs["critic"])

    def update_normalization(self, obs):
        pass


class PpoSimRunner(SuperMimicRunner):
    def _init_policy(self):
        actor = build_actor(self.cfg, self._wm, self.device)
        self._active_task_specs = [ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name,0)>0]
        _, critic_dim = self._policy_input_dims()
        self.actor = OfficialActorDistribution(actor, self._num_actions).to(self.device)
        self.critic = OfficialCritic(critic_dim, self.cfg.critic_hidden_dims).to(self.device)
        n = self.env.num_envs
        template = TensorDict({
            "history":torch.zeros(n,self.cfg.history_length,actor.obs_dim,device=self.device),
            "goal_motion":torch.zeros(n,self._active_task_specs[0].goal_dim,device=self.device),
            "critic":torch.zeros(n,critic_dim,device=self.device)}, batch_size=[n], device=self.device)
        storage = RolloutStorage("rl", n, 32, template, [self._num_actions], device=self.device)
        self.alg = PPO(self.actor, self.critic, storage, num_learning_epochs=5, num_mini_batches=4,
            clip_param=.2, gamma=.99, lam=.95, learning_rate=1e-3, entropy_coef=.005,
            desired_kl=.01, device=self.device)

    def obs_td(self):
        obs = self.env.get_observations()
        state = obs["full_state"]
        rep, aux = self._representation.represent(state), self._representation.extract_aux_state(state)
        ctx = self._sim_context.read(self.env.unwrapped)
        td = self._build_sim_obs_td(rep, aux, obs["actor"], self.motion_cmd.time_steps, ctx)
        td["critic"] = torch.cat([rep, obs["goal"], ctx],dim=-1)
        return td

    def train_iteration(self):
        self.alg.train_mode()
        for _ in range(32):
            with torch.no_grad():
                obs = self.obs_td()
                actions = self.alg.act(obs)
                nxt, rewards, dones, extras = self.env.step(actions)
                self._advance_sim_history(nxt["actor"], dones)
                self.alg.process_env_step(self.obs_td(), rewards, dones, extras)
        with torch.no_grad():
            self.alg.compute_returns(self.obs_td())
        return self.alg.update()
