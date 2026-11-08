"""Capture pre-reset next observations without changing heading task rewards.

Isaac DirectRLEnv automatically resets before returning observations. Replay
must instead bootstrap time limits from the final pre-reset state. The frozen
task has no observation noise or interval events, so an extra observation read
after rewards is side-effect free. All simulation/tensor work remains on CUDA.
"""
from .heading_environment import SesameHeadingEnv


class SesameOffPolicyEnv(SesameHeadingEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.dr_observation_noise or cfg.observation_noise_model is not None:
            raise ValueError('Pre-reset observation capture requires disabled observation noise')
        if cfg.action_noise_model is not None or cfg.events is not None:
            raise ValueError('This frozen task expects no action noise or interval events')
        self.transition_next_obs = None
        super().__init__(cfg, **kwargs)

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.transition_next_obs = super()._get_observations()['policy'].detach().clone()
        return reward
