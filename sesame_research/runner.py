"""Official optimization with the user-selected Sesame task reward."""
from supermimic.runners.supermimic_runner import SuperMimicRunner
from sesame_research.reward import bind_imagination_reward


class SesameSuperMimicRunner(SuperMimicRunner):
    def _init_common(self):
        super()._init_common()
        bind_imagination_reward(self._representation, self.env)
