# Third-party code notice:
# This module uses vendored code from Overcooked-AI (https://github.com/HumanCompatibleAI/overcooked_ai)
# See ../overcooked_new/THIRD_PARTY_NOTICE.md for license and attribution details.

from ..overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,  # noqa: F401
)
from ..overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)


class OvercookedRender:
    def __init__(self):
        self.visualizer = StateVisualizer()

    def render(self, env):
        surface = self.visualizer.render_state(
            env.gym_env.base_env.state,
            env.gym_env.base_mdp.terrain_mtx,
            hud_data={},
        )
        return surface
