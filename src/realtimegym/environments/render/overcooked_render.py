from ..overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ..overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import *

class OvercookedRender:
    def __init__(self):
        self.visualizer = StateVisualizer()

    def render(self, env):
        surface = self.visualizer.render_state(env.gym_env.base_env.state, env.gym_env.base_mdp.terrain_mtx, hud_data={})
        return surface