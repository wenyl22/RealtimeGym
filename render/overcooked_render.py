import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from envs.minatar.environments.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from envs.minatar.environments.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import *
from envs.overcooked import setup_env
import pygame
from PIL import Image
def surface_to_pil(surface):
    data = pygame.image.tostring(surface, 'RGBA')
    img = Image.frombytes('RGBA', surface.get_size(), data)
    return img.convert('RGB')
visualizer = StateVisualizer()

def turn_traj_to_gif(traj_path):
    seed = int(traj_path.split('_')[-1].split('.')[0])
    difficulty = 'M' if 'M' in traj_path else 'E' if 'E' in traj_path else 'H' if 'H' in traj_path else 'I'
    env, smp = setup_env(seed = seed, difficulty = difficulty)
    grid = env.env.gym_env.base_mdp.terrain_mtx
    df = pd.read_csv(traj_path)
    imgs = []
    for a in df['action']:
        hud_data = {}
        result = visualizer.render_state(env.env.gym_env.base_env.state, grid, hud_data = hud_data)
        img = surface_to_pil(result)
        imgs.append(img)
        env.act(a)
    result = visualizer.render_state(env.env.gym_env.base_env.state, grid, hud_data = hud_data)
    img = surface_to_pil(result)
    imgs.append(img)
    imgs[0].save(
        traj_path.replace('.csv', '.gif'),
        save_all=True,
        append_images=imgs[1:],
        duration=500,
        loop=1
    )
