from envs.minatar.environment import Environment
from envs.minatar.environments.snake import Env
from copy import deepcopy
seed_mapping = {
    'E': {i: 1000 + i for i in range(32)},
    'M': {i: 5000 + i for i in range(32)},
    'H': {i: 8000 + i for i in range(32)},
}
def setup_env(seed, difficulty):
    env = Environment("snake", sticky_action_prob=0)
    env.seed(seed_mapping[difficulty][seed])
    env.reset()
    return env, seed_mapping[difficulty][seed]

def summarize(seed, difficulty, env):
    print(f"Seed {seed} - {env.env.game_turn} turns, reward: {env.env.reward}")
    return False

def llm_state_builder(env: Env):
    snake = deepcopy(env.snake[::-1])
    foods = []
    for (x, y) in env.food:
        l, v = env.food_attributes[x][y]
        foods.append((x, y, l, v))
    return {
        "turn": env.game_turn,
        "snake_dir": env.dir,
        "internal_obstacles": env.obstacle,
        "foods": foods,
        "snake": snake,
        "size": env.B
    }

def state_to_description(state_for_llm, scratch_pad = None, fast = False):
    description = f"**Current Turn**: \( t_{0 if fast else 1} = {state_for_llm['turn']} \)\n"
    description += f"**Cells occupied by walls**:\n"
    description += f"\t - Border Cells: x=0/x={state_for_llm['size'] - 1} or y=0/y={state_for_llm['size'] - 1}.\n"
    description += f"\t - Internal Obstacles: {state_for_llm['internal_obstacles'] if len(state_for_llm['internal_obstacles']) > 0 else 'No internal obstacles'}\n"
    description += f"**Snake Positions**:{state_for_llm['snake']}\n**Snake Head Direction**: {state_for_llm['snake_dir']}\n"
    description += f"**Food Positions, Life Span and Value**:\n"

    for (x, y, life_span, value) in state_for_llm['foods']:
        description += f"\t- ({x}, {y}, {life_span}, {value})\n"
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
