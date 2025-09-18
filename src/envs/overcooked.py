import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minatar.environment import Environment
from minatar.environments.overcooked import Env
import sys
import os
from copy import deepcopy
from pathlib import Path
from minatar.environments.overcooked_new.config import get_config
from minatar.environment import Environment
from minatar.environments.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import Recipe
difficulty_layout_mapping = {
    'E': 'cc_easy',
    'M': 'cc_hard',
    'H': 'cc_insane',
}
def pair(seed, difficulty):
    # if difficulty == 'E':
    #     if seed % 2 == 0:
    #         return "script:place_tomato_in_pot"
    #     else:
    #         return "script:place_tomato_and_deliver_soup"
    # if difficulty == 'M':
    #     if seed % 2 == 0:
    #         return "script:place_tomato_in_pot"
    #     else:
    #         return "script:place_tomato_and_deliver_soup"
    # if difficulty == 'H':
    #     if seed % 2 == 0:
    #         return "script:put_onion_everywhere"
    #     else:
    #         return "script:put_dish_everywhere"
    return "script:put_onion_everywhere"

orientation_to_char_mapping = {
    (0, 1): 'U',  # Up
    (0, -1): 'D',  # Down
    (-1, 0): 'L',  # Left
    (1, 0): 'R',  # Right
}

def setup_env(seed, difficulty):
    parser = get_config()
    all_args = parse_args([], parser)
    all_args.layout_name = difficulty_layout_mapping[difficulty]
    all_args.env_name = "overcooked"
    all_args.algorithm_name = "population"
    all_args.agent0_policy_name = "script:LLM"
    all_args.agent1_policy_name = pair(seed, difficulty)
    all_args.episode_length = 100
    all_args.num_agents = 2
    run_dir = Path("vislogs/overcooked-vislogs") / all_args.layout_name
    if not os.path.exists(str(run_dir)):
        os.makedirs(str(run_dir), exist_ok=True)
    env = Environment('overcooked', sticky_action_prob=0.0)
    env.seed(seed)
    env.env.all_args = all_args
    env.env.run_dir = run_dir

    env.reset()
    return env, seed

def llm_state_builder(env: Env):
    """
    "players": [p.to_dict() for p in self.players]
        - position, orientation, held_object
    "objects": [obj.to_dict() for obj in self.objects.values()]
        - Object can be soup or put on the counter X.
        - name, position
        - (SoupState): _ingredients, cooking_tick, is_cooking, is_ready, is_idle, cook_time
    "all_orders" : [order.to_dict() for order in self.all_orders]
    """
    # --- State Information --- #
    state = env.gym_env.base_env.state.to_dict()
    # --- Layout Information --- #
    all_order_info = env.gym_env.base_env.state.all_order_info()
    terrain = env.gym_env.base_mdp.terrain_pos_dict
    state_for_llm = {
        "history": env.history if len(env.history[0]) <= 5 else [env.history[0][-5:], env.history[1][-5:]],
        "game_turn": env.game_turn,
        "state": state,
        "all_orders": all_order_info,
        "layout": terrain,
    }
    return state_for_llm
from prompts.overcooked import GAME_STATE_PROMPT
def state_to_description(state_for_llm, scratch_pad=None, fast = False):
    kitchen_counters = state_for_llm['layout']['X']
    tomatoes = state_for_llm['layout']['T']
    onions = state_for_llm['layout']['O']
    plates = state_for_llm['layout']['D']
    pots = state_for_llm['layout']['P']
    serving_counters = state_for_llm['layout']['S']
    recipe_infos = state_for_llm['all_orders']
    text_recipe_infos = ""
    for i, recipe in enumerate(recipe_infos):
        ingredients = recipe['ingredients']
        num_onions = len([ingredient for ingredient in ingredients if ingredient == Recipe.ONION])
        num_tomatoes = len([ingredient for ingredient in ingredients if ingredient == Recipe.TOMATO])
        reward = recipe['value']
        time = recipe['time']
        text_recipe_infos += f"Recipe {i+1}: {num_onions} onions, {num_tomatoes} tomatoes; reward: {reward}; time to cook: {time} turns\n"
    position = [0, 0]
    orientation = [0, 0]
    held_object = [0, 0]
    history = [0, 0]
    for i in range(2):
        player = state_for_llm['state']['players'][i]
        position[i] = player['position']
        orientation[i] = orientation_to_char_mapping[player['orientation']]
        held_object[i] = deepcopy(player['held_object'])
        if len(state_for_llm['history'][i]) > 0:
            history[i] = ", ".join(state_for_llm['history'][i])
        else:
            history[i] = "No action history"
        if held_object[i] is not None:
            held_object[i] = "one " + held_object[i]['name']
            if held_object == "dish":
                held_object[i] = "clean plate"
            elif held_object[i] == "soup":
                held_object[i] = "soup in plate"
        else:
            held_object[i] = "nothing"
    pot_state = {}
    kitchen_counter_state = {}
    for soup in state_for_llm['state']['objects']:
        pot_id = soup['position']
        if pot_id in kitchen_counters:
            kitchen_counter_state[pot_id] = f"Kitchen Counter on {pot_id}: contains a {soup['name'].replace("dish", "clean plate")}; "
            continue
        if pot_id not in pots:
            assert pot_id in position, f"{pot_id} not in a valid spot"
            continue
        assert soup['name'] == 'soup', f"Object {soup['name']} is not a soup."
        ingredients = soup['_ingredients']
        assert sum([ingredient['position'] != pot_id for ingredient in ingredients]) == 0, f"No ingredients found in pot {pot_id}."
        ingredients = [ingredient['name'] for ingredient in ingredients]
        num_onions = len([ingredient for ingredient in ingredients if ingredient == Recipe.ONION])
        num_tomatoes = len([ingredient for ingredient in ingredients if ingredient == Recipe.TOMATO])
        if len(ingredients) == 0:
            ingredients = "nothing"
        else:
            ingredients = f"{num_onions} onions and {num_tomatoes} tomatoes"
        if soup['is_idle']:
            state = "Pot is not full thus cooking hasn't started yet."
        elif soup['is_cooking']:
            state = f"Cooked for {soup['cooking_tick']} turns, still need {soup['cook_time'] - soup['cooking_tick']} turns to finish."
        elif soup['is_ready']:
            state = "Ready to serve."
        pot_state[pot_id] = f"Pot on {pot_id}: contains {ingredients}; {state}"   
    text_kitchen_counter_state = "\n".join(kitchen_counter_state.values())
    if text_kitchen_counter_state == "":
        text_kitchen_counter_state = "All kitchen counters are empty."
    text_pot_state = "\n".join(pot_state.values())
    if text_pot_state == "":
        text_pot_state = "All pots are empty."
    game_turn = state_for_llm['game_turn']

    description = GAME_STATE_PROMPT.format(
        kitchen_counter = kitchen_counters,
        tomato = tomatoes if len(tomatoes) > 0 else "No tomato dispensers",
        onion = onions,
        plate = plates,
        pot = pots,
        serving_counter = serving_counters,
        recipe_infos = text_recipe_infos,
        t_format = f"t_{0 if fast else 1} = {game_turn}",
        my_position = position[0],
        my_orientation = orientation[0],
        my_holding = held_object[0],
        my_action_history = history[0],
        he_position = position[1],
        he_orientation = orientation[1],
        he_holding = held_object[1],
        he_action_history = history[1],
        kitchen_counter_state = text_kitchen_counter_state,
        pot_state = text_pot_state,
    )
        
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description

def summarize(seed, difficulty, env):
    print(f"Seed {seed} - {difficulty_layout_mapping[difficulty]} turn: {env.env.game_turn}, reward: {env.env.reward}")


def parse_args(args, parser):
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--overcooked_version", default="new", type=str, choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float)
    parser.add_argument("--store_traj", default=False, action='store_true')
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    
    # overcooked evaluation
    parser.add_argument("--agent0_policy_name", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", type=str, help="policy name of agent 1")

    all_args = parser.parse_known_args(args)[0]

    return all_args
