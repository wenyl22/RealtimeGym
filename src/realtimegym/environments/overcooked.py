# This is Overcooked environment for one player cooperated with a script agent.
#
# Third-party code notice:
# This module uses vendored code from Overcooked-AI (https://github.com/HumanCompatibleAI/overcooked_ai)
# See overcooked_new/THIRD_PARTY_NOTICE.md for license and attribution details.

import os
from copy import deepcopy
from pathlib import Path

from .base import BaseEnv
from .overcooked_new.config import get_config
from .overcooked_new.Overcooked_Env import Overcooked  # type: ignore
from .overcooked_new.src.overcooked_ai_py.mdp.actions import (  # type: ignore
    Action,
    Direction,
)
from .overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    Recipe,  # type: ignore
)
from .prompts.overcooked import GAME_STATE_PROMPT
from .render.overcooked_render import OvercookedRender

cognitive_load_layout_mapping = {
    "E": "cc_easy",
    "M": "cc_hard",
    "H": "cc_insane",
}
orientation_to_char_mapping = {
    (0, 1): "U",  # Up
    (0, -1): "D",  # Down
    (-1, 0): "L",  # Left
    (1, 0): "R",  # Right
}


def parse_args(args, parser):
    parser.add_argument(
        "--layout_name",
        type=str,
        default="cramped_room",
        help="Name of Submap, 40+ in choice. See /src/data/layouts/.",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument(
        "--initial_reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_horizon",
        type=int,
        default=2.5e6,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument(
        "--use_agent_policy_id",
        default=False,
        action="store_true",
        help="Add policy id into share obs, default False",
    )
    parser.add_argument(
        "--overcooked_version", default="new", type=str, choices=["new", "old"]
    )
    parser.add_argument(
        "--use_detailed_rew_shaping", default=False, action="store_true"
    )
    parser.add_argument("--random_start_prob", default=0.0, type=float)
    parser.add_argument("--store_traj", default=False, action="store_true")
    # population
    parser.add_argument(
        "--population_yaml_path",
        type=str,
        help="Path to yaml file that stores the population info.",
    )

    # overcooked evaluation
    parser.add_argument("--agent0_policy_name", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", type=str, help="policy name of agent 1")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def setup_env(seed, cognitive_load, save_trajectory_gifs=False):
    parser = get_config()
    all_args = parse_args([], parser)
    all_args.layout_name = cognitive_load_layout_mapping[cognitive_load]
    all_args.env_name = "overcooked"
    all_args.algorithm_name = "population"
    all_args.agent0_policy_name = "script:LLM"
    all_args.agent1_policy_name = "script:put_onion_everywhere"
    all_args.episode_length = 100
    all_args.num_agents = 2
    run_dir = Path("vislogs/overcooked-vislogs") / all_args.layout_name
    if not os.path.exists(str(run_dir)):
        os.makedirs(str(run_dir), exist_ok=True)
    env = OvercookedEnv()
    env.set_seed(seed)
    env.all_args = all_args
    env.run_dir = run_dir
    render = None
    if save_trajectory_gifs:
        render = OvercookedRender()
    return env, seed, render


class OvercookedEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.all_args = None  # type: ignore
        self.run_dir = None  # type: ignore

    def reset(self):
        self.gym_env = Overcooked(
            self.all_args, self.run_dir, featurize_type=("bc", "bc")
        )
        eval_obs, _, _ = self.gym_env.reset(True)

        self.reward = 0
        self.game_turn = 0
        self.terminal = False
        self.history = [
            [],
            [],
        ]  # history[0] for player 0, history[1] for player 1

        # self.eval_env_infos = defaultdict(list)
        # Return initial observation and done flag
        return self.observe(), self.terminal

    def go(self, a):
        self.game_turn += 1

        if a == "U":
            action = Direction.SOUTH
        elif a == "D":
            action = Direction.NORTH
        elif a == "L":
            action = Direction.WEST
        elif a == "R":
            action = Direction.EAST
        elif a == "I":
            action = Action.INTERACT
        else:
            action = Action.STAY
        self.gym_env.script_agent[0].next_action = action
        (
            eval_ob,
            eval_share_ob,
            eval_reward,
            eval_done,
            eval_info,
            eval_available_action,
            joint_action,
        ) = self.gym_env.step([[0], [0]])
        self.reward += sum(eval_reward[0])
        self.terminal = eval_done[0]
        self.history[0].append(Action.A_TO_CHAR[joint_action[0]])
        self.history[1].append(Action.A_TO_CHAR[joint_action[1]])
        return self.observe(), self.terminal, self.reward

    def step(self, a):
        """Legacy method for backward compatibility."""
        obs, done, reward = self.go(a)
        return obs, done, reward, False

    def state_string(self):
        ret = self.gym_env.base_mdp.state_string(self.gym_env.base_env.state)
        ret = ret.split("\n")
        ret = ret[::-1]
        ret = "\n".join(ret)
        ret = ret.replace("↑", "x").replace("↓", "y")
        ret = ret.replace("x", "↓").replace("y", "↑")
        return ret

    def llm_state_builder(self):
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
        state = self.gym_env.base_env.state.to_dict()
        # --- Layout Information --- #
        all_order_info = self.gym_env.base_env.state.all_order_info()
        terrain = self.gym_env.base_mdp.terrain_pos_dict
        state_for_llm = {
            "history": self.history
            if len(self.history[0]) <= 5
            else [self.history[0][-5:], self.history[1][-5:]],
            "game_turn": self.game_turn,
            "state": state,
            "all_orders": all_order_info,
            "layout": terrain,
        }
        return state_for_llm

    def observe(self):
        state_for_llm = self.llm_state_builder()
        kitchen_counters = state_for_llm["layout"]["X"]
        tomatoes = state_for_llm["layout"]["T"]
        onions = state_for_llm["layout"]["O"]
        plates = state_for_llm["layout"]["D"]
        pots = state_for_llm["layout"]["P"]
        serving_counters = state_for_llm["layout"]["S"]
        recipe_infos = state_for_llm["all_orders"]
        text_recipe_infos = ""
        for i, recipe in enumerate(recipe_infos):
            ingredients = recipe["ingredients"]
            num_onions = len(
                [ingredient for ingredient in ingredients if ingredient == Recipe.ONION]
            )
            num_tomatoes = len(
                [
                    ingredient
                    for ingredient in ingredients
                    if ingredient == Recipe.TOMATO
                ]
            )
            reward = recipe["value"]
            time = recipe["time"]
            text_recipe_infos += f"Recipe {i + 1}: {num_onions} onions, {num_tomatoes} tomatoes; reward: {reward}; time to cook: {time} turns\n"
        position = [0, 0]
        orientation = [0, 0]
        held_object: list = [0, 0]
        history = [0, 0]
        for i in range(2):
            player = state_for_llm["state"]["players"][i]
            position[i] = player["position"]
            orientation[i] = orientation_to_char_mapping[player["orientation"]]
            held_object[i] = deepcopy(player["held_object"])
            if len(state_for_llm["history"][i]) > 0:
                history[i] = ", ".join(state_for_llm["history"][i])
            else:
                history[i] = "No action history"
            if held_object[i] is not None:
                held_object[i] = "one " + held_object[i]["name"]  # type: ignore
                if held_object[i] == "dish":
                    held_object[i] = "clean plate"
                elif held_object[i] == "soup":
                    held_object[i] = "soup in plate"
            else:
                held_object[i] = "nothing"
        pot_state = {}
        kitchen_counter_state = {}
        for soup in state_for_llm["state"]["objects"]:
            pot_id = soup["position"]
            if pot_id in kitchen_counters:
                kitchen_counter_state[pot_id] = (
                    f"Kitchen Counter on {pot_id}: contains a {soup['name'].replace('dish', 'clean plate')}; "
                )
                continue
            if pot_id not in pots:
                assert pot_id in position, f"{pot_id} not in a valid spot"
                continue
            assert soup["name"] == "soup", f"Object {soup['name']} is not a soup."
            ingredients = soup["_ingredients"]
            assert (
                sum([ingredient["position"] != pot_id for ingredient in ingredients])
                == 0
            ), f"No ingredients found in pot {pot_id}."
            ingredients = [ingredient["name"] for ingredient in ingredients]
            num_onions = len(
                [ingredient for ingredient in ingredients if ingredient == Recipe.ONION]
            )
            num_tomatoes = len(
                [
                    ingredient
                    for ingredient in ingredients
                    if ingredient == Recipe.TOMATO
                ]
            )
            if len(ingredients) == 0:
                ingredients = "nothing"
            else:
                ingredients = f"{num_onions} onions and {num_tomatoes} tomatoes"
            if soup["is_idle"]:
                state = "Pot is not full thus cooking hasn't started yet."
            elif soup["is_cooking"]:
                state = f"Cooked for {soup['cooking_tick']} turns, still need {soup['cook_time'] - soup['cooking_tick']} turns to finish."
            elif soup["is_ready"]:
                state = "Ready to serve."
            pot_state[pot_id] = f"Pot on {pot_id}: contains {ingredients}; {state}"
        text_kitchen_counter_state = "\n".join(kitchen_counter_state.values())
        if text_kitchen_counter_state == "":
            text_kitchen_counter_state = "All kitchen counters are empty."
        text_pot_state = "\n".join(pot_state.values())
        if text_pot_state == "":
            text_pot_state = "All pots are empty."
        game_turn = state_for_llm["game_turn"]

        model1_description = GAME_STATE_PROMPT.format(
            kitchen_counter=kitchen_counters,
            tomato=tomatoes if len(tomatoes) > 0 else "No tomato dispensers",
            onion=onions,
            plate=plates,
            pot=pots,
            serving_counter=serving_counters,
            recipe_infos=text_recipe_infos,
            t_format=f"t_0 = {game_turn}",
            my_position=position[0],
            my_orientation=orientation[0],
            my_holding=held_object[0],
            my_action_history=history[0],
            he_position=position[1],
            he_orientation=orientation[1],
            he_holding=held_object[1],
            he_action_history=history[1],
            kitchen_counter_state=text_kitchen_counter_state,
            pot_state=text_pot_state,
        )
        model2_description = GAME_STATE_PROMPT.format(
            kitchen_counter=kitchen_counters,
            tomato=tomatoes if len(tomatoes) > 0 else "No tomato dispensers",
            onion=onions,
            plate=plates,
            pot=pots,
            serving_counter=serving_counters,
            recipe_infos=text_recipe_infos,
            t_format=f"t_1 = {game_turn}",
            my_position=position[0],
            my_orientation=orientation[0],
            my_holding=held_object[0],
            my_action_history=history[0],
            he_position=position[1],
            he_orientation=orientation[1],
            he_holding=held_object[1],
            he_action_history=history[1],
            kitchen_counter_state=text_kitchen_counter_state,
            pot_state=text_pot_state,
        )
        return {
            "model1_description": model1_description,
            "model2_description": model2_description,
            "game_turn": self.game_turn,
            "state_string": self.state_string(),
        }

    def summary(self):
        print(
            f"Seed {self.seed} - {self.all_args.layout_name} turn: {self.game_turn}, reward: {self.reward}"
        )
