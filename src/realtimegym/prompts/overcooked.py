"""Overcooked game prompts - Python logic for state-to-description conversion."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from realtimegym.environments.overcooked import (
    Recipe,
    orientation_to_char_mapping,
)

# Module-level constants
ALL_ACTIONS = "UDLRIS"
DEFAULT_ACTION = "S"

# Load prompt templates from YAML (in same directory as this module)
_TEMPLATE_FILE = Path(__file__).parent / "overcooked.yaml"

with open(_TEMPLATE_FILE, "r") as f:
    _TEMPLATES = yaml.safe_load(f)

# Export prompt templates as module-level constants for compatibility
SLOW_AGENT_PROMPT = _TEMPLATES["slow_agent_prompt"]
ACTION_FORMAT_PROMPT = _TEMPLATES["action_format_prompt"]
CONCLUSION_FORMAT_PROMPT = _TEMPLATES["conclusion_format_prompt"]
FAST_AGENT_PROMPT = _TEMPLATES["fast_agent_prompt"]
GAME_STATE_PROMPT = _TEMPLATES["game_state_prompt"]


def state_to_description(
    state_for_llm: Dict[str, Any], mode: Optional[str] = None
) -> Union[str, Dict[str, str]]:
    """Convert game state to natural language description.

    Args:
        state_for_llm: Dictionary containing the game state information
        mode: Agent mode - "reactive", "planning", or "agile"

    Returns:
        String description for reactive/planning modes, or dict with both for agile mode
    """
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
            [ingredient for ingredient in ingredients if ingredient == Recipe.TOMATO]
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
            sum([ingredient["position"] != pot_id for ingredient in ingredients]) == 0
        ), f"No ingredients found in pot {pot_id}."
        ingredients = [ingredient["name"] for ingredient in ingredients]
        num_onions = len(
            [ingredient for ingredient in ingredients if ingredient == Recipe.ONION]
        )
        num_tomatoes = len(
            [ingredient for ingredient in ingredients if ingredient == Recipe.TOMATO]
        )
        if len(ingredients) == 0:
            ingredients_str = "nothing"
        else:
            ingredients_str = f"{num_onions} onions and {num_tomatoes} tomatoes"
        if soup["is_idle"]:
            state = "Pot is not full thus cooking hasn't started yet."
        elif soup["is_cooking"]:
            state = f"Cooked for {soup['cooking_tick']} turns, still need {soup['cook_time'] - soup['cooking_tick']} turns to finish."
        elif soup["is_ready"]:
            state = "Ready to serve."
        pot_state[pot_id] = f"Pot on {pot_id}: contains {ingredients_str}; {state}"

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

    if mode == "reactive":
        return FAST_AGENT_PROMPT + model1_description
    elif mode == "planning":
        return SLOW_AGENT_PROMPT + ACTION_FORMAT_PROMPT + model2_description
    elif mode == "agile":
        return {
            "planning": SLOW_AGENT_PROMPT
            + CONCLUSION_FORMAT_PROMPT
            + model2_description,
            "reactive": FAST_AGENT_PROMPT + model1_description,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
