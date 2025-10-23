"""Snake game prompts - Python logic for state-to-description conversion."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Module-level constants
ALL_ACTIONS = "LRUD"
DEFAULT_ACTION = "S"

# Load prompt templates from YAML (in same directory as this module)
_TEMPLATE_FILE = Path(__file__).parent / "snake.yaml"

with open(_TEMPLATE_FILE, "r") as f:
    _TEMPLATES = yaml.safe_load(f)

# Export prompt templates as module-level constants for compatibility
SLOW_AGENT_PROMPT = _TEMPLATES["slow_agent_prompt"]
ACTION_FORMAT_PROMPT = _TEMPLATES["action_format_prompt"]
CONCLUSION_FORMAT_PROMPT = _TEMPLATES["conclusion_format_prompt"]
FAST_AGENT_PROMPT = _TEMPLATES["fast_agent_prompt"]


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
    game_turn = state_for_llm["game_turn"]
    description = "**Cells occupied by walls**:\n"
    description += f"\t - Border Cells: x=0/x={state_for_llm['size'] - 1} or y=0/y={state_for_llm['size'] - 1}.\n"
    description += f"\t - Internal Obstacles: {state_for_llm['internal_obstacles'] if len(state_for_llm['internal_obstacles']) > 0 else 'No internal obstacles'}\n"
    description += f"**Snake Positions**:{state_for_llm['snake']}\n**Snake Head Direction**: {state_for_llm['snake_dir']}\n"
    description += "**Food Positions, Life Span and Value**:\n"
    for x, y, life_span, value in state_for_llm["foods"]:
        description += f"\t- ({x}, {y}, {life_span}, {value})\n"

    model1_description = f"**Current Turn**: \( t_0 = {game_turn} \)\n" + description
    model2_description = f"**Current Turn**: \( t_1 = {game_turn} \)\n" + description

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
