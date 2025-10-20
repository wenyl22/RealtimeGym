"""Real-time Reasoning Gym - A gym for evaluating language agents in dynamic environments."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

# Environment registry mapping environment IDs to module names and difficulty
_REGISTRY = {
    "Freeway-v0": ("freeway", "E"),
    "Freeway-v1": ("freeway", "M"),
    "Freeway-v2": ("freeway", "H"),
    "Snake-v0": ("snake", "E"),
    "Snake-v1": ("snake", "M"),
    "Snake-v2": ("snake", "H"),
    "Overcooked-v0": ("overcooked", "E"),
    "Overcooked-v1": ("overcooked", "M"),
    "Overcooked-v2": ("overcooked", "H"),
}


def make(env_id: str, seed: int = 0, render: bool = False):
    """
    Create an environment instance.

    Args:
        env_id: Environment identifier (e.g., 'Freeway-v0', 'Snake-v1', 'Overcooked-v2')
                Version suffixes: -v0 (Easy), -v1 (Medium), -v2 (Hard)
        seed: Random seed for environment initialization (0-7)
        render: Whether to enable rendering for trajectory visualization

    Returns:
        Tuple of (env, actual_seed, render_object)
        - env: The environment instance
        - actual_seed: The actual seed used (mapped from input seed)
        - render_object: Renderer instance if render=True, else None

    Examples:
        >>> import realtimegym
        >>> env, seed, renderer = realtimegym.make('Freeway-v0', seed=0)
        >>> obs = env.reset()
        >>> action = 'U'  # Move up
        >>> reward, done = env.act(action)
    """
    if env_id not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment ID: {env_id}. Available environments: {available}"
        )

    game_name, cognitive_load = _REGISTRY[env_id]

    # Import the environment module
    env_module: Any = import_module(f"realtimegym.environments.{game_name}")

    # Use the setup_env function from the environment module
    env, actual_seed, render_obj = env_module.setup_env(  # type: ignore
        seed=seed, cognitive_load=cognitive_load, save_trajectory_gifs=render
    )

    return env, actual_seed, render_obj


# Export commonly used classes using lazy imports
def __getattr__(name):
    """Lazy imports for agents and environments to avoid requiring all dependencies upfront."""
    if name == "BaseAgent":
        from realtimegym.agents.base import BaseAgent

        return BaseAgent
    elif name == "ReactiveAgent":
        from realtimegym.agents.reactive import ReactiveAgent

        return ReactiveAgent
    elif name == "PlanningAgent":
        from realtimegym.agents.planning import PlanningAgent

        return PlanningAgent
    elif name == "AgileThinker":
        from realtimegym.agents.agile import AgileThinker

        return AgileThinker
    elif name == "BaseEnv":
        from realtimegym.environments.base import BaseEnv

        return BaseEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "make",
    "BaseAgent",
    "ReactiveAgent",
    "PlanningAgent",
    "AgileThinker",
    "BaseEnv",
]
