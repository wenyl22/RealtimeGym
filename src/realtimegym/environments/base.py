from typing import Any, NoReturn

import numpy as np


class BaseEnv:
    def __init__(self) -> None:
        self.random = np.random.RandomState()
        self.seed = 42
        self.terminal = False
        self.game_turn = 0
        self.reward = 0

    def set_seed(self, seed: int) -> None:
        self.random = np.random.RandomState(seed)
        self.seed = seed

    def reset(self) -> NoReturn:
        """
        Reset the environment and return initial observation and done flag.

        Returns:
            obs (dict): Initial observation with keys like 'description', 'state_string', 'game_turn'
            done (bool): Whether the episode is done (should be False after reset)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, action: str) -> NoReturn:
        """
        Execute action in the environment.

        Args:
            action: The action to execute

        Returns:
            obs (dict): New observation after taking the action
            done (bool): Whether the episode is done
            reward (float/int): Reward obtained
            reset (bool): Whether the environment needs to be reset
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def state_string(self) -> NoReturn:
        """
        Visualization in string format.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def state_builder(self) -> NoReturn:
        """
        Build a state representation for agents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def observe(self) -> dict[str, Any]:
        """
        Get the current observation.
        """
        if self.terminal:
            return {}
        return {
            "state_string": self.state_string(),
            "game_turn": self.game_turn,
            "state": self.state_builder(),
        }
