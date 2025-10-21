import numpy as np


class BaseEnv:
    def __init__(self):
        self.random = np.random.RandomState()
        self.seed = 42
        self.terminal = False
        self.game_turn = 0
        self.reward = 0

    def set_seed(self, seed):
        self.random = np.random.RandomState(seed)
        self.seed = seed

    def reset(self):
        """
        Reset the environment and return initial observation and done flag.

        Returns:
            obs (dict): Initial observation with keys like 'description', 'state_string', 'game_turn'
            done (bool): Whether the episode is done (should be False after reset)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, action):
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

    def state_string(self):
        """
        Visualization in string format.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    # Legacy methods for backward compatibility
    def act(self, a):
        """
        Legacy method. Use step() instead.

        """
        raise NotImplementedError(
            "This method has been deprecated. Use step() instead."
        )

    def observe(self):
        """
        Legacy method. Now step() returns observations directly.
        Get the current observation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
