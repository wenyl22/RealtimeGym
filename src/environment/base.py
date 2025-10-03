import numpy as np
class BaseEnv():
    def __init__(self):
        self.random = np.random.RandomState()
        self.seed = 42
    def set_seed(self, seed):
        self.random = np.random.RandomState(seed)
        self.seed = seed
    def reset(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    def act(self, a):
        """
        Take action a, return (reward, terminal, reset)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def state_string(self):
        """
        Visualization in string format.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def observe(self):
        """
        Get the current observation, including
        1. description: in natural language
        2. state_string: visualization in string format
        3. game_turn: current game turn
        """
        raise NotImplementedError("This method should be overridden by subclasses.")