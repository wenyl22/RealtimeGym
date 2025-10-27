"""
Basic usage example for RealtimeGym.

This example demonstrates:
1. Creating an environment using the make() function
2. Implementing a simple agent
3. Running the basic game loop
"""

from typing import Any, Optional

import realtimegym


class SimpleAgent:
    """A simple rule-based agent for demonstration."""

    def __init__(self, default_action: str = "S") -> None:
        self.current_observation: Optional[dict[str, Any]] = None
        self.default_action: str = default_action
        self.action: str = default_action

    def observe(self, observation: dict[str, Any]) -> None:
        """Receive observation from environment."""
        self.current_observation = observation
        print(f"  Agent observed turn {observation['game_turn']}")

    def think(self, timeout: Optional[int] = None) -> None:
        """Decide on an action based on observation."""
        if self.current_observation is None:
            return

        # Simple strategy: alternate actions based on turn number
        turn = self.current_observation.get("game_turn", 0)
        self.action = "U" if turn % 2 == 0 else "S"
        print(f"  Agent decided to take action: {self.action}")

    def act(self) -> str:
        """Return the chosen action."""
        return self.action


def main() -> None:
    """Run a simple game with the basic agent."""
    print("=" * 60)
    print("RealtimeGym - Basic Usage Example")
    print("=" * 60)

    # Create environment
    print("\n1. Creating Freeway environment...")
    env, seed, renderer = realtimegym.make("Freeway-v0", seed=0, render=False)
    print(f"   Environment created with seed: {seed}")

    # Create agent
    print("\n2. Creating simple agent...")
    agent = SimpleAgent(default_action="S")
    DEFAULT_ACTION = "S"

    # Run game loop
    print("\n3. Running game loop...")
    obs, done = env.reset()
    print(f"   Initial observation: Turn {obs['game_turn']}")

    step_count = 0
    max_steps = 10
    total_reward = 0

    while not done and step_count < max_steps:
        step_count += 1
        print(f"\nStep {step_count}:")
        print(f"Observation: \n{obs['state_string']}")

        # Agent observes
        agent.observe(obs)

        # Agent thinks
        agent.think(timeout=8192)

        # Agent acts
        action = agent.act()
        if action is None:
            action = DEFAULT_ACTION

        # Environment steps
        obs, done, reward, reset = env.step(action)
        total_reward = reward

        print(f"  Reward: {reward}, Done: {done}, Reset: {reset}")

    print("\n4. Simulation Ended:")
    print(f"   Total steps: {step_count}")
    print(f"   Current reward: {total_reward}")
    print(f"   Game completed: {done}")
    print("=" * 60)


if __name__ == "__main__":
    main()
