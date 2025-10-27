"""
Example demonstrating all three environments in RealtimeGym.

This shows how to use Freeway, Snake, and Overcooked environments
with the same agent interface.
"""

from typing import Any, Optional

import realtimegym


class UniversalAgent:
    """An agent that works with all environments."""

    def __init__(self) -> None:
        self.current_observation: Optional[dict[str, Any]] = None
        self.action: str = "S"
        self.step_count: int = 0

    def observe(self, observation: dict[str, Any]) -> None:
        """Receive observation from environment."""
        self.current_observation = observation
        self.step_count += 1

    def think(self, timeout: Optional[int] = None) -> None:
        """Simple decision making."""
        if self.current_observation is None:
            return

        # Cycle through common actions
        actions = ["U", "D", "L", "R", "S"]
        self.action = actions[self.step_count % len(actions)]

    def act(self) -> str:
        """Return the chosen action."""
        return self.action


def run_environment(env_id: str, max_steps: int = 10) -> None:
    """Run a single environment with the agent."""
    print(f"\n{'=' * 60}")
    print(f"Running {env_id}")
    print("=" * 60)

    # Create environment
    env, seed, _ = realtimegym.make(env_id, seed=0, render=False)
    print(f"Environment: {env_id} (seed={seed})")

    # Create agent
    agent = UniversalAgent()
    DEFAULT_ACTION = "S"

    # Run game loop
    obs, done = env.reset()
    print(f"Initial state:\n{obs['state_string']}")

    step_count = 0
    total_reward = 0

    while not done and step_count < max_steps:
        step_count += 1
        # Agent loop
        agent.observe(obs)
        agent.think(timeout=8192)
        action = agent.act() or DEFAULT_ACTION

        # Environment step
        obs, done, reward, reset = env.step(action)
        total_reward = reward

        print(f"Step {step_count}: Action={action}, Reward={reward}, Done={done}")

    print("\nResults:")
    print(f"  Steps: {step_count}")
    print(f"  Total Reward: {total_reward}")
    print(f"  Completed: {done}")


def main() -> None:
    """Run all three environments."""
    print("\n" + "=" * 60)
    print("RealtimeGym - All Environments Example")
    print("=" * 60)

    # Test each environment
    environments = [
        "Freeway-v0",
        "Snake-v0",
        "Overcooked-v0",
        "Freeway-v1",
        "Snake-v1",
        "Overcooked-v1",
        "Freeway-v2",
        "Snake-v2",
        "Overcooked-v2",
    ]

    for env_id in environments:
        run_environment(env_id, max_steps=5)

    print("\n" + "=" * 60)
    print("All environments tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
