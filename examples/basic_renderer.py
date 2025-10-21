"""
Basic usage example for RealtimeGym.

This example demonstrates:
1. Creating an environment using the make() function
2. Implementing a simple agent
3. Running the basic game loop
"""

import realtimegym
import pygame
from PIL import Image


class SimpleAgent:
    """A simple rule-based agent for demonstration."""

    def __init__(self, default_action="S"):
        self.current_observation = None
        self.default_action = default_action
        self.action = default_action

    def observe(self, observation):
        """Receive observation from environment."""
        self.current_observation = observation
        print(f"  Agent observed turn {observation['game_turn']}")

    def think(self, timeout=None):
        """Decide on an action based on observation."""
        if self.current_observation is None:
            return

        # Simple strategy: alternate actions based on turn number
        turn = self.current_observation.get("game_turn", 0)
        self.action = "L" if turn % 2 == 0 else "U"
        print(f"  Agent decided to take action: {self.action}")

    def act(self):
        """Return the chosen action."""
        return self.action


def main():
    """Run a simple game with the basic agent."""
    print("=" * 60)
    print("RealtimeGym - Basic Usage Example")
    print("=" * 60)

    # Create environment
    print("\n1. Creating Snake environment...")
    env, seed, renderer = realtimegym.make("Snake-v0", seed=10, render=True)
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

    surfaces = []
    while not done and step_count < max_steps:
        step_count += 1
        print(f"\nStep {step_count}:")
        print(f"Observation: \n{obs['state_string']}")
        surfaces.append(renderer.render(env))
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

    print(f"\n4. Simulation Ended:")
    print(f"   Total steps: {step_count}")
    print(f"   Current reward: {total_reward}")
    print(f"   Game completed: {done}")
    print("=" * 60)
    print("\n5. Rendering the game trajectory...")
    gif_path = "examples/game_trajectory.gif"
    images = [pygame.surfarray.array3d(surface) for surface in surfaces]
    pil_images = [
        Image.fromarray(images[i].swapaxes(0, 1)) for i in range(len(images))
    ]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000,
        loop=0,
    )

if __name__ == "__main__":
    main()
