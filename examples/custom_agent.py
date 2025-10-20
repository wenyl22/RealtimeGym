"""
Example showing how to create a custom agent.

This demonstrates:
1. Subclassing or implementing the agent interface
2. Using environment observations effectively
3. Implementing custom thinking logic
4. Managing state across steps
"""

import realtimegym


class StatefulAgent:
    """
    A custom agent that maintains state across steps.

    This agent demonstrates:
    - Storing observation history
    - Making decisions based on past observations
    - Using timeout information
    """

    def __init__(self, memory_size=5):
        self.current_observation = None
        self.observation_history = []
        self.memory_size = memory_size
        self.action_counts = {"U": 0, "D": 0, "L": 0, "R": 0, "S": 0, "I": 0}
        self.chosen_action = "S"

    def observe(self, observation):
        """
        Receive and store observation.

        Args:
            observation (dict): Contains 'state_string', 'game_turn', etc.
        """
        self.current_observation = observation

        # Store observation in history (limited size)
        self.observation_history.append(observation)
        if len(self.observation_history) > self.memory_size:
            self.observation_history.pop(0)

    def think(self, timeout=None):
        """
        Decide on action based on current and past observations.

        Args:
            timeout: Time/token budget for thinking
        """
        if self.current_observation is None:
            return

        # Example: Use turn number and history to decide
        turn = self.current_observation.get("game_turn", 0)
        _ = self.current_observation.get("state_string", "")

        # Strategy: Analyze observation history
        if len(self.observation_history) < 2:
            # Not enough history, be cautious
            self.chosen_action = "S"
        else:
            # Look for patterns in recent turns
            if turn % 3 == 0:
                self.chosen_action = "U"
            elif turn % 3 == 1:
                self.chosen_action = "D"
            else:
                self.chosen_action = "S"

        # Track action choices
        if self.chosen_action in self.action_counts:
            self.action_counts[self.chosen_action] += 1

    def act(self):
        """
        Return the chosen action.

        Returns:
            str or None: Action to take, or None for default
        """
        return self.chosen_action

    def get_statistics(self):
        """Return statistics about agent's behavior."""
        return {
            "total_observations": len(self.observation_history),
            "action_counts": self.action_counts.copy(),
            "most_used_action": max(self.action_counts, key=self.action_counts.get),
        }


def main():
    """Run example with custom agent."""
    print("=" * 60)
    print("RealtimeGym - Custom Agent Example")
    print("=" * 60)

    # Create environment
    print("\n1. Creating environment...")
    env, seed, _ = realtimegym.make("Freeway-v0", seed=42, render=False)
    print(f"   Environment created (seed={seed})")

    # Create custom agent
    print("\n2. Creating custom stateful agent...")
    agent = StatefulAgent(memory_size=10)
    DEFAULT_ACTION = "S"

    # Run game
    print("\n3. Running game with custom agent...\n")
    obs, done = env.reset()
    step_count = 0
    max_steps = 20
    total_reward = 0

    while not done and step_count < max_steps:
        step_count += 1

        # Agent processes observation
        agent.observe(obs)

        # Agent thinks (with timeout)
        agent.think(timeout=8192)

        # Agent acts
        action = agent.act() or DEFAULT_ACTION

        # Environment steps
        obs, done, reward = env.step(action)
        total_reward += reward

        if step_count % 5 == 0:
            print(f"Step {step_count}: Action={action}, Reward={reward}")

    # Show results
    print("\n4. Game completed!")
    print(f"   Total steps: {step_count}")
    print(f"   Total reward: {total_reward}")

    # Show agent statistics
    print("\n5. Agent statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
