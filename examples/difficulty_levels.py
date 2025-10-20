"""
Example demonstrating different difficulty levels.

RealtimeGym environments support three difficulty levels:
- v0: Easy
- v1: Medium
- v2: Hard
"""

import realtimegym


class AdaptiveAgent:
    """Agent that adapts to different difficulty levels."""

    def __init__(self):
        self.current_observation = None
        self.action = "S"

    def observe(self, observation):
        """Receive observation."""
        self.current_observation = observation

    def think(self, timeout=None):
        """Make decision."""
        if self.current_observation is None:
            return

        turn = self.current_observation.get("game_turn", 0)
        # Alternate between moving up and staying
        self.action = "U" if turn % 2 == 0 else "S"

    def act(self):
        """Return action."""
        return self.action


def test_difficulty_level(env_id, max_steps=15):
    """Test a specific difficulty level."""
    print(f"\nTesting {env_id}")
    print("-" * 60)

    env, seed, _ = realtimegym.make(env_id, seed=0, render=False)
    agent = AdaptiveAgent()
    DEFAULT_ACTION = "S"

    obs, done = env.reset()
    step_count = 0
    total_reward = 0

    while not done and step_count < max_steps:
        step_count += 1
        agent.observe(obs)
        agent.think(timeout=8192)
        action = agent.act() or DEFAULT_ACTION
        obs, done, reward = env.step(action)
        total_reward += reward

    print(f"  Completed: {step_count} steps")
    print(f"  Total reward: {total_reward}")
    print(f"  Game finished: {done}")

    return step_count, total_reward


def main():
    """Compare all difficulty levels."""
    print("=" * 60)
    print("RealtimeGym - Difficulty Levels Example")
    print("=" * 60)

    games = ["Freeway", "Snake", "Overcooked"]

    for game in games:
        print(f"\n{'=' * 60}")
        print(f"{game} - All Difficulty Levels")
        print("=" * 60)

        results = {}
        for difficulty, level in [
            ("Easy", "v0"),
            ("Medium", "v1"),
            ("Hard", "v2"),
        ]:
            env_id = f"{game}-{level}"
            steps, reward = test_difficulty_level(env_id)
            results[difficulty] = {"steps": steps, "reward": reward}

        print(f"\nSummary for {game}:")
        print("-" * 60)
        for difficulty, result in results.items():
            print(
                f"  {difficulty:8s}: {result['steps']:2d} steps, "
                f"reward={result['reward']}"
            )

    print("\n" + "=" * 60)
    print("All difficulty levels tested!")
    print("=" * 60)


if __name__ == "__main__":
    main()
