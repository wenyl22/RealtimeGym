"""
Test script to verify the new API loop works correctly.

This demonstrates the desired API:
    obs, done = env.reset()
    while not done:
        agent.observe(obs)
        agent.think(timeout=T_E)
        action = agent.act() or DEFAULT_ACTION
        obs, done, reward = env.step(action)
"""

import sys

sys.path.insert(0, "src")

import realtimegym

print("=" * 60)
print("Testing New API Loop")
print("=" * 60)

# Test 1: Environment API
print("\n1. Testing Environment API (Freeway)")
print("-" * 60)

env, seed, renderer = realtimegym.make("Freeway-v0", seed=0, render=False)
print(f"✓ Environment created: {type(env).__name__}")

# Test reset() returns (obs, done)
obs, done = env.reset()
print(f"✓ reset() returns: obs (dict with {len(obs)} keys), done={done}")
assert isinstance(obs, dict), "obs should be a dict"
assert isinstance(done, bool), "done should be a bool"
assert not done, "done should be False after reset"
assert "state_string" in obs, "obs should contain 'state_string'"
print(f"  Observation keys: {list(obs.keys())}")

# Test step() returns (obs, done, reward)
action = "S"  # Stand still
obs, done, reward = env.step(action)
print(f"✓ step('{action}') returns: obs (dict), done={done}, reward={reward}")
assert isinstance(obs, dict), "obs should be a dict"
assert isinstance(done, bool), "done should be a bool"
assert isinstance(reward, (int, float)), "reward should be numeric"

print("\n2. Testing Mock Agent with New API")
print("-" * 60)


# Create a simple mock agent to test the API
class MockAgent:
    def __init__(self):
        self.current_observation = None
        self.action = "S"

    def observe(self, obs):
        """Receive observation from environment."""
        self.current_observation = obs
        print(f"  Agent observed: game_turn={obs.get('game_turn', 0)}")

    def think(self, timeout=None):
        """Process observation and decide action."""
        if self.current_observation is None:
            return
        # Simple logic: alternate between S and U
        turn = self.current_observation.get("game_turn", 0)
        self.action = "U" if turn % 2 == 0 else "S"
        print(f"  Agent thinking (timeout={timeout}): chose action '{self.action}'")

    def act(self):
        """Return the chosen action."""
        return self.action


agent = MockAgent()
DEFAULT_ACTION = "S"

# Run the new API loop
print("\n3. Running Complete API Loop")
print("-" * 60)

env, seed, renderer = realtimegym.make("Freeway-v0", seed=0, render=False)
obs, done = env.reset()
print(f"Starting game loop (seed={seed})")

step_count = 0
max_steps = 5  # Just run a few steps for testing

while not done and step_count < max_steps:
    step_count += 1
    print(f"\nStep {step_count}:")

    # agent.observe(obs)
    agent.observe(obs)

    # agent.think(timeout=T_E)
    agent.think(timeout=8192)

    # action = agent.act() or DEFAULT_ACTION
    action = agent.act()
    if action is None:
        action = DEFAULT_ACTION

    # obs, done, reward = env.step(action)
    obs, done, reward = env.step(action)
    print(f"  Result: done={done}, reward={reward}")

print(f"\n✓ Completed {step_count} steps successfully")

print("\n4. Testing All Three Environments")
print("-" * 60)

for env_id in ["Freeway-v0", "Snake-v0", "Overcooked-v0"]:
    env, seed, _ = realtimegym.make(env_id, seed=0, render=False)
    obs, done = env.reset()
    assert not done, f"{env_id}: done should be False after reset"

    # Take one step
    obs, done, reward = env.step("S")
    assert isinstance(obs, dict), f"{env_id}: step should return dict observation"
    assert isinstance(done, bool), f"{env_id}: step should return bool done"
    assert isinstance(reward, (int, float)), (
        f"{env_id}: step should return numeric reward"
    )

    print(f"✓ {env_id:20s} - API working correctly")

print("\n" + "=" * 60)
print("✅ All API tests passed!")
print("=" * 60)
print("\nThe new API loop is working correctly:")
print("  obs, done = env.reset()")
print("  while not done:")
print("      agent.observe(obs)")
print("      agent.think(timeout=T_E)")
print("      action = agent.act() or DEFAULT_ACTION")
print("      obs, done, reward = env.step(action)")
print("=" * 60)
