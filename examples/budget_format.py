"""
Example demonstrating budget format usage in RealtimeGym.

RealtimeGym supports two budget formats:
1. "token" - Budget measured in tokens (for LLM agents)
2. "time" - Budget measured in seconds (for time-constrained thinking)

The budget format controls how the agent's thinking time is managed.
"""

import time

import realtimegym

# Note: Budget format is primarily used when creating real LLM-based agents
# (ReactiveAgent, PlanningAgent, AgileThinker) from the command line or programmatically

print("=" * 60)
print("RealtimeGym - Budget Format Guide")
print("=" * 60)

print("\n## Budget Format Options\n")

print("1. TOKEN BUDGET (default)")
print("-" * 60)
print("   - Budget measured in tokens")
print("   - Used for LLM-based agents to control token usage")
print("   - Example: --budget_format token --time_pressure 8192")
print("   - The agent can generate up to 8192 tokens")
print()

print("2. TIME BUDGET")
print("-" * 60)
print("   - Budget measured in seconds")
print("   - Used for time-constrained environments")
print("   - Example: --budget_format time --time_pressure 10")
print("   - The agent has 10 seconds to think")
print()

print("\n## Command Line Usage\n")
print("-" * 60)

print("\n# Token-based budget (default)")
print("realtime-gym --budget_format token \\")
print("             --time_pressure 8192 \\")
print("             --internal_budget 4096 \\")
print("             --game freeway \\")
print("             --system agile")

print("\n# Time-based budget")
print("realtime-gym --budget_format time \\")
print("             --time_pressure 5.0 \\")
print("             --internal_budget 2.0 \\")
print("             --game snake \\")
print("             --system reactive")

print("\n## Programmatic Usage (Custom Agents)\n")
print("-" * 60)

# Example: Token-based budget
print("\n### Token Budget Example:")
print("""
from realtimegym.agents.reactive import ReactiveAgent
from realtimegym.environments.prompts import freeway

agent = ReactiveAgent(
    prompts=freeway,
    file='log.csv',
    budget_form='token',      # Token-based budget
    port1='https://api.deepseek.com',
    port2='https://api.deepseek.com',
    api_key='your-api-key',
    internal_budget=4096,     # Max tokens for thinking
    model1='deepseek-chat',
    model2='deepseek-reasoner',
    skip_action=True
)

# Agent will use token budget
agent.think(timeout=8192)  # timeout = token limit
""")

# Example: Time-based budget
print("\n### Time Budget Example:")
print("""
from realtimegym.agents.planning import PlanningAgent
from realtimegym.environments.prompts import snake

agent = PlanningAgent(
    prompts=snake,
    file='log.csv',
    budget_form='time',       # Time-based budget
    port1='https://api.deepseek.com',
    port2='https://api.deepseek.com',
    api_key='your-api-key',
    internal_budget=2.0,      # Initial thinking time in seconds
    model1=None,
    model2='deepseek-reasoner',
    skip_action=True
)

# Agent will use time budget
agent.think(timeout=5.0)  # timeout = seconds
""")

print("\n## Budget Parameters\n")
print("-" * 60)
print("""
Key parameters when using budget formats:

1. budget_format / budget_form
   - Controls how budget is measured
   - Choices: "token" or "time"

2. time_pressure
   - Total budget available per turn
   - For token: number of tokens (e.g., 8192)
   - For time: seconds (e.g., 5.0)

3. internal_budget
   - Budget allocated for internal thinking
   - Must be <= time_pressure
   - For token: tokens allocated to thinking
   - For time: seconds allocated to thinking
""")

print("\n## Budget Format Behavior\n")
print("-" * 60)

print("\nToken Budget:")
print("  - Agent generates tokens until limit is reached")
print("  - Tracks token usage from LLM API")
print("  - More deterministic for LLM-based agents")

print("\nTime Budget:")
print("  - Agent thinks for specified duration")
print("  - Uses wall-clock time measurement")
print("  - Better for real-time constrained scenarios")

print("\n## Example: Custom Agent with Mock Budget\n")
print("-" * 60)


class BudgetAwareAgent:
    """Custom agent that respects budget constraints."""

    def __init__(self, budget_format="token"):
        self.budget_format = budget_format
        self.current_observation = None
        self.action = "S"
        self.budget_used = 0

    def observe(self, observation):
        """Receive observation."""
        self.current_observation = observation

    def think(self, timeout=None):
        """Think within budget constraints."""
        if self.current_observation is None:
            return

        print(f"\n  Thinking with {self.budget_format} budget...")
        print(f"  Budget available: {timeout}")

        if self.budget_format == "token":
            # Simulate token usage
            tokens_used = min(100, timeout) if timeout else 100
            self.budget_used = tokens_used
            print(f"  Tokens used: {tokens_used}")
        else:  # time
            # Simulate time-based thinking
            think_time = min(0.1, timeout) if timeout else 0.1
            start = time.time()
            time.sleep(think_time)
            elapsed = time.time() - start
            self.budget_used = elapsed
            print(f"  Time used: {elapsed:.3f} seconds")

        # Make decision
        turn = self.current_observation.get("game_turn", 0)
        self.action = "U" if turn % 2 == 0 else "S"

    def act(self):
        """Return action."""
        return self.action


# Demonstrate token budget
print("\nToken Budget Demo:")

env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
agent = BudgetAwareAgent(budget_format="token")

obs, done = env.reset()
agent.observe(obs)
agent.think(timeout=8192)  # Token budget
action = agent.act()
print(f"  Action chosen: {action}")

# Demonstrate time budget
print("\nTime Budget Demo:")
agent = BudgetAwareAgent(budget_format="time")
obs, done = env.reset()
agent.observe(obs)
agent.think(timeout=1.0)  # Time budget in seconds
action = agent.act()
print(f"  Action chosen: {action}")

print("\n" + "=" * 60)
print("Budget Format Guide Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("  1. Use 'token' budget for LLM-based agents")
print("  2. Use 'time' budget for real-time constraints")
print("  3. Set via --budget_format on command line")
print("  4. Set via budget_form parameter when creating agents")
print("  5. internal_budget must be <= time_pressure")
print("=" * 60)
