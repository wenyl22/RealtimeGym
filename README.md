# ⚡ Real-Time Reasoning Gym

**A gym for evaluating language agents in real-time environments with time constraints.**

RealtimeGym provides a simple, unified interface for testing how AI agents perform when they have limited time to think and act. Perfect for evaluating LLMs in dynamic, time-pressured scenarios.

## Quick Start

```python
import realtimegym

# Create environment
env, seed, _ = realtimegym.make('Freeway-v0', seed=0, render=False)

# Simple agent
class SimpleAgent:
    def __init__(self):
        self.observation = None

    def observe(self, obs):
        self.observation = obs

    def think(self, timeout=None):
        pass  # Your thinking logic here

    def act(self):
        return "U"  # Move up

# Run game loop
agent = SimpleAgent()
obs, done = env.reset()

while not done:
    agent.observe(obs)
    agent.think(timeout=8192)
    action = agent.act() or "S"
    obs, done, reward = env.step(action)

print(f"Final reward: {reward}")
```

## Installation

```bash
pip install -e .
```

For development (includes testing, linting, type checking):
```bash
pip install -e ".[dev]"
```

## Environments

RealtimeGym includes three classic games with real-time constraints:

| Environment | Description | Actions | Difficulty Levels |
|------------|-------------|---------|-------------------|
| **Freeway** | Cross the road avoiding cars | `U` (up), `D` (down), `S` (stay) | v0 (Easy), v1 (Medium), v2 (Hard) |
| **Snake** | Classic snake game | `U`, `D`, `L` (left), `R` (right), `S` | v0, v1, v2 |
| **Overcooked** | Cooperative cooking | `U`, `D`, `L`, `R`, `I` (interact), `S` | v0, v1, v2 |

```python
# Create any environment with difficulty level
env, seed, renderer = realtimegym.make('Snake-v2', seed=0, render=False)
```

## The Agent Interface

Every agent must implement three methods:

```python
class MyAgent:
    def observe(self, observation: dict):
        """
        Receive observation from environment.

        observation contains:
        - 'state_string': Text description of game state
        - 'game_turn': Current turn number
        - 'description': Detailed state info (varies by game)
        """
        pass

    def think(self, timeout: int | float):
        """
        Process observation and decide action.

        timeout is either:
        - Number of tokens (if budget_format='token')
        - Seconds (if budget_format='time')
        """
        pass

    def act(self) -> str | None:
        """
        Return chosen action or None for default.

        Returns action like 'U', 'D', 'L', 'R', 'S', 'I'
        """
        pass
```

## Complete Example

Here's a stateful agent that tracks observation history:

```python
import realtimegym

class StatefulAgent:
    def __init__(self):
        self.observations = []
        self.action = "S"

    def observe(self, observation):
        self.observations.append(observation)

    def think(self, timeout=None):
        if not self.observations:
            return

        # Strategy based on history
        turn = len(self.observations)
        if turn % 2 == 0:
            self.action = "U"
        else:
            self.action = "S"

    def act(self):
        return self.action

# Run the agent
env, _, _ = realtimegym.make('Freeway-v0', seed=42, render=False)
agent = StatefulAgent()

obs, done = env.reset()
total_reward = 0

while not done:
    agent.observe(obs)
    agent.think(timeout=8192)
    action = agent.act() or "S"
    obs, done, reward = env.step(action)
    total_reward += reward

print(f"Game finished! Total reward: {total_reward}")
```

## Using LLM-Based Agents

RealtimeGym includes built-in agents for LLM evaluation:

### Command Line

```bash
# Token-based budget (control by token count)
python run_game.py \
    --api_key YOUR_API_KEY \
    --budget_format token \
    --time_pressure 8192 \
    --internal_budget 4096 \
    --game freeway \
    --cognitive_load E \
    --system agile \
    --model1 deepseek-chat \
    --model2 deepseek-reasoner

# Time-based budget (control by seconds)
python run_game.py \
    --api_key YOUR_API_KEY \
    --budget_format time \
    --time_pressure 5.0 \
    --internal_budget 2.0 \
    --game snake \
    --cognitive_load M \
    --system reactive
```

### Programmatically

```python
from realtimegym.agents.reactive import ReactiveAgent
from realtimegym.environments.prompts import freeway

agent = ReactiveAgent(
    prompts=freeway,
    file='game_log.csv',
    budget_form='token',  # or 'time'
    port1='https://api.deepseek.com',
    api_key='your-api-key',
    internal_budget=4096,
    model1='deepseek-chat',
    model2='deepseek-reasoner',
    skip_action=True
)
```

## Budget Formats Explained

RealtimeGym supports two ways to constrain agent thinking:

### Token Budget (default)
```python
# Agent can use up to 8192 tokens for thinking
agent.think(timeout=8192)
```
- Best for: LLM-based agents
- Measures: Token count from API
- Use case: Controlling computational cost

### Time Budget
```python
# Agent has 5 seconds to think
agent.think(timeout=5.0)
```
- Best for: Real-time scenarios
- Measures: Wall-clock seconds
- Use case: Time-critical decision making

Set via `--budget_format token` or `--budget_format time` on command line.

## Available Agent Types

| Agent | Description | Use Case |
|-------|-------------|----------|
| **ReactiveAgent** | Fast, reactive responses | Low latency scenarios |
| **PlanningAgent** | Strategic planning | Complex decision making |
| **AgileThinker** | Hybrid approach | Balance of speed and planning |

## Examples

Check out the `examples/` directory:

```bash
# Getting started
python examples/basic_usage.py

# Try all environments
python examples/all_environments.py

# Advanced agent patterns
python examples/custom_agent.py

# Compare difficulty levels
python examples/difficulty_levels.py

# Budget format guide
python examples/budget_format.py
```

## API Reference

### Environment Methods

```python
# Create environment
env, seed, renderer = realtimegym.make(env_id, seed=0, render=False)

# Reset environment
obs, done = env.reset()
# Returns: (observation dict, done flag)

# Take action
obs, done, reward = env.step(action)
# Returns: (observation dict, done flag, reward)
```

### Observation Structure

```python
{
    'state_string': 'Text representation of game state',
    'game_turn': 42,  # Current turn number
    'description': 'Detailed state info (game-specific)',
    # ... other game-specific fields
}
```

## Testing

Run the test suite:

```bash
# All tests (skip Overcooked due to NumPy 2.0 issue)
pytest -k "not Overcooked"

# With coverage
pytest --cov=src/realtimegym --cov-report=html

# Specific test file
pytest tests/test_agents.py
```

**Test Coverage**: 37/46 tests passing (80%) - all core functionality validated.

See [`tests/README.md`](tests/README.md) for details.

## Development

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Quality Tools

```bash
# Linting
ruff check

# Formatting
ruff format

# Type checking
ty check
```

### CI/CD

GitHub Actions automatically runs on PRs:
- ✅ Type checking with `ty`
- ✅ Linting with `ruff`
- ✅ Test suite with `pytest`

See [`.github/workflows/pr-checks.yml`](.github/workflows/pr-checks.yml)

## Project Structure

```
realtimegym/
├── src/realtimegym/
│   ├── __init__.py          # Main API (make function)
│   ├── agents/              # Built-in LLM agents
│   │   ├── base.py
│   │   ├── reactive.py
│   │   ├── planning.py
│   │   └── agile.py
│   └── environments/        # Game environments
│       ├── base.py
│       ├── freeway.py
│       ├── snake.py
│       └── overcooked.py
├── tests/                   # Test suite
├── examples/                # Example scripts
├── run_game.py             # CLI entry point
└── pyproject.toml          # Package configuration
```

## Key Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `env_id` | Environment identifier | `Freeway-v0`, `Snake-v1`, etc. | - |
| `seed` | Random seed | 0-7 | 0 |
| `render` | Enable visualization | `True`/`False` | `False` |
| `budget_format` | Budget measurement | `token`/`time` | `token` |
| `time_pressure` | Total budget per turn | tokens or seconds | 8192 |
| `internal_budget` | Thinking budget | tokens or seconds | 4096 |
| `cognitive_load` | Difficulty level | `E`/`M`/`H` | `E` |
| `system` | Agent type | `reactive`/`planning`/`agile` | - |

## Known Issues

**Overcooked NumPy 2.0**: The Overcooked environment uses vendored third-party code with a NumPy 2.0 incompatibility. To work around:
- Skip Overcooked tests: `pytest -k "not Overcooked"`
- Or use NumPy <2.0: `pip install "numpy<2.0"`

See [`THIRD_PARTY_NOTICE.md`](src/realtimegym/environments/overcooked_new/THIRD_PARTY_NOTICE.md) for details.

## Documentation

- **[TESTING.md](TESTING.md)** - Testing infrastructure overview
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[tests/README.md](tests/README.md)** - Detailed test documentation
- **[examples/README.md](examples/README.md)** - Examples guide

## Citation

If you use RealtimeGym in your research, please cite:

```bibtex
@software{realtimegym2025,
  title={RealtimeGym: A Real-time Gym for Evaluating Language Agents},
  author={wenyl},
  year={2025},
  url={https://github.com/wenyl22/RealtimeGym}
}
```

## Links

- **Homepage**: https://github.com/wenyl22/RealtimeGym
- **Documentation**: https://bleaves.github.io/real-time-reasoning/
- **Issues**: https://github.com/wenyl22/RealtimeGym/issues

## License

MIT License - see LICENSE file for details.

---

**Quick Links**: [Installation](#installation) • [Examples](#examples) • [API Reference](#api-reference) • [Testing](#testing) • [Contributing](CONTRIBUTING.md)
