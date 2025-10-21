# ⚡ Real-Time Reasoning Gym

**A gym for evaluating language agents in real-time environments with time constraints.**

RealtimeGym provides a simple, unified interface for testing how AI agents perform when they have limited time to think and act. Perfect for evaluating LLMs in dynamic, time-pressured scenarios.

## Quick Start

```python
import realtimegym

# Create environment
env, seed, renderer = realtimegym.make('Freeway-v0', seed=0, render=False)

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
total_reward = 0
while not done:
    agent.observe(obs)
    agent.think(timeout=8192)
    action = agent.act() or "S"
    obs, done, reward, reset = env.step(action) 
    total_reward = reward # Pay attention that reward here is not accumulative

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

## Using LLM-Based Agents

RealtimeGym includes following built-in LLM agents:

| Agent | Description | Use Case | Supported LLM |
|-------|-------------|----------|--------------|
| **ReactiveAgent** | Fast, reactive responses | Bounded Latency | All OpenAI-compatible |
| **PlanningAgent** | Strategic planning | Unbounded Latency | All OpenAI-compatible |
| **AgileThinker** | Hybrid approach | Combination of Above | Models with transparent thinking tokens


You can evaluate them via cli-command

### Command Line

```bash
agile_eval --budget_format token \
    --time_pressure 8192 \
    --internal_budget 4096 \
    --game freeway \
    --cognitive_load E \
    --mode agile \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
     --seed_num 1 --repeat_times 1 \
    --log_dir logs-debug2
```

Or more compactly:

```bash
agile_eval --budget_format token \
    --settings freeway_H_8192_agile_4096 \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
    --log_dir logs-debug2 --seed_num 1 --repeat_times 1
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

### Time Budget
```python
# Agent has 5 seconds to think
agent.think(timeout=5.0)
```
- Best for: Real-time scenarios
- Measures: Wall-clock seconds

Set via `--budget_format token` or `--budget_format time` on command line.


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
obs, done, reward, reset = env.step(action)
# Returns: (observation dict, done flag, reward, reset_flag)
```

### Observation Structure

```python
{
    'state_string': 'Text representation of game state',
    'game_turn': 42,  # Current turn number
    'model1_description': 'Detailed state info (game-specific) for reactive agent',
    'model2_description': 'Detailed state info (game-specific) for planning agent',
    # ... other game-specific fields
}
```

## Testing

Run the test suite:

```bash
pytest
```

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
## Project Structure

```
realtimegym/
├── src/realtimegym/
│   ├── __init__.py          # Main API (make function)
│   ├── agile_eval.py       # Agent evaluation script
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
└── pyproject.toml          # Package configuration
```

## Documentation

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
