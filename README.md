
# ⚡🧠🏋️ Realtime Reasoning Gym

**Realtime Reasoning Gym** is a specialized evaluation framework for testing how well language agents can reason and make decisions under real-time constraints. Unlike traditional OpenAI Gym environments where agents have unlimited thinking time, Realtime Reasoning Gym enforces strict **time budgets** (measured in seconds) or **token budgets** (measured in LLM decoding tokens) to simulate real-world pressure.

Furthermore, each environment in the gym offers multiple **cognitive load levels** that vary the intellectual complexity of tasks, enabling a comprehensive assessment of whether agents can balance decision quality and speed when facing different levels of cognitive load and time pressure.


<figure>
    <img src="assets/Figure1.png" alt="Performance under time constraints">
    <figcaption>Upper: We create three real-time games, <em>Freeway</em>, <em>Snake</em>, and <em>Overcooked</em>, to study the challenge of real-time reasoning. Lower: Under <em>cognitive load</em> and <em>time pressure</em>, <strong>AgileThinker</strong> (Ours), which engages both <em>reactive</em> and <em>planning</em> reasoning paradigms, consistently outperforms agents that engage only one of them. Scores are averaged across different games.</figcaption>
</figure>

You can see our post, paper, [website](https://bleaves.github.io/real-time-reasoning/) for more detailed demonstrations and explanations.

- [Quick Start](#quick-start)
- [Real-Time Reasoning Challenge](#the-real-time-reasoning-challenge)
    - [Cognitive Load Control](#cognitive-load-control)
    - [Time Pressure Control](#time-pressure-control)
- [Built-in LLM Agents](#built-in-llm-agents)
- [Add a New Environment](#add-a-new-environment)
- [API Reference](#api-reference)
- [More Examples](#examples)

## Quick Start
1. Install Realtime Reasoning Gym:
    ```bash
    git clone https://github.com/wenyl22/RealtimeGym.git
    cd RealtimeGym

    # Install in development mode
    pip install -e .

    # Install development dependencies
    pip install -e ".[dev]"
    ```
2. Set up API keys in `.env`, a template is provided in `.env.example`.


## The Real-Time Reasoning Challenge

In typical OpenAI Gym environments, agents have unlimited time to think:

```python
# Traditional approach - unbounded thinking time
obs, done = env.reset()
while not done:
    action = agent.act(obs)  # Can take minutes!
    obs, reward, done, info = env.step(action)
```

RealtimeGym introduces **explicit constraints** on the agent's thinking time:

```python
# Real-time approach - bounded thinking time
obs, done = env.reset()
while not done:
    agent.observe(obs)        # Fast observation
    agent.think(timeout=8192) # Bounded thinking (tokens or seconds)
    action = agent.act() or DEFAULT_ACTION # Default if a decision isn't made in time
    obs, done, reward, reset = env.step(action)
```

Real-time reasoning requires agents to balance **correctness** and **timeliness**.

### Cognitive Load Control

Realtime Reasoning Gym includes three real-time games with increasing cognitive loads:

| Game | Description | Actions | Cognitive Load Levels |
|------|-------------|---------|-------------------|
| **Freeway** | Cross busy roads avoiding cars | `U` (up), `D` (down), `S` (stay) | v0 (Easy), v1 (Medium), v2 (Hard) |
| **Snake** | Classic snake game with growing body | `U`, `D`, `L` (left), `R` (right), `S` | v0, v1, v2 |
| **Overcooked** | Cooperative cooking simulation | `U`, `D`, `L`, `R`, `I` (interact), `S` | v0, v1, v2 |

Create any environment:
```python
env, real_seed, renderer = realtimegym.make('Freeway-v2', seed=0, render=False)
# Note:
# seed != real_seed, because real_seed embeds the cognitive load level.
# renderer = None if render=False, otherwise call renderer.render(env) to visualize the game state.
# See examples/basic_renderer.py for details.
```

### Time Pressure Control

Realtime Reasoning Gym supports two time constraint types:

#### Token Budget
```python
agent.think(timeout=8192)  # Environment evolves after 8192 decoding steps
```
- **Best for:** LLM-based agents
- **Measures:** Token count from API calls
- **Advantage:** Platform-independent, reproducible

#### Time Budget
```python
agent.think(timeout=5.0)  # Environment evolves after 5 seconds
```
- **Best for:** Real-time deployment scenarios
- **Measures:** Wall-clock time
- **Advantage:** Direct real-world applicability

Set via `--time_unit token` or `--time_unit seconds`.


## Built-in LLM Agents

Real-time Reasoning Gym's built-in agents all implement the `BaseAgent` interface:

```python
from .base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(
        self,
        prompts: Any,  # prompt: dynamically loaded module, mapping game state to text
        file: str, # log file path
        time_unit: str, # 'token' or 'seconds'
        **kwargs,
    ) -> None:
        super().__init__(prompts, file, time_unit)
        # Initialize internal state here

    def think(self, timeout: int): -> None:
        """Process information and plan action within budget.

        Args:
            timeout: Token count (time_unit='token') or seconds (time_unit='seconds')
        """
        # Decision making within timeout budget
        # store chosen action here
        self.action = ...
```

RealtimeGym provides three ready-to-use agent types:

| Agent | Strategy | Best For | Supported Models |
|-------|----------|----------|------------------|
| **ReactiveAgent** | Generate responses in bounded time | Low cognitive load, high time pressure scenarios | All OpenAI-compatible |
| **PlanningAgent** | Comprehensive planning without time constraints | High cognitive load, low time pressure scenarios | All OpenAI-compatible |
| **AgileThinker** | Hybrid reactive + planning | Balanced performance | Models with explicit reasoning tokens |

### Evaluation

Evaluate agents using the command-line interface:

```bash
# Detailed configuration
agile_eval --time_unit token \
    --time_pressure 8192 \
    --internal_budget 4096 \
    --game freeway \
    --cognitive_load E \
    --mode agile \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
    --seed_num 1 --repeat_times 1

# Using more compact configurations
agile_eval --time_unit token \
    --settings freeway_H_8192_agile_4096 \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
    --seed_num 8 --repeat_times 1
```


## Add a New Environment

To create a custom environment:

1. Create a new file in `src/realtimegym/environments/` inheriting from `BaseEnvironment`:
   ```python
   from .base import BaseEnvironment

   class MyGameEnv(BaseEnvironment):
        def __init__(self):
            super().__init__()
        def set_seed(self, seed: int):
            # Set game random seed
            # In our implementation, cognitive load level is embedded in seed
            pass
        def reset(self):
            # Initialize game state
            return observation, done
        def step(self, action):
            # Process action and update state
            return observation, done, reward, reset
        def state_string(self):
            # Return human-readable state in text
            return state_string
        def state_builder(self):
            # Return detailed state dict
            return state_dict
   ```

2. Register your environment in `src/realtimegym/__init__.py`:

3. (Optional) To evaluate built-in LLM agents, create prompts in `src/realtimegym/prompts/mygame.py` following existing patterns. Specifically, you need to implement a function:
    ```python
    def state_to_description(observation: dict, mode: str) -> str | dict:
        ## Return different descriptions based on agent mode
        if mode == "reactive":
            return text_description_for_reactive_agent
        elif mode == "planning":
            return text_description_for_planning_agent
        elif mode == "agile":
            return {
                "planning": text_description_for_planning_agent,
                "reactive": text_description_for_reactive_agent
            }
    ```

## API Reference

### Environment API

```python
# Environment creation
env, seed, renderer = realtimegym.make(env_id, seed=0, render=False)

# Environment interaction
obs, done = env.reset()
obs, done, reward, reset = env.step(action)
```

### Environment Observation Structure

```python
{
    'state_string': str,    # Human-readable game state
    'game_turn': int,       # Current turn number
    'state': dict          # Detailed game-specific state
}
```

### Agent Configuration

Agents accept YAML configuration files specifying model parameters:
```yaml
model: "gpt-4o-mini"
api_key: "your-api-key"
inference_parameters:
    temperature: 0.7
    max_tokens: 1000
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific test categories
pytest tests/test_environments.py
pytest tests/test_agents.py
```

## Examples

Explore the `examples/` directory:

```bash
# Basic usage patterns
python examples/basic_usage.py

# Basic usage with rendering
python examples/basic_renderer.py

# Compare all environments
python examples/all_environments.py

# Custom agent implementation
python examples/custom_agent.py

# Cognitive load level analysis
python examples/difficulty_levels.py
```


## Citation

If you use Realtime Reasoning Gym in your research, please cite our work:

```bibtex
@software{realtimegym2025,
  title={Realtime Reasoning Gym: A Real-Time Learning Environment for Language Agents},
  author={Yule Wen, Yixin Ye, Yanzhe Zhang, Diyi Yang and Hao Zhu},
  year={2025},
  url={https://github.com/wenyl22/RealtimeGym},
  note={A framework for evaluating language agents under real-time constraints}
}
```

## Acknowledgements

RealtimeGym builds upon the excellent open-source project [Overcooked](https://github.com/HumanCompatibleAI/human_aware_rl). We thank the original authors for their contributions.

## Links

- **🏠 Homepage**: https://github.com/wenyl22/RealtimeGym
- **📖 Documentation**: https://bleaves.github.io/real-time-reasoning/
- **🐛 Issues**: https://github.com/wenyl22/RealtimeGym/issues
- **💬 Discussions**: https://github.com/wenyl22/RealtimeGym/discussions

---

**RealtimeGym** - *Advancing real-time reasoning capabilities in language agents* ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!--
## Task Suite

`realtimegym` includes three games with real-time constraints:

| Game | Description | Actions | Difficulty Levels |
|------------|-------------|---------|-------------------|
| **Freeway** | Cross the road avoiding cars | `U` (up), `D` (down), `S` (stay) | v0 (Easy), v1 (Medium), v2 (Hard) |
| **Snake** | Greedy snake game | `U`, `D`, `L` (left), `R` (right), `S` | v0, v1, v2 |
| **Overcooked** | Cooperative cooking | `U`, `D`, `L`, `R`, `I` (interact), `S` | v0, v1, v2 |

To create an task in our provided suite, use the `realtimegym.make()` function:
```python
# Create any environment with difficulty level
env, seed, renderer = realtimegym.make('Snake-v2', seed=0, render=False)
```

## The challenge of Real-Time Reasoning

Real-time reasoning requires agents to not only produce correct actions, but also make the timely decision.
The following figure shows the benchmark results of different LLM-based agents under various time budgets in the
three tasks that we provide.

![Fig. 1](assets/Figure1.png)



## The Agent Interface

Every agent must implement three methods:

```python
from realtimegym.agents.base import BaseAgent

class MyAgent(BaseAgent):
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
        - Number of tokens (if time_unit='token')
        - Seconds (if time_unit='seconds')
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


You can evaluate these agents and large language models via cli-command `agile_eval`:

```bash
agile_eval --time_unit token \
    --time_pressure 8192 \
    --internal_budget 4096 \
    --game freeway \
    --cognitive_load E \
    --mode agile \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
     --seed_num 1 --repeat_times 1
```

Or more compactly:

```bash
agile_eval --time_unit token \
    --settings freeway_H_8192_agile_4096 \
    --reactive-model-config configs/deepseek-v3.2-reactive.yaml \
    --planning-model-config configs/deepseek-v3.2-planning.yaml \
    --seed_num 1 --repeat_times 1
```

### Customizing Prompts

You can customize the prompts by setting `--prompt-config` to a YAML file that maps game names to prompt modules. Each prompt module should contain:
- A `state_to_description()` function that takes the observation and agent type (`'reactive'`, `'planning'`, or `'agile'`) and returns descriptions of the game state
- Constants: `ALL_ACTIONS` and `DEFAULT_ACTION`

**Built-in prompt structure:**
- `configs/example-prompts.yaml` - Maps game names to prompt modules
- `src/realtimegym/prompts/` - Contains both Python modules and YAML template files
  - `*.py` files contain the state conversion logic
  - `*.yaml` files contain the prompt template strings

The YAML templates are bundled with the package and will be included when installed via pip.

**To create custom prompts:**
1. Create a new Python module following the pattern in `src/realtimegym/prompts/`
2. Optionally create a YAML file with your prompt templates
3. Reference your module in a custom prompt config YAML file

## Budget Formats Explained

RealtimeGym supports two ways to constrain agent thinking:

### Token Budget (default)
```python
# Agent can use up to 8192 tokens for thinking
agent.think(timeout=8192)
```
- Best for: LLM-based agents
- Measures: Token count from API

### Physical Time Budget
```python
# Agent has 5 seconds to think
agent.think(timeout=5.0)
```
- Best for: Real-time scenarios
- Measures: Wall-clock seconds

Set via `--time_unit token` or `--time_unit seconds` on command line.


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
    "state": { ... }  # Game-specific detailed state info in dict
}
```

## Testing

Run the test suite:

```bash
pytest
```

## Development

Make sure to use `pre-commit`:

### Pre-commit Hooks

```bash
# Install hooks
uv run pre-commit install
```

### Static Typing

```bash
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

**Quick Links**: [Installation](#installation) • [Examples](#examples) • [API Reference](#api-reference) • [Testing](#testing) • [Contributing](CONTRIBUTING.md) -->
