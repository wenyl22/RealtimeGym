### Real-time Reasoning Gym

## Installation

```bash
pip install -e .
```

For development (includes linting, type checking, and pre-commit hooks):
```bash
pip install -e ".[dev]"
```

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. To set up:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files (optional)
pre-commit run --all-files
```

The pre-commit hooks will automatically run:
- **ruff** - Python linting and formatting
- **ty** - Python type checking
- Standard checks (trailing whitespace, YAML validation, etc.)

### Running Checks Manually

```bash
# Run ruff linter
uv run ruff check

# Run ruff formatter
uv run ruff format

# Run type checker
uv run ty check

# Run all checks together
uv run ruff check && uv run ruff format && uv run ty check
```

## Example Usage:

If `budget_format == time`, then  `time_pressure` and `internal_budget` are physical times (unit: second).
If `budget_format == token`, then `time_pressure` and `internal_budget` are token numbers.

```bash
python run_game.py --api_key DEEPSEEK_API_KEY \
    --port https://api.deepseek.com/v3.1_terminus_expires_on_20251015 \
    --model2 deepseek-reasoner --model1 deepseek-chat \
    --budget_format [token/time] \
    --game [freeway/snake/overcooked] --cognitive_load [E/M/H] --time_pressure 8192 \
    --system [planning/reactive/agile] --internal_budget 4096 \
    --log_dir logs --seed_num [1-8]
```
