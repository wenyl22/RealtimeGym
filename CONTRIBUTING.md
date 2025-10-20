# Contributing to RealtimeGym

Thank you for your interest in contributing to RealtimeGym!

## Development Setup

### 1. Install Development Dependencies

```bash
# Clone the repository
git clone https://github.com/wenyl22/RealtimeGym.git
cd RealtimeGym

# Install with development dependencies
pip install -e ".[dev]"
```

### 2. Set Up Pre-commit Hooks

We use pre-commit hooks to maintain code quality:

```bash
# Install the pre-commit hooks
pre-commit install

# (Optional) Run on all files to test
pre-commit run --all-files
```

## Code Quality Standards

### Automated Checks

All code must pass the following checks before being merged:

1. **ruff** - Python linting and formatting
2. **ty** - Python type checking
3. **pre-commit** - Additional code quality checks

### Running Checks Locally

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

### Pre-commit Hooks

The pre-commit hooks will automatically run when you commit code. They include:

- **ruff** - Lints and formats Python code
- **ty** - Type checks Python code
- **trailing-whitespace** - Removes trailing whitespace
- **end-of-file-fixer** - Ensures files end with a newline
- **check-yaml** - Validates YAML files
- **check-toml** - Validates TOML files
- **check-merge-conflict** - Checks for merge conflict markers
- **check-added-large-files** - Prevents accidentally committing large files

### CI/CD

All pull requests automatically run the following checks via GitHub Actions:

- Ruff linting
- Ruff formatting check
- ty type checking
- Tests (when available)

You can see the workflow configuration in `.github/workflows/pr-checks.yml`.

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure all checks pass locally
5. Commit your changes (pre-commit hooks will run automatically)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Pull Request Guidelines

- Ensure your code passes all automated checks
- Add tests for new features (when test infrastructure is available)
- Update documentation as needed
- Follow the existing code style
- Write clear commit messages

## Code Style

We use **ruff** for both linting and formatting. The configuration is in `pyproject.toml`:

```toml
[tool.ruff]
exclude = [
    "src/realtimegym/environments/overcooked_new",  # Third-party code
]
```

## Type Checking

We use **ty** for type checking. The configuration is in `pyproject.toml`:

```toml
[tool.ty.src]
exclude = [
    "src/realtimegym/environments/overcooked_new/**",  # Third-party code
]
```

## Third-Party Code

The `src/realtimegym/environments/overcooked_new/` directory contains vendored code from [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai). This code is excluded from our linting and type checking standards. See `src/realtimegym/environments/overcooked_new/THIRD_PARTY_NOTICE.md` for details.

## Questions?

If you have questions about contributing, please open an issue on GitHub.
