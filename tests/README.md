# RealtimeGym Test Suite

Comprehensive test suite for the RealtimeGym package.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=src/realtimegym --cov-report=html
```

### Run specific test files
```bash
pytest tests/test_environments.py
pytest tests/test_agents.py
```

### Run tests for specific environments (excluding Overcooked)
```bash
pytest tests/ -k "not Overcooked"
```

## Test Organization

### `test_environments.py`
Tests for all three game environments (Freeway, Snake, Overcooked):

- **TestEnvironmentRegistry**: Environment creation and the `make()` function
- **TestEnvironmentAPI**: Standard gym-like API (`reset()`, `step()`)
- **TestFreewayEnvironment**: Freeway-specific functionality
- **TestSnakeEnvironment**: Snake-specific functionality
- **TestOvercookedEnvironment**: Overcooked-specific functionality
- **TestSeeding**: Reproducibility and seeding
- **TestBackwardCompatibility**: Legacy `act()` method compatibility

### `test_agents.py`
Tests for the agent API and interface:

- **TestAgentAPI**: Core agent pattern (observe → think → act)
- **TestAgentIntegration**: Integration between agents and environments
- **TestRealAgents**: Validation of BaseAgent class interface

##Test Coverage

**Current Status**: ✅ 37 out of 46 tests passing (80%)

### Passing Tests (37)
- ✅ All Freeway environment tests
- ✅ All Snake environment tests
- ✅ All agent API tests (with Freeway/Snake)
- ✅ All base class interface tests
- ✅ All seeding and reproducibility tests
- ✅ All backward compatibility tests

### Known Issues (9)
- ⚠️ **Overcooked tests** (3 failed, 6 errors): The Overcooked environment uses third-party vendored code that has a NumPy 2.0 incompatibility (`np.Inf` → `np.inf`). This affects all Overcooked-related tests.

## Known Limitations

### Overcooked NumPy 2.0 Incompatibility

The Overcooked environment is based on third-party code from [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) (see `THIRD_PARTY_NOTICE.md`). This code uses the deprecated `np.Inf` constant which was removed in NumPy 2.0.

**Workaround Options:**
1. Downgrade NumPy to <2.0: `pip install "numpy<2.0"`
2. Skip Overcooked tests: `pytest -k "not Overcooked"`
3. Fix vendored code: Replace `np.Inf` with `np.inf` in `src/realtimegym/environments/overcooked_new/Overcooked_Env.py:353`

The third-party code is excluded from type checking and linting per `pyproject.toml`.

## Test Configuration

Pytest configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=src/realtimegym",
    "--cov-report=term-missing",
    "--cov-report=html",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

## Writing New Tests

### Example Test Structure

```python
import pytest
import realtimegym

def test_my_feature():
    """Test description."""
    env, seed, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
    obs, done = env.reset()

    # Your test logic here
    assert isinstance(obs, dict)
    assert not done
```

### Testing Custom Agents

```python
class TestMyAgent:
    def test_agent_interface(self):
        """Test custom agent implements required interface."""
        agent = MyCustomAgent()

        assert hasattr(agent, 'observe')
        assert hasattr(agent, 'think')
        assert hasattr(agent, 'act')
```

## Continuous Integration

Tests are automatically run on pull requests via GitHub Actions. See `.github/workflows/pr-checks.yml`.

The CI runs:
1. Type checking with `ty`
2. Linting with `ruff`
3. Test suite with `pytest`

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add integration tests for new environments or agents
4. Update this README if adding new test categories

For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md).
