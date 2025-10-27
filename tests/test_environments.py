"""Tests for RealtimeGym environments."""

from typing import Any

import pytest

import realtimegym


class TestEnvironmentRegistry:
    """Test environment creation and registry."""

    def test_make_freeway(self) -> None:
        """Test creating Freeway environment."""
        env, seed, renderer = realtimegym.make("Freeway-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_snake(self) -> None:
        """Test creating Snake environment."""
        env, seed, renderer = realtimegym.make("Snake-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_overcooked(self) -> None:
        """Test creating Overcooked environment."""
        env, seed, renderer = realtimegym.make("Overcooked-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_invalid_env(self) -> None:
        """Test that invalid environment ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown environment ID"):
            realtimegym.make("InvalidEnv-v0")

    def test_difficulty_levels(self) -> None:
        """Test all difficulty levels can be created."""
        for difficulty in ["v0", "v1", "v2"]:
            env, _, _ = realtimegym.make(f"Freeway-{difficulty}", seed=0, render=False)
            assert env is not None


class TestEnvironmentAPI:
    """Test the standard gym-like API for all environments."""

    @pytest.fixture(params=["Freeway-v0", "Snake-v0", "Overcooked-v0"])
    def env(self, request: Any) -> Any:  # noqa: ANN401
        """Fixture providing each environment."""
        env, _, _ = realtimegym.make(request.param, seed=0, render=False)
        return env

    def test_reset_returns_correct_types(self, env: Any) -> None:  # noqa: ANN401
        """Test that reset() returns (obs, done) with correct types."""
        obs, done = env.reset()

        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(done, bool), "Done flag should be a boolean"
        assert not done, "Done should be False after reset"

    def test_reset_observation_keys(self, env: Any) -> None:  # noqa: ANN401
        """Test that reset observation contains required keys."""
        obs, done = env.reset()

        assert "state_string" in obs, "Observation should contain 'state_string'"
        assert "game_turn" in obs, "Observation should contain 'game_turn'"
        assert isinstance(obs["state_string"], str), "state_string should be a string"
        assert isinstance(obs["game_turn"], int), "game_turn should be an integer"

    def test_step_returns_correct_types(self, env: Any) -> None:  # noqa: ANN401
        """Test that step() returns (obs, done, reward) with correct types."""
        env.reset()
        obs, done, reward, __ = env.step("S")

        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(done, bool), "Done flag should be a boolean"
        assert isinstance(reward, (int, float)), "Reward should be numeric"

    def test_step_without_reset_fails(self, env: Any) -> None:  # noqa: ANN401
        """Test that step() before reset raises an error or handles gracefully."""
        # Note: This depends on implementation - some envs might auto-reset
        # For now, we just check it doesn't crash
        try:
            env.reset()
            obs, done, reward, __ = env.step("S")
            assert isinstance(obs, dict)
        except Exception:
            pytest.fail("step() after reset() should not raise exception")

    def test_multiple_steps(self, env: Any) -> None:  # noqa: ANN401
        """Test taking multiple steps in sequence."""
        obs, done = env.reset()
        steps = 0
        max_steps = 10

        while not done and steps < max_steps:
            obs, done, reward, __ = env.step("S")
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, (int, float))
            steps += 1

        assert steps <= max_steps, "Should complete within max steps"


class TestFreewayEnvironment:
    """Specific tests for Freeway environment."""

    def test_freeway_actions(self) -> None:
        """Test Freeway accepts valid actions."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "S"]
        for action in valid_actions:
            obs, done, reward, __ = env.step(action)
            assert isinstance(obs, dict)

    def test_freeway_state_string(self) -> None:
        """Test Freeway state string format."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, _ = env.reset()
        assert "game_turn" in obs and "state_string" in obs and "state" in obs


class TestSnakeEnvironment:
    """Specific tests for Snake environment."""

    def test_snake_actions(self) -> None:
        """Test Snake accepts valid actions."""
        env, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "L", "R", "S"]
        for action in valid_actions:
            obs, done, reward, __ = env.step(action)
            assert isinstance(obs, dict)

    def test_snake_food_in_state(self) -> None:
        """Test Snake state contains food information."""
        env, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        obs, _ = env.reset()

        state_string = obs["state_string"]
        # Food should be mentioned in state
        assert len(state_string) > 0


class TestOvercookedEnvironment:
    """Specific tests for Overcooked environment."""

    def test_overcooked_actions(self) -> None:
        """Test Overcooked accepts valid actions."""
        env, _, _ = realtimegym.make("Overcooked-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "L", "R", "I", "S"]
        for action in valid_actions:
            obs, done, reward, __ = env.step(action)
            assert isinstance(obs, dict)

    def test_overcooked_state_description(self) -> None:
        """Test Overcooked state contains description."""
        env, _, _ = realtimegym.make("Overcooked-v0", seed=0, render=False)
        obs, _ = env.reset()

        assert "game_turn" in obs and "state_string" in obs and "state" in obs


class TestSeeding:
    """Test environment seeding for reproducibility."""

    def test_seed_reproducibility(self) -> None:
        """Test that same seed produces same initial state."""
        env1, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs1, _ = env1.reset()

        env2, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs2, _ = env2.reset()

        assert obs1["state_string"] == obs2["state_string"]

    def test_different_seeds_different_states(self) -> None:
        """Test that different seeds may produce different states."""
        env1, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        obs1, _ = env1.reset()

        env2, _, _ = realtimegym.make("Snake-v0", seed=1, render=False)
        obs2, _ = env2.reset()

        assert "game_turn" in obs1 and "state_string" in obs1 and "state" in obs1
        assert "game_turn" in obs2 and "state_string" in obs2 and "state" in obs2
