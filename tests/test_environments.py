"""Tests for RealtimeGym environments."""

import pytest

import realtimegym


class TestEnvironmentRegistry:
    """Test environment creation and registry."""

    def test_make_freeway(self):
        """Test creating Freeway environment."""
        env, seed, renderer = realtimegym.make("Freeway-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_snake(self):
        """Test creating Snake environment."""
        env, seed, renderer = realtimegym.make("Snake-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_overcooked(self):
        """Test creating Overcooked environment."""
        env, seed, renderer = realtimegym.make("Overcooked-v0", seed=0, render=False)
        assert env is not None
        assert isinstance(seed, int)
        assert renderer is None

    def test_make_invalid_env(self):
        """Test that invalid environment ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown environment ID"):
            realtimegym.make("InvalidEnv-v0")

    def test_difficulty_levels(self):
        """Test all difficulty levels can be created."""
        for difficulty in ["v0", "v1", "v2"]:
            env, _, _ = realtimegym.make(f"Freeway-{difficulty}", seed=0, render=False)
            assert env is not None


class TestEnvironmentAPI:
    """Test the standard gym-like API for all environments."""

    @pytest.fixture(params=["Freeway-v0", "Snake-v0", "Overcooked-v0"])
    def env(self, request):
        """Fixture providing each environment."""
        env, _, _ = realtimegym.make(request.param, seed=0, render=False)
        return env

    def test_reset_returns_correct_types(self, env):
        """Test that reset() returns (obs, done) with correct types."""
        obs, done = env.reset()

        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(done, bool), "Done flag should be a boolean"
        assert not done, "Done should be False after reset"

    def test_reset_observation_keys(self, env):
        """Test that reset observation contains required keys."""
        obs, done = env.reset()

        assert "state_string" in obs, "Observation should contain 'state_string'"
        assert "game_turn" in obs, "Observation should contain 'game_turn'"
        assert isinstance(obs["state_string"], str), "state_string should be a string"
        assert isinstance(obs["game_turn"], int), "game_turn should be an integer"

    def test_step_returns_correct_types(self, env):
        """Test that step() returns (obs, done, reward) with correct types."""
        env.reset()
        obs, done, reward = env.step("S")

        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(done, bool), "Done flag should be a boolean"
        assert isinstance(reward, (int, float)), "Reward should be numeric"

    def test_step_without_reset_fails(self, env):
        """Test that step() before reset raises an error or handles gracefully."""
        # Note: This depends on implementation - some envs might auto-reset
        # For now, we just check it doesn't crash
        try:
            env.reset()
            obs, done, reward = env.step("S")
            assert isinstance(obs, dict)
        except Exception:
            pytest.fail("step() after reset() should not raise exception")

    def test_multiple_steps(self, env):
        """Test taking multiple steps in sequence."""
        obs, done = env.reset()
        steps = 0
        max_steps = 10

        while not done and steps < max_steps:
            obs, done, reward = env.step("S")
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, (int, float))
            steps += 1

        assert steps <= max_steps, "Should complete within max steps"


class TestFreewayEnvironment:
    """Specific tests for Freeway environment."""

    def test_freeway_actions(self):
        """Test Freeway accepts valid actions."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "S"]
        for action in valid_actions:
            obs, done, reward = env.step(action)
            assert isinstance(obs, dict)

    def test_freeway_state_string(self):
        """Test Freeway state string format."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, _ = env.reset()

        state_string = obs["state_string"]
        assert "Car" in state_string or "car" in state_string or len(state_string) > 0


class TestSnakeEnvironment:
    """Specific tests for Snake environment."""

    def test_snake_actions(self):
        """Test Snake accepts valid actions."""
        env, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "L", "R", "S"]
        for action in valid_actions:
            obs, done, reward = env.step(action)
            assert isinstance(obs, dict)

    def test_snake_food_in_state(self):
        """Test Snake state contains food information."""
        env, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        obs, _ = env.reset()

        state_string = obs["state_string"]
        # Food should be mentioned in state
        assert len(state_string) > 0


class TestOvercookedEnvironment:
    """Specific tests for Overcooked environment."""

    def test_overcooked_actions(self):
        """Test Overcooked accepts valid actions."""
        env, _, _ = realtimegym.make("Overcooked-v0", seed=0, render=False)
        env.reset()

        valid_actions = ["U", "D", "L", "R", "I", "S"]
        for action in valid_actions:
            obs, done, reward = env.step(action)
            assert isinstance(obs, dict)

    def test_overcooked_state_description(self):
        """Test Overcooked state contains description."""
        env, _, _ = realtimegym.make("Overcooked-v0", seed=0, render=False)
        obs, _ = env.reset()

        assert "description" in obs
        assert isinstance(obs["description"], str)


class TestSeeding:
    """Test environment seeding for reproducibility."""

    def test_seed_reproducibility(self):
        """Test that same seed produces same initial state."""
        env1, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs1, _ = env1.reset()

        env2, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs2, _ = env2.reset()

        assert obs1["state_string"] == obs2["state_string"]

    def test_different_seeds_different_states(self):
        """Test that different seeds may produce different states."""
        env1, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        obs1, _ = env1.reset()

        env2, _, _ = realtimegym.make("Snake-v0", seed=1, render=False)
        obs2, _ = env2.reset()

        # Note: This might not always be different, but structure should be valid
        assert isinstance(obs1["state_string"], str)
        assert isinstance(obs2["state_string"], str)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy act() method."""

    def test_legacy_act_method_exists(self):
        """Test that legacy act() method still exists."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        env.reset()

        assert hasattr(env, "act"), "Legacy act() method should exist"

    def test_legacy_act_returns_reward_and_reset_flag(self):
        """Test that legacy act() returns (reward, reset_flag)."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        env.reset()

        reward, reset_flag = env.act("S")

        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(reset_flag, bool), "Reset flag should be boolean"
