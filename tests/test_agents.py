"""Tests for RealtimeGym agents API."""

import pytest

import realtimegym


class MockAgent:
    """Mock agent for testing the agent API pattern."""

    def __init__(self):
        self.current_observation = None
        self.thought_count = 0
        self.action = "S"

    def observe(self, observation):
        """Receive observation from environment."""
        assert isinstance(observation, dict), "Observation should be a dict"
        self.current_observation = observation

    def think(self, timeout=None):
        """Process observation and decide action."""
        if self.current_observation is None:
            # Handle gracefully when no observation available
            return
        self.thought_count += 1
        # Simple logic for testing
        turn = self.current_observation.get("game_turn", 0)
        self.action = "U" if turn % 2 == 0 else "S"

    def act(self):
        """Return the chosen action."""
        return self.action


class TestAgentAPI:
    """Test the agent API pattern: observe -> think -> act."""

    @pytest.fixture(params=["Freeway-v0", "Snake-v0", "Overcooked-v0"])
    def env_and_agent(self, request):
        """Fixture providing environment and mock agent."""
        env, _, _ = realtimegym.make(request.param, seed=0, render=False)
        agent = MockAgent()
        return env, agent

    def test_observe_think_act_loop(self, env_and_agent):
        """Test the complete observe -> think -> act loop."""
        env, agent = env_and_agent
        DEFAULT_ACTION = "S"

        obs, done = env.reset()

        # Test loop for a few steps
        steps = 0
        max_steps = 5

        while not done and steps < max_steps:
            # Agent observes
            agent.observe(obs)
            assert agent.current_observation is not None

            # Agent thinks
            agent.think(timeout=8192)
            assert agent.thought_count > 0

            # Agent acts
            action = agent.act()
            if action is None:
                action = DEFAULT_ACTION

            # Environment steps
            obs, done, reward = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, (int, float))

            steps += 1

        assert steps > 0, "Should have taken at least one step"

    def test_agent_observe_updates_state(self):
        """Test that observe() properly updates agent state."""
        agent = MockAgent()
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, _ = env.reset()

        assert agent.current_observation is None
        agent.observe(obs)
        assert agent.current_observation is not None
        assert agent.current_observation == obs

    def test_agent_think_requires_observation(self):
        """Test that think() should be called after observe()."""
        agent = MockAgent()

        # This should not crash, but agent should handle gracefully
        agent.think(timeout=100)

    def test_agent_act_returns_action(self):
        """Test that act() returns a valid action."""
        agent = MockAgent()
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, _ = env.reset()

        agent.observe(obs)
        agent.think(timeout=100)
        action = agent.act()

        assert action is not None
        assert isinstance(action, str)

    def test_multiple_observe_think_cycles(self):
        """Test multiple observe -> think cycles update agent state."""
        agent = MockAgent()
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, done = env.reset()

        initial_thought_count = agent.thought_count

        for i in range(3):
            agent.observe(obs)
            agent.think(timeout=100)
            action = agent.act() or "S"
            obs, done, _ = env.step(action)

        assert agent.thought_count == initial_thought_count + 3


class TestAgentIntegration:
    """Integration tests combining environments and agents."""

    def test_complete_game_loop(self):
        """Test a complete game with agent."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        agent = MockAgent()
        DEFAULT_ACTION = "S"

        obs, done = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 20

        while not done and steps < max_steps:
            agent.observe(obs)
            agent.think(timeout=8192)
            action = agent.act() or DEFAULT_ACTION
            obs, done, reward = env.step(action)
            total_reward += reward
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, (int, float))

    def test_timeout_parameter(self):
        """Test that think() accepts timeout parameter."""
        agent = MockAgent()
        env, _, _ = realtimegym.make("Snake-v0", seed=0, render=False)
        obs, _ = env.reset()

        agent.observe(obs)

        # Test different timeout values
        for timeout in [100, 1000, 8192]:
            agent.think(timeout=timeout)
            action = agent.act()
            assert action is not None

    def test_default_action_fallback(self):
        """Test that None action can fallback to default."""
        env, _, _ = realtimegym.make("Freeway-v0", seed=0, render=False)
        obs, _ = env.reset()

        DEFAULT_ACTION = "S"

        # Simulate agent returning None
        action = None or DEFAULT_ACTION

        obs, done, reward = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(done, bool)


class TestRealAgents:
    """Test that real agent classes have the correct API."""

    def test_base_agent_has_observe(self):
        """Test BaseAgent has observe method."""
        BaseAgent = realtimegym.BaseAgent
        assert hasattr(BaseAgent, "observe")

    def test_base_agent_has_think(self):
        """Test BaseAgent has think method."""
        BaseAgent = realtimegym.BaseAgent
        assert hasattr(BaseAgent, "think")

    def test_base_agent_has_act(self):
        """Test BaseAgent has act method."""
        BaseAgent = realtimegym.BaseAgent
        assert hasattr(BaseAgent, "act")

    def test_base_agent_observe_signature(self):
        """Test BaseAgent.observe has correct signature."""
        import inspect

        BaseAgent = realtimegym.BaseAgent
        sig = inspect.signature(BaseAgent.observe)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "observation" in params

    def test_base_agent_think_signature(self):
        """Test BaseAgent.think has correct signature."""
        import inspect

        BaseAgent = realtimegym.BaseAgent
        sig = inspect.signature(BaseAgent.think)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "timeout" in params

    def test_base_agent_act_signature(self):
        """Test BaseAgent.act has correct signature."""
        import inspect

        BaseAgent = realtimegym.BaseAgent
        sig = inspect.signature(BaseAgent.act)
        params = list(sig.parameters.keys())

        assert "self" in params
