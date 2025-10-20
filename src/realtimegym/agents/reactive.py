import re

from .base import BaseAgent, extract_boxed


class ReactiveAgent(BaseAgent):
    def __init__(
        self,
        prompts,
        file,
        budget_form,
        port1,
        port2,
        api_key,
        internal_budget,
        **kwargs,
    ):
        super().__init__(
            prompts, file, budget_form, port1, port2, api_key, internal_budget
        )
        self.model1 = kwargs.get("model1", None)

    def truncate_logs(self):
        return

    def think(self, timeout=None):
        """Process observation and generate action with given timeout."""
        if self.current_observation is None:
            self.action = self.prompts.DEFAULT_ACTION
            return

        budget = timeout if timeout is not None else self.internal_budget
        observation = self.current_observation

        messages = [
            {
                "role": "user",
                "content": self.prompts.FAST_AGENT_PROMPT
                + observation["model1_description"],
            }
        ]
        text, token_num = self.reactive_inference(messages, budget)
        self.action = re.sub(
            r"[^" + self.prompts.ALL_ACTIONS + "]", "", extract_boxed(text)
        )
        if self.action == "":
            self.action = self.prompts.DEFAULT_ACTION
        if self.log_thinking:
            self.logs["plan"].append("N/A")
            self.logs["model1_prompt"].append(messages[-1]["content"])
            self.logs["model1_response"].append(text)
        self.logs["model1_token_num"].append(token_num)
