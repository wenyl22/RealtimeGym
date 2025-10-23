import re

from .base import BaseAgent, extract_boxed


class ReactiveAgent(BaseAgent):
    def __init__(
        self,
        prompts,
        file,
        time_unit,
        model1_config,
        internal_budget,
    ):
        super().__init__(prompts, file, time_unit)
        self.config_model1(model1_config, internal_budget)

    def truncate_logs(self):
        return

    def think(self, timeout=None):
        assert self.current_observation is not None and timeout is not None
        budget = timeout
        observation = self.current_observation
        prompt_gen = self.prompts.state_to_description(
            observation["state"], mode="reactive"
        )
        messages = [{"role": "user", "content": prompt_gen}]
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
