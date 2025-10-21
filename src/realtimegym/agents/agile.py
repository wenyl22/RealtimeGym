import re

from transformers import AutoTokenizer

from .base import BaseAgent, extract_boxed


class AgileThinker(BaseAgent):
    def __init__(
        self,
        prompts,
        file,
        budget_form,
        model1_config,
        model2_config,
        internal_budget,
        **kwargs,
    ):
        super().__init__(
            prompts, file, budget_form, model1_config, model2_config, internal_budget
        )

    def truncate_logs(self):
        final_step = 0
        for i in range(len(self.logs["action"])):
            if "model2_prompt" in self.logs and self.logs["model2_prompt"][i] != "":
                final_step = i
        for col in self.logs:
            self.logs[col] = self.logs[col][:final_step]

    def think(self, timeout=None):
        assert self.current_observation is not None and timeout is not None
        budget = timeout
        observation = self.current_observation
        self.state_string = observation["state_string"]
        game_turn = observation["game_turn"]
        prompt = ""
        if self.gen_text == "":  # check whether the last generation is finished
            messages = [
                {
                    "role": "user",
                    "content": self.prompts.SLOW_AGENT_PROMPT
                    + self.prompts.CONCLUSION_FORMAT_PROMPT
                    + observation["model2_description"],
                }
            ]
            prompt = messages[-1]["content"]
        else:
            messages = []
        text, token_num, turn = self.planning_inference(messages, budget, game_turn)
        self.plan = f"""**Guidance from a Previous Thinking Model:** Turn \\( t_1 = {turn} \\)\n{text}"""
        if self.log_thinking:
            self.logs["plan"].append(self.plan)
            self.logs["model2_prompt"].append(prompt)
            self.logs["model2_response"].append(text)
        self.logs["model2_token_num"].append(token_num)

        prompt = self.prompts.FAST_AGENT_PROMPT + observation["model1_description"]
        if self.plan is not None:
            lines = self.plan.split("\n")
            for line in lines:
                prompt += f"> {line.strip()}\n"
        messages = [{"role": "user", "content": prompt}]
        text, token_num = self.reactive_inference(messages, self.internal_budget)
        self.action = re.sub(
            r"[^" + self.prompts.ALL_ACTIONS + "]", "", extract_boxed(text)
        )
        if self.log_thinking:
            self.logs["model1_prompt"].append(prompt)
            self.logs["model1_response"].append(text)
        self.logs["model1_token_num"].append(token_num)

