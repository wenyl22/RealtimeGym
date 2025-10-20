import re

from transformers import AutoTokenizer

from .base import BaseAgent, extract_boxed


class AgileThinker(BaseAgent):
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
        self.model2 = kwargs.get("model2", None)
        if "deepseek-reasoner" in self.model2:
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        if "gpt-oss-20b" in self.model2:
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        if "Qwen3-30B-A3B-Thinking-2507-FP8" in self.model2:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
            )
        # TODO: add more open source model's tokenizer here

    def truncate_logs(self):
        final_step = 0
        for i in range(len(self.logs["action"])):
            if "model2_prompt" in self.logs and self.logs["model2_prompt"][i] != "":
                final_step = i
        for col in self.logs:
            self.logs[col] = self.logs[col][:final_step]

    def think(self, timeout=None):
        """Process observation using dual reasoning systems with given timeout."""
        if self.current_observation is None:
            self.action = self.prompts.DEFAULT_ACTION
            return

        budget = timeout if timeout is not None else self.internal_budget
        observation = self.current_observation
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
        self.plan = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {turn} \)\n{text}"""
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
