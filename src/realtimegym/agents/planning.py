import re
from typing import Any, Optional

from .base import BaseAgent, extract_boxed


class PlanningAgent(BaseAgent):
    def __init__(
        self,
        prompts: Any,  # noqa: ANN401 - prompts is a dynamically loaded module
        file: str,
        time_unit: str,
        model2_config: str,
        skip_action: bool = False,
    ) -> None:
        super().__init__(prompts, file, time_unit)
        self.config_model2(model2_config)
        self.skip_action = skip_action

    def truncate_logs(self) -> None:
        final_step, final_plan = 0, ""
        for i in range(len(self.logs["action"])):
            if "model2_prompt" in self.logs and self.logs["model2_prompt"][i] != "":
                final_step = i
                final_plan = str(self.logs["plan"][i])
        self.plan = final_plan[1:] if final_plan != "nan" and final_plan != "" else ""
        for col in self.logs:
            self.logs[col] = self.logs[col][:final_step]

    def think(self, timeout: Optional[float] = None) -> None:
        assert timeout is not None and self.current_observation is not None
        budget = timeout

        observation = self.current_observation
        game_turn = observation["game_turn"]
        prompt_gen = self.prompts.state_to_description(
            observation["state"], mode="planning"
        )

        prompt = ""
        if self.gen_text == "":  # check whether the last generation is finished
            messages = [{"role": "user", "content": prompt_gen}]
            prompt = messages[-1]["content"]
        else:
            messages = []

        text, token_num, turn = self.planning_inference(messages, budget, game_turn)
        temp = extract_boxed(text)
        if temp != "":
            self.plan = re.sub(r"[^" + self.prompts.ALL_ACTIONS + "]", "", temp)
            if self.skip_action:
                self.plan = (
                    self.plan[observation["game_turn"] - turn :]
                    if len(self.plan) > observation["game_turn"] - turn
                    else ""
                )

        if self.log_thinking:
            self.logs["plan"].append(self.plan)
            self.logs["model2_prompt"].append(prompt)
            self.logs["model2_response"].append(text)
        self.logs["model2_token_num"].append(token_num)
        self.action = self.plan[0] if self.plan != "" else self.prompts.DEFAULT_ACTION
        self.plan = self.plan[1:] if self.plan != "" else ""
