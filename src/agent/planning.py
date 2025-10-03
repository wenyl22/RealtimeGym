import re
from agent.base import BaseAgent, extract_boxed

class PlanningAgent(BaseAgent):
    def __init__(self, prompts, file, budget_form, port, api_key, internal_budget, **kwargs):
        assert internal_budget == 0, "Internal budget must be a 0 for PlanningAgent."
        super().__init__(prompts, file, budget_form, port, api_key, internal_budget)
        self.model2 = kwargs.get('model2', None)

    def think(self, observation, budget):
        assert budget is not None
        self.state_string = observation['state_string']
        game_turn = observation['game_turn']
        prompt = ""
        if self.gen_text == "": # check whether the last generation is finished
            messages = [ {"role": "user", "content": self.prompts.SLOW_AGENT_PROMPT + self.prompts.ACTION_FORMAT_PROMPT + observation['model2_description']} ]
            prompt = messages[-1]['content']
        else:
            messages = []
 
        text, token_num, turn = self.planning_inference(messages, budget, game_turn)
        temp = extract_boxed(text)
        if temp != "":
            self.plan = re.sub(r'[^' + self.prompts.ALL_ACTIONS + ']', '', temp)
            self.plan = self.plan[observation['game_turn'] - turn:] if len(self.plan) > observation['game_turn'] - turn else "" 
        if self.log_thinking:
            self.logs['plan'].append(self.plan)
            self.logs['model2_prompt'].append(prompt)
            self.logs['model2_response'].append(text)
            self.logs['model2_token_num'].append(token_num)
        self.action = self.plan[0] if self.plan != "" else self.prompts.DEFAULT_ACTION
        self.plan = self.plan[1:] if self.plan != "" else ""
