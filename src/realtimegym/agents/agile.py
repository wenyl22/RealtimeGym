from .base import BaseAgent, extract_boxed
from openai import OpenAI
from transformers import AutoTokenizer

class AgileThinker(BaseAgent):
    def __init__(self, prompts, file, budget_form, port, api_key, internal_budget, **kwargs):

        super().__init__(prompts, file, budget_form, port, api_key, internal_budget)
        self.model1 = kwargs.get('model1', None)
        self.model2 = kwargs.get('model2', None)
        if "deepseek-reasoner" in self.model2:
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        # TODO: add more open source model's tokenizer here

    def think(self, observation, budget):
        self.state_string = observation['state_string']
        game_turn = observation['game_turn']
        prompt = ""
        if self.gen_text == "": # check whether the last generation is finished
            messages = [ {"role": "user", "content": self.prompts.SLOW_AGENT_PROMPT + self.prompts.CONCLUSION_FORMAT_PROMPT + observation['model2_description']} ]
            prompt = messages[-1]['content']
        else:
            messages = []
        text, token_num, turn = self.planning_inference(messages, budget, game_turn)
        self.plan = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {turn} \)\n{text}"""
        if self.log_thinking:
            self.logs['plan'].append(self.plan)
            self.logs['model2_prompt'].append(prompt)
            self.logs['model2_response'].append(text)
            self.logs['model2_token_num'].append(token_num)

        prompt = self.prompts.FAST_AGENT_PROMPT + observation['model1_description']
        if self.plan is not None:
            lines = self.plan.split('\n')
            for line in lines:
                prompt += f"> {line.strip()}\n"
        messages = [ {"role": "user", "content": prompt} ]
        text, token_num = self.reactive_inference(messages)
        self.action = extract_boxed(text)
        if self.log_thinking:
            self.logs['model1_prompt'].append(prompt)
            self.logs['model1_response'].append(text)
            self.logs['model1_token_num'].append(token_num)


