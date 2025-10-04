from .base import BaseAgent, extract_boxed

class ReactiveAgent(BaseAgent):
    def __init__(self, prompts, file, budget_form, port, api_key, internal_budget, **kwargs):
        super().__init__(prompts, file, budget_form, port, api_key, internal_budget)
        self.model1 = kwargs.get('model1', None)
    
    def think(self, observation, budget):
        self.state_string = observation['state_string']
        messages = [ {"role": "user", "content": self.prompts.FAST_AGENT_PROMPT + observation['model1_description']} ]
        text, token_num = self.reactive_inference(messages)
        self.action = extract_boxed(text)
        if self.log_thinking:
            self.logs['plan'].append("N/A")
            self.logs['model1_prompt'].append(messages[-1]['content'])
            self.logs['model1_response'].append(text)
            self.logs['model1_token_num'].append(token_num)        
