from collections import defaultdict
from utils.extract_utils import extract_boxed
from openai import OpenAI
from typing import List, Dict
import time
import pandas as pd
class BaseAgent:
    def __init__(self, prompts, file, budget_form, port, api_key, internal_budget):
        self.prompts = prompts
        self.file = file
        self.budget_form = budget_form
        self.llm = OpenAI(api_key = api_key, base_url=port)
        self.model1 = None
        self.model2 = None
        self.tokenizer = None
        self.internal_budget = internal_budget

        self.logs = defaultdict(list)
        self.log_thinking = True if budget_form == "token" else False
        self.action = prompts.DEFAULT_ACTION
        self.to_flush = ""
        self.to_flush_turn = 0
        self.gen_turn = 0
        self.gen_accum = 0
        self.gen_text = ""
        self.gen_token = []
        self.gen_token_num = 0
        
        self.plan = ""
        self.state_string = ""

    def think(self, observation, budget):
        raise NotImplementedError("This method should be overridden by subclasses.")
    def act(self):
        return self.action
    def log(self, reward, reset):
        self.logs['render'].append(self.state_string)
        self.logs['action'].append(self.action)
        self.logs['reward'].append(reward)
        if reset == True:
            self.plan = ""
            while self.gen_text != "":
                print(self.gen_accum, self.gen_token_num)
                self.planning_inference([], 32768, 0)
            self.to_flush = ""
        df = pd.DataFrame(self.logs)
        df.to_csv(self.file)

    def generate(self, model: str, messages: List[Dict], sampling_params: Dict) -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": sampling_params.get("max_tokens", 32768),
            "temperature": sampling_params.get("temperature", 1),
            "top_p": sampling_params.get("top_p", 1),
            "timeout": 600,
        }
        while True:
            try:
                text = ""
                response = self.llm.chat.completions.create(**params)
                if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                    text = '<think>' + response.choices[0].message.reasoning_content + "\n</think>\n"
                if response.choices[0].message.content != None:
                    text += response.choices[0].message.content
                token_num = response.usage.completion_tokens
                print(token_num)
                return text, token_num
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def reactive_inference(self, messages):
        assert self.model1 is not None, "Reactive LLM is not initialized!"
        if self.budget_form == "token":
            sampling_params = {"max_tokens": self.internal_budget, "temperature": 1, "top_p": 1}
            text, token_num = self.generate(self.model1, messages, sampling_params)
        else:
            raise NotImplementedError("This method only supports token-based budget now.")
        if "<think>" in text and "</think>" not in text:
            text += "</think>"
        if "oxed" in text.split('</think>')[-1]:
            return text, token_num
        ### s1 budget forcing ###
        text += "\nTherefore, the final answer is \\boxed{"
        max_attempt = 3
        while max_attempt > 0:
            max_attempt -= 1
            try:
                response = self.llm.chat.completions.create(
                    model=self.model1, 
                    messages=messages + [{"role": "assistant", "content": text}], 
                    max_tokens=1, temperature=0, top_p=1,
                )
                if response.choices[0].message.content.strip()[0] in self.prompts.ALL_ACTIONS:
                    text += response.choices[0].message.content.strip()[0] + '}'
                    break
            except Exception as e:
                time.sleep(0.2)
            if max_attempt == 0:
                text += self.prompts.DEFAULT_ACTION + '}'
        return text, token_num

    def planning_inference(self, messages, budget, game_turn):
        assert self.model2 is not None, "Planning LLM is not initialized!"
        token_num = 0
        if self.budget_form == "token":
            if messages != []:
                self.gen_turn = game_turn
                self.gen_accum = -self.internal_budget
                sampling_params = {"max_tokens": 32768, "temperature": 0.6, "top_p": 0.95}
                self.gen_text, self.gen_token_num = self.generate(self.model2, messages, sampling_params)
                if self.tokenizer is not None:
                    self.gen_token = self.tokenizer.encode(self.gen_text)
                _token_num = self.gen_token_num
            self.gen_accum += budget
            can_flush = self.gen_accum >= self.gen_token_num or self.tokenizer is not None
            if can_flush:
                self.to_flush_turn = self.gen_turn
                if self.gen_accum >= self.gen_token_num:
                    self.to_flush = self.gen_text
                elif self.tokenizer is not None:
                    self.to_flush = self.tokenizer.decode(self.gen_token[:self.gen_accum], skip_special_tokens=True)
            text = self.to_flush
            turn = self.to_flush_turn
            self.to_flush = ""
            # check for completion: set the system in idle state
            if self.gen_accum + self.internal_budget >= self.gen_token_num:
                self.gen_text = ""
        else:
            raise NotImplementedError("This method only supports token-based budget now.")
        return text, token_num, turn