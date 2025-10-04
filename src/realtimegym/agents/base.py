from collections import defaultdict
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
        # better not set log_thinking to True for time-based budget, since storing logs can be slow and interfere with timing
        self.log_thinking = True if budget_form == "token" else False
        self.action = prompts.DEFAULT_ACTION
        self.to_flush = ""
        self.to_flush_turn = 0
        self.gen_turn = 0
        self.gen_accum = 0
        self.gen_text = ""
        self.gen_token = []
        self.gen_token_num = 0
        self.planning_stream = None
        self.planning_reasoning = False
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

    def streaming_generate(self, model: str, messages: List[Dict], sampling_params: Dict, stream_obj = None) -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": sampling_params.get("max_tokens", 32768),
            "temperature": sampling_params.get("temperature", 1),
            "top_p": sampling_params.get("top_p", 1),
            "stream": True,
        }
        max_time = sampling_params["max_time"]
        start_time = time.time()
        text, token_num = "", 0
        if stream_obj is None:
            while True:
                try:
                    stream_obj = self.llm.chat.completions.create(**params)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(1)
        try:
            for chunk in stream_obj:
                if time.time() - start_time > max_time:
                    return text, token_num, stream_obj
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content != None:
                    if self.planning_reasoning == False:
                        text += "<think>"
                        self.planning_reasoning = True
                    text += chunk.choices[0].delta.reasoning_content
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content != None:
                    if self.planning_reasoning == True:
                        text += "\n</think>\n"
                        self.planning_reasoning = False
                    text += chunk.choices[0].delta.content
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    token_num = chunk.usage.completion_tokens
        except Exception as e:
            print(f"Error during streaming: {e}")
            return text, token_num, None
        # uncomment the following line for realistic simulation
        # time.sleep(max_time - (time.time() - start_time) if max_time - (time.time() - start_time) > 0 else 0)
        print(token_num, time.time() - start_time)
        return text, token_num, None

    def reactive_inference(self, messages):
        assert self.model1 is not None, "Reactive LLM is not initialized!"
        if self.budget_form == "token":
            sampling_params = {"max_tokens": self.internal_budget, "temperature": 1, "top_p": 1}
            text, token_num = self.generate(self.model1, messages, sampling_params)
        else:
            sampling_params = {"max_time": self.internal_budget, "temperature": 1, "top_p": 1, "max_tokens": 8192}
            text, token_num, _ = self.streaming_generate(self.model1, messages, sampling_params)
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
                token_num = self.gen_token_num
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
            sampling_params = {"max_time": budget - self.internal_budget, "temperature": 0.6, "top_p": 0.95}
            if messages != []:
                self.gen_turn = game_turn

            new_text, token_num, self.planning_stream = self.streaming_generate(self.model2, messages, sampling_params, self.planning_stream)
            self.gen_text += new_text
            text = self.gen_text
            turn = self.gen_turn
            if self.planning_stream is None:
                self.gen_text = ""
        return text, token_num, turn
        
        
import re
def extract_text(text, default_value=""):
    """
    Extracts the text{...} from the input string.
    Returns the last match found or the default value if no match is found.
    """
    matches = re.findall(r'ext{(.*?)}', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip() if text else default_value


def extract_boxed(text, default_value=""):
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r'oxed{' 
    start_index = text.rfind(pattern)
    if start_index == -1:
        # Try to extract content enclosed in triple backticks if \boxed{...} is not found
        triple_backtick_pattern = r"```(.*?)```"
        matches = re.findall(triple_backtick_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return default_value
    start_index += len(pattern) - 1
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            if stack:
                stack.pop()
            if not stack:
                if 'ext{' in text[start_index:i]:
                    return extract_text(text[start_index:i])
                return text[start_index + 1:i].strip()
    return default_value if default_value else text[start_index + 1:].strip()
