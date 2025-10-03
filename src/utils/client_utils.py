from openai import OpenAI
from anthropic import AnthropicVertex
from transformers import AutoTokenizer
from google import genai
import time
from together import Together
from vllm import SamplingParams
from typing import List, Dict

def Generate(llm: OpenAI | AnthropicVertex | genai.Client | Together, model: str, messages: List[Dict], sampling_params: SamplingParams, fast: bool) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "timeout": 600,
    }
    #print(model, fast, llm)
    while True:
        try:
            text = ""
            response = llm.chat.completions.create(**params)
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                text = '<think>' + response.choices[0].message.reasoning_content + "\n</think>\n"
            if response.choices[0].message.content != None:
                text += response.choices[0].message.content
            token_num = response.usage.completion_tokens
            return dict(text=text, token_num=token_num)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

class LLMClient:
    def __init__(self, args, api_keys):
        self.api_keys = api_keys[0]
        self.to_flush = ""
        self.to_flush_turn = 0
        self.gen_turn = 0
        self.gen_accum = 0
        self.gen_text = ""
        self.gen_token = []
        self.gen_token_num = 0
#        self.turns = []
        self.slow_llm = None
        self.fast_llm = None

        self.token_per_tick = args.token_per_tick
        self.fast_max_token = args.fast_max_token
        self.method = args.method
        self.slow_model = args.slow_model
        self.slow_base_url = args.slow_base_url
        self.fast_model = args.fast_model
        self.fast_base_url = args.fast_base_url
        self.add_new_thread()

    def add_new_thread(self):
        if self.method != "fast":
            if "claude" in self.slow_model:
                self.slow_llm = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            elif "gemini" in self.slow_model:
                self.slow_llm = genai.Client(vertexai=True, project="gcp-multi-agent", location="global")
            elif "tg" in self.api_keys:
                self.slow_llm = Together(api_key=self.api_keys)
            else:
                self.slow_llm = OpenAI(base_url=self.slow_base_url, api_key=self.api_keys)
                print(f"Using slow model: {self.slow_model} at {self.slow_base_url} with API key {self.api_keys}")
            if self.slow_model in ["deepseek-reasoner", "deepseek-ai/DeepSeek-R1"]:
                self.slow_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
            if self.slow_model == "Qwen/Qwen3-235B-A22B-Thinking-2507":
                self.slow_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Thinking-2507")
        if self.method != "slow":
            if "claude" in self.fast_model:
                self.fast_llm = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            elif "gemini" in self.fast_model:
                self.fast_llm = genai.Client(vertexai=True, project="gcp-multi-agent", location="global")
            elif "tg" in self.api_keys:
                self.fast_llm = Together(api_key = self.api_keys)
            else:
                print(self.fast_model, self.fast_base_url, self.api_keys)
                self.fast_llm = OpenAI(base_url=self.fast_base_url, api_key=self.api_keys)
                print(f"Using fast model: {self.fast_model} at {self.fast_base_url} with API key {self.api_keys}")

    def generate(self, messages, sampling_params, fast = False):
        if messages == []:
            return {"text": "", "token_num": 0}
        llm = self.slow_llm if not fast else self.fast_llm
        model = self.slow_model if not fast else self.fast_model            
        return Generate(llm, model, messages, sampling_params, fast)
    def run_slow_inference(self, messages, sampling_params, turn):
        _token_num = 0
        # new input, i.e. trigger slow system
        if messages != []:
            self.gen_turn = turn
            self.gen_accum = -self.fast_max_token
            response = self.generate(messages, sampling_params)
            self.gen_text = response['text']
            self.gen_token_num = response['token_num']
            if self.method == "parallel":
                self.gen_token = self.slow_tokenizer.encode(self.gen_text)
            _token_num = self.gen_token_num
        # generate self.token_per_tick tokens
        self.gen_accum += self.token_per_tick
        # prepare flush content
        can_flush = self.gen_accum >= self.gen_token_num or self.method == "parallel"
        if can_flush:
            self.to_flush_turn = self.gen_turn
            if self.gen_accum >= self.gen_token_num:
                self.to_flush = self.gen_text
            elif self.method == "parallel":
                self.to_flush = self.slow_tokenizer.decode(self.gen_token[:self.gen_accum], skip_special_tokens=True)
        _text = self.to_flush
        _turn = self.to_flush_turn
        self.to_flush = ""
        # check for completion: set the system in idle state
        if self.gen_accum + self.fast_max_token >= self.gen_token_num:
            self.gen_text = ""
        return _text, _turn, _token_num

    def run_fast_inference(self, messages, sampling_params, ALL_ACTIONS, DEFAULT_ACTION):
        response = self.generate(messages, sampling_params, fast = True)
        text = response['text']
        token_num = response['token_num']
        if "<think>" in text and "</think>" not in text:
            text += "</think>"
        if "oxed" in text.split('</think>')[-1]:
            return text, token_num
        text += "\n Therefore, the final answer is \\boxed{"
        new_message = messages.copy() + [{"role": "assistant", "content": text}]
        max_attempt = 3
        while max_attempt > 0:
            max_attempt -= 1
            try:
                response = self.fast_llm.chat.completions.create(
                    model=self.fast_model, messages=new_message, max_tokens=1, temperature=0, top_p=1,
                )
                if response.choices[0].message.content.strip()[0] in ALL_ACTIONS:
                    text += response.choices[0].message.content.strip()[0] + '}'
                    break
            except Exception as e:
                time.sleep(1)
            if max_attempt == 0:
                text += DEFAULT_ACTION + '}'
        return text, token_num

class WallTimeLLMClients:
    def __init__(self, args, api_keys):
        self.fast_model = args.fast_model
        self.slow_model = args.slow_model
        self.fast_base_url = args.fast_base_url
        self.slow_base_url = args.slow_base_url
        self.fast_llm = OpenAI(base_url=self.fast_base_url, api_key=api_keys)
        self.slow_llm = OpenAI(base_url=self.slow_base_url, api_key=api_keys)

    def generate(self, messages, sampling_params, fast = False):
        client = self.fast_llm if fast else self.slow_llm
        model = self.fast_model if fast else self.slow_model
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "stream": True,
        }
        #print("Messages:", messages)
        response = client.chat.completions.create(**params)
        return response

class SimpleLLMClients:
    def __init__(self, args, api_keys):
        self.api_keys = api_keys
        self.fast_model = args.fast_model
        self.slow_model = args.slow_model
        self.fast_base_url = args.fast_base_url
        self.slow_base_url = args.slow_base_url
        self.fast_llm = OpenAI(base_url=self.fast_base_url, api_key=api_keys)
        self.slow_llm = OpenAI(base_url=self.slow_base_url, api_key=api_keys)
    
    def get_fast_response(self, prompt):
        response = self.fast_llm.responses.create(
            model=self.fast_model,
            reasoning={"effort": "low", "summary": "detailed"},
            input=[{"role": "user", "content": prompt}],
        )
        return response

    def get_slow_response(self, prompt):
        response = self.slow_llm.responses.create(
            model=self.slow_model,
            reasoning={"effort": "high", "summary": "detailed"},
            input=[{"role": "user", "content": prompt}],
        )
        return response
    def get_slow_response_streaming(self, prompt, cancel_event):
        response = self.slow_llm.responses.create(
            model=self.slow_model,
            reasoning={"effort": "high", "summary": "detailed"},
            input=[{"role": "user", "content": prompt}],
            stream=True,
        )
        return response
            