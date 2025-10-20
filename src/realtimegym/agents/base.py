import queue
import re
import threading
import time
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from openai import OpenAI


def regularize_param(sampling_params, model, messages):
    reasoning_effort = None
    if "gpt-oss" in model:
        reasoning_effort = model.split("-")[-1]
        model = model.replace("-" + reasoning_effort, "")
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.get("max_tokens", 80000),
        "temperature": sampling_params.get("temperature", 1),
        "top_p": sampling_params.get("top_p", 1),
    }
    if "deepseek-reasoner" in model:
        params["max_tokens"] = min(params["max_tokens"], 65536)
    elif "deepseek" in model or "Qwen" in model:
        params["max_tokens"] = min(params["max_tokens"], 32768)
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    return params


class BaseAgent:
    def __init__(
        self, prompts, file, budget_form, port1, port2, api_key, internal_budget
    ):
        self.prompts = prompts
        self.file = file
        self.budget_form = budget_form
        self.llm1 = OpenAI(api_key=api_key, base_url=port1)
        self.llm2 = OpenAI(api_key=api_key, base_url=port2)
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
        self.planning_queue = None
        self.planning_done = None
        self.planning_reasoning = False
        self.plan = ""
        self.state_string = ""

        # New API: store current observation
        self.current_observation = None

    def observe(self, observation):
        """
        Receive and store the current observation from the environment.

        Args:
            observation (dict): Observation containing state information
        """
        self.current_observation = observation
        self.state_string = observation.get("state_string", "")

    def think(self, timeout=None):
        """
        Process the current observation and decide on an action.

        Args:
            timeout: Time/token budget for thinking (optional, uses internal_budget if not provided)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def act(self):
        """
        Return the chosen action. Returns None if no action was decided.

        Returns:
            action (str or None): The action to take, or None for default action
        """
        return self.action if self.action != "" else None

    def log(self, reward, reset):
        self.logs["render"].append(self.state_string)
        self.logs["action"].append(self.action)
        self.logs["reward"].append(reward)
        if reset:
            self.plan = ""
            if self.budget_form == "token":
                while self.gen_text != "":
                    self.planning_inference([], 80000, 0)
            else:
                while self.gen_text != "":
                    self.planning_inference([], 10.0 + self.internal_budget, 0)
            self.to_flush = ""
        df = pd.DataFrame(self.logs)
        df.to_csv(self.file)

    def resume_from_checkpoint(self, env, checkpoint_file):
        df = pd.read_csv(checkpoint_file)
        self.logs = df.to_dict("list")  # remove unnamed column
        self.logs.pop("Unnamed: 0", None)
        self.truncate_logs()
        for a in self.logs["action"]:
            env.act(a)
        df = pd.DataFrame(self.logs)
        df.to_csv(self.file)

    def truncate_logs(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generate(
        self, llm, model: str, messages: List[Dict], sampling_params: Dict
    ) -> tuple[str, int]:
        params = regularize_param(sampling_params, model, messages)
        while True:
            try:
                text = ""
                response = llm.chat.completions.create(**params)
                if (
                    hasattr(response.choices[0].message, "reasoning_content")
                    and response.choices[0].message.reasoning_content is not None
                ):
                    text = (
                        "<think>"
                        + response.choices[0].message.reasoning_content
                        + "\n</think>\n"
                    )
                if response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                token_num = response.usage.completion_tokens
                return text, token_num
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def start_planning_stream(
        self, llm, model: str, messages: List[Dict], sampling_params: Dict
    ) -> None:
        self.planning_queue = queue.Queue()
        self.planning_done = threading.Event()
        params = regularize_param(sampling_params, model, messages)
        params["stream"] = True

        def planning_worker():
            try:
                stream_obj = llm.chat.completions.create(**params)
                for chunk in stream_obj:
                    self.planning_queue.put(chunk)
                self.planning_done.set()
            except Exception as e:
                print(f"Streaming error: {e}")
                self.planning_done.set()

        threading.Thread(target=planning_worker, daemon=True).start()

    def get_planning_chunks(self):
        text = ""
        token_num = 0
        while not self.planning_queue.empty():
            chunk = self.planning_queue.get()
            if (
                hasattr(chunk.choices[0].delta, "reasoning_content")
                and chunk.choices[0].delta.reasoning_content is not None
            ):
                if not self.planning_reasoning:
                    text += "<think>"
                    self.planning_reasoning = True
                text += chunk.choices[0].delta.reasoning_content
            if (
                hasattr(chunk.choices[0].delta, "content")
                and chunk.choices[0].delta.content is not None
            ):
                if self.planning_reasoning:
                    text += "\n</think>\n"
                    self.planning_reasoning = False
                text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                token_num = chunk.usage.completion_tokens
        return text, token_num

    def is_planning_finished(self):
        return self.planning_done.is_set()

    def start_reactive_stream(self, llm, model, messages, sampling_params, max_time):
        params = regularize_param(sampling_params, model, messages)
        params["stream"] = True
        start_time = time.time()
        stream_obj = llm.chat.completions.create(**params)
        text, token_num = "", 0
        for chunk in stream_obj:
            if time.time() - start_time > max_time:
                break
            if (
                hasattr(chunk.choices[0].delta, "content")
                and chunk.choices[0].delta.content is not None
            ):
                text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                token_num = chunk.usage.completion_tokens
        current_time = time.time()
        if max_time - (current_time - start_time) > 0:
            time.sleep(max_time - (current_time - start_time))
        return text, token_num
        # max_time = sampling_params["max_time"]
        # start_time = time.time()
        # text, token_num = "", 0
        # if stream_obj is None:
        #     while True:
        #         try:
        #             stream_obj = llm.chat.completions.create(**params)
        #             break
        #         except Exception as e:
        #             print(f"Error: {e}")
        #             time.sleep(1)
        # try:
        #     for chunk in stream_obj:
        #         if time.time() - start_time > max_time:
        #             return text, token_num, stream_obj
        #         if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content != None:
        #             if self.planning_reasoning == False:
        #                 text += "<think>"
        #                 self.planning_reasoning = True
        #             text += chunk.choices[0].delta.reasoning_content
        #         if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content != None:
        #             if self.planning_reasoning == True:
        #                 text += "\n</think>\n"
        #                 self.planning_reasoning = False
        #             text += chunk.choices[0].delta.content
        #         if hasattr(chunk, 'usage') and chunk.usage is not None:
        #             token_num = chunk.usage.completion_tokens
        # except Exception as e:
        #     print(f"Error during streaming: {e}")
        #     return text, token_num, None
        # return text, token_num, None

    def reactive_inference(self, messages, budget):
        assert self.model1 is not None, "Reactive LLM is not initialized!"
        if self.budget_form == "token":
            sampling_params = {
                "max_tokens": self.internal_budget,
                "temperature": 1,
                "top_p": 1,
            }
            text, token_num = self.generate(
                self.llm1, self.model1, messages, sampling_params
            )
        else:
            sampling_params = {"temperature": 1, "top_p": 1, "max_tokens": 8192}
            text, token_num = self.start_reactive_stream(
                self.llm1,
                self.model1,
                messages,
                sampling_params,
                self.internal_budget,
            )
            time.sleep(budget - self.internal_budget)
        if "<think>" in text and "</think>" not in text:
            text += "</think>"
        if "oxed" in text.split("</think>")[-1]:
            return text, token_num
        ### s1 budget forcing ###
        text += "\nTherefore, the final answer is \\boxed{"
        max_attempt = 3
        while max_attempt > 0:
            max_attempt -= 1
            try:
                response = self.llm1.chat.completions.create(
                    model=self.model1,
                    messages=messages + [{"role": "assistant", "content": text}],
                    max_tokens=1,
                    temperature=0,
                    top_p=1,
                )
                if (
                    response.choices[0].message.content.strip()[0]
                    in self.prompts.ALL_ACTIONS
                ):
                    text += response.choices[0].message.content.strip()[0] + "}"
                    break
            except Exception:
                time.sleep(0.2)
            if max_attempt == 0:
                text += self.prompts.DEFAULT_ACTION + "}"
        return text, token_num

    def planning_inference(self, messages, budget, game_turn):
        assert self.model2 is not None, "Planning LLM is not initialized!"
        token_num = 0
        if self.budget_form == "token":
            if messages != []:
                self.gen_turn = game_turn
                self.gen_accum = -self.internal_budget
                sampling_params = {
                    "max_tokens": 80000,
                    "temperature": 0.6,
                    "top_p": 0.95,
                }
                self.gen_text, self.gen_token_num = self.generate(
                    self.llm2, self.model2, messages, sampling_params
                )
                if self.tokenizer is not None:
                    self.gen_token = self.tokenizer.encode(self.gen_text)
                token_num = self.gen_token_num
            self.gen_accum += budget
            can_flush = (
                self.gen_accum >= self.gen_token_num or self.tokenizer is not None
            )
            if can_flush:
                self.to_flush_turn = self.gen_turn
                if self.gen_accum >= self.gen_token_num:
                    self.to_flush = self.gen_text
                elif self.tokenizer is not None:
                    self.to_flush = self.tokenizer.decode(
                        self.gen_token[: self.gen_accum],
                        skip_special_tokens=True,
                    )
            text = self.to_flush
            turn = self.to_flush_turn
            self.to_flush = ""
            # check for completion: set the system in idle state
            if self.gen_accum + self.internal_budget >= self.gen_token_num:
                self.gen_text = ""
        else:
            sampling_params = {"temperature": 0.6, "top_p": 0.95}
            if messages != []:
                self.gen_turn = game_turn
                self.start_planning_stream(
                    self.llm2, self.model2, messages, sampling_params
                )
            time.sleep(budget - self.internal_budget)
            new_text, token_num = self.get_planning_chunks()
            self.gen_text += new_text
            text = self.gen_text
            turn = self.gen_turn
            if self.is_planning_finished():
                self.gen_text = ""
        return text, token_num, turn


def extract_boxed(text, default_value=""):
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r"oxed{"
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
        if text[i] == "{":
            stack.append("{")
        elif text[i] == "}":
            if stack:
                stack.pop()
            if not stack:
                return text[start_index + 1 : i].strip()
    return default_value if default_value else text[start_index + 1 :].strip()
