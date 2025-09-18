import pandas as pd
from utils.client_utils import LLMClient
from utils.extract_utils import extract_boxed
from vllm import SamplingParams
import time
import re
from importlib import import_module
from collections import defaultdict
def average(lst):
    return sum(lst) / len(lst) if lst else 0
def meta_controller(args, client, env):
    """
    Decides whether to trigger slow agent based on:
    """
    if args.method == "fast":
        return False
    if args.meta_control == "continuous":
        return client.gen_text == ""
    elif args.meta_control.startswith("periodic"):
        f = int(args.meta_control[8:])
        return env.env.game_turn % f == 0
    elif args.meta_control == "triggered":
        return client.run_slow_trigger()
    elif args.meta_control == "event":
        return env.env.has_event()
        

def main_game_loop(file, seed, args):
    env_m = import_module('envs.' + args.game)
    prompt_m = import_module('envs.prompts.' + args.game)
    FORMAT = prompt_m.ACTION_FORMAT_PROMPT if args.method == "slow" else prompt_m.CONCLUSION_FORMAT_PROMPT
    client = LLMClient(args, args.api_keys)

    env, real_seed = env_m.setup_env(seed, args.difficulty)
    plan = ""

    start_time = time.time()
    logs = defaultdict(list)

    while env.env.terminal == False and env.env.game_turn <= 8:
        logs['render'].append('\n' + env.env.state_string())
        state_for_llm = env_m.llm_state_builder(env.env)
        state_description = env_m.state_to_description(state_for_llm, fast = False)
        fast_agent_response, slow_agent_response = "", ""
        fast_agent_prompt, slow_agent_prompt = "", ""
        fast_response_token_num, slow_response_token_num = 0, 0
        ### --- Slow Agent --- ###
        meta_control = meta_controller(args, client, env)
        if meta_control:
            messages = [ {"role": "user", "content": prompt_m.SLOW_AGENT_PROMPT + FORMAT + state_description} ]
            slow_agent_prompt = messages[-1]['content']
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        slow_agent_response, turns, slow_response_token_num = client.run_slow_inference(messages, sampling_params, env.env.game_turn)
        ## --- Update Plan --- ###
        if args.method == "slow":
            temp = extract_boxed(slow_agent_response)
            if temp != "":
                plan = re.sub(r'[^' + prompt_m.ALL_ACTIONS + ']', '', temp)
                if args.game != 'overcooked':
                    plan = plan[env.env.game_turn - turns:] if len(plan) > env.env.game_turn - turns else ""
        else:
            plan = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {turns} \)\n"""
            plan += slow_agent_response
        logs['plan'].append(plan)
        ### --- Fast Agent --- ###
        if args.method == "slow":
            action = plan[0] if plan != "" else prompt_m.DEFAULT_ACTION
            plan = plan[1:] if plan != "" else ""
        else:
            state_description = env_m.state_to_description(state_for_llm, plan if plan != "" else None, fast = True)
            messages = [ {"role": "user", "content": prompt_m.FAST_AGENT_PROMPT + state_description} ]
            fast_agent_prompt = messages[-1]['content']
            sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=args.fast_max_token)
            fast_agent_response, fast_response_token_num = client.run_fast_inference(messages, sampling_params, prompt_m.ALL_ACTIONS, prompt_m.DEFAULT_ACTION)
            action = extract_boxed(fast_agent_response)
        ### --- Act in Environment --- ###
        env.act(action)
        ### --- Log Information --- ###
        logs['description'].append(state_description)
        logs['action'].append(action)
        logs['reward'].append(env.env.reward)
        logs['meta_control'].append(meta_control)
        logs['slow_agent_prompt'].append(slow_agent_prompt)
        logs['slow_agent_response'].append(slow_agent_response)
        logs['fast_agent_prompt'].append(fast_agent_prompt)
        logs['fast_agent_response'].append(fast_agent_response)
        logs['slow_response_token_num'].append(slow_response_token_num)
        logs['fast_response_token_num'].append(fast_response_token_num)
        df = pd.DataFrame(logs)
        df.to_csv(file)

        if env_m.summarize(seed, args.difficulty, env):
            plan = ""
            while client.gen_text != "":
                client.run_slow_inference([], None, None)
            client.to_flush = ""
    df = pd.DataFrame(logs)
    df.to_csv(file)
    ret = {
        'seed': real_seed,
        'reward': env.env.reward,
        'game_time': time.time() - start_time,
    }
    del env
    return ret