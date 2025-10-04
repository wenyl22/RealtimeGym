
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from realtimegym.agents.reactive import ReactiveAgent
from realtimegym.agents.planning import PlanningAgent
from realtimegym.agents.agile import AgileThinker
import pygame
from PIL import Image
def check_args(args):
    if args.system == "planning":
        assert args.internal_budget == 0, "Internal budget must be 0 when system is planning."
    assert args.internal_budget <= args.time_pressure, "Internal budget must be less than or equal to time pressure when method is fast."

def game_loop(file, raw_seed, args):
    env_m = import_module('realtimegym.environments.' + args.game)
    env, seed, render = env_m.setup_env(raw_seed, args.cognitive_load, args.save_trajectory_gifs)
    agent = None
    params = {
        "prompts": import_module('realtimegym.environments.prompts.' + args.game),
        "file": file,
        "budget_form": args.budget_format,
        "port": args.port,
        "api_key": args.api_key,
        "internal_budget": args.internal_budget,
        "model1": args.model1 if args.system != "planning" else None,
        "model2": args.model2 if args.system != "reactive" else None,
        "skip_action": True,
    }
    if args.game == "overcooked":
        params["skip_action"] = False
    if args.system == "reactive":
        agent = ReactiveAgent(**params)
    elif args.system == "planning":
        agent = PlanningAgent(**params)
    elif args.system == "agile":
        agent = AgileThinker(**params)
    else:
        raise NotImplementedError("System not recognized.")
    surfaces = []
    if render is not None:
        surfaces.append(render.render(env))
    while env.terminal == False:
        observation = env.observe()
        agent.think(observation, args.time_pressure)
        action = agent.act()
        reward, reset = env.act(action)
        env.summary()
        agent.log(reward, reset)
        if render is not None:
            surfaces.append(render.render(env))
    if render is not None:
        gif_path = file.replace('.csv', '.gif')
        images = [pygame.surfarray.array3d(surface) for surface in surfaces]
        pil_images = [Image.fromarray(images[i].swapaxes(0, 1)) for i in range(len(images))]
        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=1000, loop=0)
    ret = { 'seed': seed, 'reward': env.reward, }
    del env
    return ret

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Real-time reasoning gym configurations.')
    args.add_argument('--api_key', type=str, default='')
    args.add_argument('--port', type=str, default='https://api.deepseek.com')
    args.add_argument('--model1', type=str, default = 'deepseek-chat')
    args.add_argument('--model2', type=str, default = 'deepseek-reasoner')    
    args.add_argument('--game', type=str, choices=['freeway', 'snake', 'overcooked'])
    args.add_argument('--budget_format', type=str, choices=['token', 'time'], default='token')
    args.add_argument('--time_pressure', type=int, default=8192)
    args.add_argument('--cognitive_load', type=str, choices=['E', 'M', 'H'])
    args.add_argument('--system', type=str, choices=['agile', 'reactive', 'planning'])
    args.add_argument('--internal_budget', type=int, default=8192)
    args.add_argument('--log_dir', type=str, default='logs')
    args.add_argument('--seed_num', type=int, default=1)
    args.add_argument('--repeat_times', type=int, default=1)
    args.add_argument('--save_trajectory_gifs', action='store_true', default=False)
    args = args.parse_args()
    log_dir = f"{args.log_dir}/{args.game}_{args.cognitive_load}_{args.time_pressure}_{args.system}_{args.internal_budget}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    check_args(args)
    instance = []
    for seed in range(args.seed_num):
        for r in range(args.repeat_times):        
            instance.append((log_dir + f'/{r}_{seed}.csv', seed, args))
    with open(log_dir + '/args.log', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
    with ThreadPoolExecutor(max_workers=len(instance)) as executor:
        futures = [
            executor.submit(game_loop, log_file, seed, args) for (log_file, seed, args) in instance
        ]
        results = []
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            with open(f'{log_dir}/args.log', 'a') as f:
                for key, value in result.items():
                    f.write(f"{key}: {value} ")
                f.write("\n---------------------------------------\n")
            print(f"Progress: {idx}/{total} ({idx/total*100:.2f}%)")
    with open(f'{log_dir}/args.log', 'a') as f:
        f.write("\nSummary:\n")
        for key in results[0].keys():
            avg_value = sum(result[key] for result in results) / len(results)
            f.write(f"Average {key}: {avg_value}\n")
