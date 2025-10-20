import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from typing import Any

import pygame
from PIL import Image

from src.realtimegym.agents.agile import AgileThinker
from src.realtimegym.agents.planning import PlanningAgent
from src.realtimegym.agents.reactive import ReactiveAgent


def check_args(args):
    if args.system == "planning":
        assert (
            args.internal_budget == 0
        ), "Internal budget must be 0 when system is planning."
    assert (
        args.internal_budget <= args.time_pressure
    ), "Internal budget must be less than or equal to time pressure when method is fast."


def game_loop(file, raw_seed, args):
    env_m: Any = import_module("realtimegym.environments." + args.game)
    env, seed, render = env_m.setup_env(  # type: ignore
        raw_seed, args.cognitive_load, args.save_trajectory_gifs
    )
    agent = None
    params = {
        "prompts": import_module("realtimegym.environments.prompts." + args.game),
        "file": file,
        "budget_form": args.budget_format,
        "port1": args.port1,
        "port2": args.port2,
        "api_key": args.api_key,
        "internal_budget": args.internal_budget,
        "model1": args.model1 if args.system != "planning" else None,
        "model2": args.model2 if args.system != "reactive" else None,
        "skip_action": True,
    }
    if args.game == "overcooked":
        params["skip_action"] = False
    if args.system == "reactive":
        agent = ReactiveAgent(**params)  # type: ignore
    elif args.system == "planning":
        agent = PlanningAgent(**params)  # type: ignore
    elif args.system == "agile":
        agent = AgileThinker(**params)  # type: ignore
    else:
        raise NotImplementedError("System not recognized.")
    if args.checkpoint is not None:  # resume from checkpoint
        checkpoint_file = file.replace(args.log_dir, args.checkpoint)
        df = pd.read_csv(checkpoint_file)
        obs, done = env.reset()
        for a in df["action"]:
            obs, done, reward = env.step(a)
        if done:
            ret = {
                "seed": seed,
                "reward": env.reward,
                "total_time": 0,
                "log_dir": os.path.dirname(file),
            }
            return ret
        else:
            env, seed, render = env_m.setup_env(  # type: ignore
                raw_seed, args.cognitive_load, args.save_trajectory_gifs
            )
            agent.resume_from_checkpoint(env, checkpoint_file)
    start_time = time.time()
    surfaces = []

    # New API loop: obs, done = env.reset()
    obs, done = env.reset()
    if render is not None:
        surfaces.append(render.render(env))

    while not done:
        # agent.observe(obs)
        agent.observe(obs)
        # agent.think(timeout=T_E)
        agent.think(timeout=args.time_pressure)
        # action = agent.act() or DEFAULT_ACTION
        action = agent.act()
        if action is None:
            action = agent.prompts.DEFAULT_ACTION
        # obs, done, reward = env.step(action)
        obs, done, reward = env.step(action)
        env.summary()
        # Legacy log method still uses old signature
        reset_flag = hasattr(env, "_just_reset") and env._just_reset
        agent.log(reward, reset_flag)
        if render is not None:
            surfaces.append(render.render(env))
    if render is not None:
        gif_path = file.replace(".csv", ".gif")
        images = [pygame.surfarray.array3d(surface) for surface in surfaces]
        pil_images = [
            Image.fromarray(images[i].swapaxes(0, 1)) for i in range(len(images))
        ]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000,
            loop=0,
        )
    ret = {
        "seed": seed,
        "reward": env.reward,
        "total_time": time.time() - start_time,
        "log_dir": os.path.dirname(file),
    }
    del env
    return ret


def main():
    """Main CLI entry point for the Real-time Reasoning Gym."""
    args = argparse.ArgumentParser(
        description="Real-time reasoning gym configurations."
    )
    args.add_argument("--api_key", type=str, default="")
    args.add_argument("--port1", type=str, default="https://api.deepseek.com")
    args.add_argument("--port2", type=str, default="https://api.deepseek.com")
    args.add_argument("--model1", type=str, default="deepseek-chat")
    args.add_argument("--model2", type=str, default="deepseek-reasoner")
    args.add_argument(
        "--game",
        type=str,
        choices=["freeway", "snake", "overcooked"],
        default="freeway",
    )
    args.add_argument(
        "--budget_format", type=str, choices=["token", "time"], default="token"
    )
    args.add_argument("--time_pressure", type=int, default=8192)
    args.add_argument("--cognitive_load", type=str, choices=["E", "M", "H"])
    args.add_argument("--system", type=str, choices=["agile", "reactive", "planning"])
    args.add_argument("--internal_budget", type=int, default=8192)
    args.add_argument("--log_dir", type=str, default="logs")
    args.add_argument("--seed_num", type=int, default=1)
    args.add_argument("--repeat_times", type=int, default=1)
    args.add_argument("--save_trajectory_gifs", action="store_true", default=False)
    args.add_argument("--settings", type=str, nargs="+", default=[])
    args.add_argument("--instance_num", type=int, default=None)
    args.add_argument("--checkpoint", type=str, default=None)
    args = args.parse_args()
    if args.settings == []:
        args.settings = [
            f"{args.game}_{args.cognitive_load}_{args.time_pressure}_{args.system}_{args.internal_budget}"
        ]
    instance = []
    for setting in args.settings:
        new_args = argparse.Namespace(**vars(args))
        game, cognitive_load, time_pressure, system, internal_budget = setting.split(
            "_"
        )
        new_args.game = game
        new_args.cognitive_load = cognitive_load
        new_args.time_pressure = int(time_pressure)
        new_args.system = system
        new_args.internal_budget = int(internal_budget)
        check_args(new_args)
        log_dir = new_args.log_dir + f"/{setting}"
        for seed in range(args.seed_num):
            for r in range(args.repeat_times):
                if not os.path.exists(log_dir + f"/{r}_{seed}.csv"):
                    instance.append((log_dir + f"/{r}_{seed}.csv", seed, new_args))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(log_dir + "/args.log"):
            with open(log_dir + "/args.log", "w") as f:
                for arg, value in vars(new_args).items():
                    f.write(f"{arg}: {value}\n")
                f.write("\n")
    if args.instance_num is not None:
        assert args.instance_num == len(instance), "instance_num incorrect!"
    with ThreadPoolExecutor(max_workers=len(instance)) as executor:
        futures = [
            executor.submit(game_loop, log_file, seed, args)
            for (log_file, seed, args) in instance
        ]
        results = []
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            log_dir = result["log_dir"]
            # remove log_dir from result
            del result["log_dir"]
            results.append(result)
            with open(f"{log_dir}/args.log", "a") as f:
                for key, value in result.items():
                    f.write(f"{key}: {value} ")
                f.write("\n-----------------------------\n")
            print(f"Progress: {idx}/{total} ({idx / total * 100:.2f}%)")


if __name__ == "__main__":
    main()
