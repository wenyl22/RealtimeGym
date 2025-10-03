from importlib import import_module
from agent.reactive import ReactiveAgent
from agent.planning import PlanningAgent
from agent.agile import AgileThinker

def average(lst):
    return sum(lst) / len(lst) if lst else 0

def game_loop(file, raw_seed, args):
    env_m = import_module('environment.' + args.game)
    env, seed = env_m.setup_env(raw_seed, args.cognitive_load)
    agent = None
    params = {
        "prompts": import_module('environment.prompts.' + args.game),
        "file": file,
        "budget_form": args.budget_format,
        "port": args.port,
        "api_key": args.api_key,
        "internal_budget": args.internal_budget,
        "model1": args.model1 if args.system != "planning" else None,
        "model2": args.model2 if args.system != "reactive" else None,
    }
    if args.system == "reactive":
        agent = ReactiveAgent(**params)
    elif args.system == "planning":
        agent = PlanningAgent(**params)
    elif args.system == "agile":
        agent = AgileThinker(**params)
    else:
        raise NotImplementedError("System not recognized.")
    while env.terminal == False:
        observation = env.observe()
        agent.think(observation, args.time_pressure)
        action = agent.act()
        reward, reset = env.act(action)
        env.summary()
        agent.log(reward, reset)
    ret = { 'seed': seed, 'reward': env.reward, }
    del env
    return ret