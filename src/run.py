import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from game_loop import main_game_loop
def check_args(args):
    if args.method == "slow":
        assert args.fast_max_token == 0, "Fast max token must be 0 when method is slow."
    if args.method == "fast":
        assert args.fast_max_token == args.token_per_tick, "Fast max token must be equal to token per tick when method is fast." 
    if args.method == "parallel":
        assert args.fast_max_token <= args.token_per_tick, "Fast max token must be less than or equal to token per tick when method is parallel." 

def schedule(args):
    instance = []
    repeat_times = 1
    seed_num = 1
    setting = f"{args.game}_{args.difficulty}_{args.method}_{args.token_per_tick}_{args.fast_max_token}"
    log_dir = args.log_dir + '/' + setting
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    check_args(args)
    with open(log_dir + '/args.log', 'w') as f:
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
    for seed in range(seed_num):
        for r in range(repeat_times):
            instance.append((log_dir + f'/{r}_{seed}.csv', seed, args))
    with ThreadPoolExecutor(max_workers=len(instance)) as executor:
        futures = [
            executor.submit(
                main_game_loop, log_file, seed, args
            )
            for (log_file, seed, args) in instance
        ]
        results = []
        total = len(futures)
       # write to the correct log file
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            with open(f'{log_dir}/args.log', 'a') as f:
                for key, value in result.items():
                    f.write(f"{key}: {value} ")
                f.write("\n---------------------------------------\n")
            print(f"Progress: {idx}/{total} ({idx/total*100:.2f}%)")

if __name__ == "__main__":
    Args = argparse.ArgumentParser(description='Run benchmark with a specific model.')
    Args.add_argument('--api_keys', nargs='+', type=str, default=[], help='List of API keys for OpenAI')
    Args.add_argument('--slow_base_url', type=str, default='https://api.deepseek.com', help='URL of the slow model server')
    Args.add_argument('--fast_base_url', type=str, default='https://api.deepseek.com', help='URL of the fast model server')
    Args.add_argument('--slow_model', type=str, default = 'deepseek-reasoner')
    Args.add_argument('--fast_model', type=str, default = 'deepseek-chat')
    Args.add_argument('--meta_control', type=str, default='continuous', help='method to trigger slow agent')
    Args.add_argument('--game', type=str, choices=['freeway', 'snake', 'overcooked'], help='game to play')
    Args.add_argument('--difficulty', type=str, choices=['E', 'M', 'H'])
    Args.add_argument('--method', type=str, choices=['slow', 'fast', 'parallel'], help='agent system to use')
    Args.add_argument('--token_per_tick', type=int, default=8192)
    Args.add_argument('--fast_max_token', type=int, default=8192)
    Args.add_argument('--log_dir', type=str, default='logs')
    Args = Args.parse_args()
    schedule(Args)
