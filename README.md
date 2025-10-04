### Real-time Reasoning Gym
Installation:
```bash
pip install -e .
```


Example Usage: 

If `budget_format == time`, then  `time_pressure` and `internal_budget` are physical times (unit: second).
If `budget_format == token`, then `time_pressure` and `internal_budget` are token numbers.

```bash
python run_game.py --api_key DEEPSEEK_API_KEY \
    --port https://api.deepseek.com/v3.1_terminus_expires_on_20251015 \
    --model2 deepseek-reasoner --model1 deepseek-chat \
    --budget_format [token/time] \
    --game [freeway/snake/overcooked] --cognitive_load [E/M/H] --time_pressure 8192 \
    --system [planning/reactive/agile] --internal_budget 4096 \
    --log_dir logs --seed_num [1-8]
```