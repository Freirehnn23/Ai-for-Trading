"""
WEEK 5 — Hyperparameter Tuning
Day 1: learning_rate grid
Day 2: batch_size grid
Day 3: epsilon_decay grid (exploration)
Day 4: comparison logging
Day 5: print best config
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time
import numpy as np
from utils.data_loader import load_forex_data
from agent.dqn_agent import DQNAgent
from env.forex_env_pro_4 import ForexEnvWeek4

DATA_PATH     = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
TRAIN_EPISODES = 200   # shorter per config
EVAL_EPISODES  = 10

CONFIGS = [
    {"name": "baseline",   "lr": 5e-4, "batch": 64,  "decay": 0.98},
    {"name": "fast_lr",    "lr": 1e-3, "batch": 64,  "decay": 0.98},
    {"name": "slow_lr",    "lr": 1e-4, "batch": 64,  "decay": 0.98},
    {"name": "big_batch",  "lr": 5e-4, "batch": 128, "decay": 0.98},
    {"name": "slow_decay", "lr": 5e-4, "batch": 64,  "decay": 0.99},
    {"name": "fast_decay", "lr": 5e-4, "batch": 64,  "decay": 0.95},
]

df = load_forex_data(DATA_PATH)


def run_config(cfg):
    env   = ForexEnvWeek4(df)
    agent = DQNAgent(
        state_size=env.observation_space.shape[0], action_size=3,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=cfg["decay"],
        buffer_capacity=20_000, batch_size=cfg["batch"], lr=cfg["lr"],
    )
    rewards = []
    t0 = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        obs, _ = env.reset()
        total_r, steps, done = 0.0, 0, False
        while not done:
            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs; total_r += reward; steps += 1
            if steps % 4 == 0: agent.train()
        agent.decay_epsilon()
        rewards.append(total_r)

    # Eval without exploration
    agent.epsilon = 0.0
    eval_r = []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        total_r, done = 0.0, False
        while not done:
            action = agent.act(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_r += reward
        eval_r.append(total_r)

    return {
        "name"      : cfg["name"],
        "lr"        : cfg["lr"],
        "batch"     : cfg["batch"],
        "decay"     : cfg["decay"],
        "train_avg" : round(float(np.mean(rewards[-20:])), 2),
        "eval_avg"  : round(float(np.mean(eval_r)), 2),
        "eval_max"  : round(float(np.max(eval_r)), 2),
        "time_s"    : round(time.time() - t0, 1),
    }


print("=" * 70)
print("  WEEK 5 — Hyperparameter Comparison")
print("=" * 70)
print(f"  {'Config':<14} {'LR':>8} {'Batch':>6} {'Decay':>6} {'Train':>10} {'Eval Avg':>10}")
print("  " + "-" * 60)

results = []
for cfg in CONFIGS:
    print(f"  Running {cfg['name']}...", end="", flush=True)
    r = run_config(cfg)
    results.append(r)
    print(f"\r  {r['name']:<14} {r['lr']:>8.0e} {r['batch']:>6} {r['decay']:>6.2f}"
          f" {r['train_avg']:>10.0f} {r['eval_avg']:>10.0f}  ({r['time_s']}s)")

best = max(results, key=lambda x: x["eval_avg"])
print(f"\n  Best config : {best['name']}")
print(f"  Eval avg    : {best['eval_avg']:.2f}")
print(f"\n  Use in train_ppo.py:")
print(f"    lr={best['lr']}, batch_size={best['batch']}, epsilon_decay={best['decay']}")

out_dir = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(f"{out_dir}/visualize", exist_ok=True)
with open(f"{out_dir}/visualize/week5_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → visualize/week5_results.json")