"""
WEEK 7 — Multi-Pair Training + Out-of-Sample Test
Day 1-2: Load multiple pairs
Day 3: Training with random pair per episode
Day 4: 80/20 split, test on unseen data (out-of-sample)
Day 5: Compare single pair vs multi pair stats
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
from utils.data_loader import load_forex_data
from utils.metrics import full_report
from agent.dqn_agent import DQNAgent
from env.forex_env_pro_4 import ForexEnvWeek4
from env.multi_pair_env import MultiPairEnv, load_multi_pair

DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
EPISODES  = 500

# ── Load pairs ────────────────────────────────────────────────────────
pair_dfs = load_multi_pair(DATA_DIR, pairs=["XAUUSD"])
# Add more pairs if you have the CSV files:
# pair_dfs = load_multi_pair(DATA_DIR, pairs=["XAUUSD", "EURUSD", "GBPUSD"])

# ── DAY 4: 80/20 Train / Test split ──────────────────────────────────
train_dfs, test_dfs = {}, {}
for name, df in pair_dfs.items():
    split = int(len(df) * 0.8)
    train_dfs[name] = df.iloc[:split].reset_index(drop=True)
    test_dfs[name]  = df.iloc[split:].reset_index(drop=True)
    print(f"  {name}: train={len(train_dfs[name])} | test={len(test_dfs[name])}")

# ── Train ─────────────────────────────────────────────────────────────
train_env = MultiPairEnv(train_dfs)
agent = DQNAgent(
    state_size      = train_env.observation_space.shape[0],
    action_size     = 3,
    epsilon         = 1.0,
    epsilon_min     = 0.05,
    epsilon_decay   = 0.98,
    buffer_capacity = 30_000,
    batch_size      = 64,
    lr              = 5e-4,
)

print("=" * 55)
print(f"  WEEK 7 — Multi-Pair Training")
print(f"  Pairs : {list(train_dfs.keys())}")
print(f"  Ep    : {EPISODES}")
print("=" * 55)

rewards_history = []
for ep in range(1, EPISODES + 1):
    obs, _ = train_env.reset()
    total_r, steps, done = 0.0, 0, False
    while not done:
        action = agent.act(obs)
        next_obs, reward, term, trunc, _ = train_env.step(action)
        done = term or trunc
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs; total_r += reward; steps += 1
        if steps % 4 == 0: agent.train()
    agent.decay_epsilon()
    rewards_history.append(float(total_r))

    if ep % 50 == 0 or ep == 1:
        avg = float(np.mean(rewards_history[-25:]))
        print(f"  Ep {ep:4d}/{EPISODES}"
              f"  |  Reward: {total_r:+8.2f}"
              f"  |  Avg: {avg:+8.2f}"
              f"  |  Eps: {agent.epsilon:.3f}"
              f"  |  Pair: {train_env.active_pair}")

print("\n  Training selesai!")

# ── DAY 4: Out-of-sample test ─────────────────────────────────────────
print("\n  ── Out-of-sample evaluation ────────────────────────")
agent.epsilon = 0.0

for name, test_df in test_dfs.items():
    if len(test_df) < 30:
        print(f"  {name}: data test terlalu sedikit, skip"); continue
    test_env = ForexEnvWeek4(test_df)
    obs, _ = test_env.reset(); done = False
    while not done:
        action = agent.act(obs)
        obs, _, term, trunc, _ = test_env.step(action)
        done = term or trunc
    full_report(test_env, label=f"{name} out-of-sample (20% data)")

# ── Save ──────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(f"{out}/models", exist_ok=True)
os.makedirs(f"{out}/visualize", exist_ok=True)

with open(f"{out}/models/trained_agent_week7.pkl", "wb") as f:
    pickle.dump({"weights": agent.model.get_weights()}, f)

with open(f"{out}/visualize/rewards_week7.json", "w") as f:
    json.dump(rewards_history, f)

print(f"\n  Model → models/trained_agent_week7.pkl")

# ── DAY 5: Per-pair stats ─────────────────────────────────────────────
print("\n  ── Episode distribution per pair ───────────────────")
for pair, s in train_env.get_pair_stats().items():
    print(f"  {pair}: {s}")