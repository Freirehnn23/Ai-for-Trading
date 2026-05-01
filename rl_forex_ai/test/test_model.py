import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_forex_data
from agent.dqn_agent import DQNAgent
from env.forex_env_pro_2 import ForexEnvWeek2

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/trained_agent.pkl")

df    = load_forex_data(DATA_PATH)
env   = ForexEnvWeek2(df)
agent = DQNAgent(state_size=4, action_size=3)

with open(MODEL_PATH, "rb") as f:
    agent.model.set_weights(pickle.load(f)["weights"])
agent.epsilon = 0.0

print("=" * 55)
print("  WEEK 2 — EVALUASI (epsilon = 0)")
print("=" * 55)

obs, _  = env.reset()
done    = False
actions = {"hold": 0, "buy": 0, "sell": 0}
names   = ["hold", "buy", "sell"]

while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    actions[names[action]] += 1

# ── DAY 5: Stats lengkap ──────────────────────────────────────────────
total = sum(actions.values())
print(f"\n  Total steps : {total}")
print(f"  Hold  : {actions['hold']:4d}  ({100*actions['hold']/total:.1f}%)")
print(f"  Buy   : {actions['buy']:4d}  ({100*actions['buy']/total:.1f}%)")
print(f"  Sell  : {actions['sell']:4d}  ({100*actions['sell']/total:.1f}%)")

print("\n  ── Trade Stats (Day 5 Debug) ──────────────────────")
stats = env.get_stats()
for k, v in stats.items():
    print(f"  {k:<20}: {v}")

# ── Plot ──────────────────────────────────────────────────────────────
rewards_path = os.path.join(os.path.dirname(__file__), "../visualize/rewards_history.json")
with open(rewards_path) as f:
    rewards = json.load(f)

ma = [np.mean(rewards[max(0,i-10):i+1]) for i in range(len(rewards))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rewards, color="#aec6e8", alpha=0.4, linewidth=0.8, label="Reward")
axes[0].plot(ma, color="#2176ae", linewidth=2.0, label="Moving Avg (10)")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[0].set_title("Reward per Episode — Week 2")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

bal = env.balance_history
axes[1].plot(bal, color="#2176ae", linewidth=1.5)
axes[1].axhline(env.initial_balance, color="gray", linestyle="--",
                linewidth=1, label=f"Modal awal ${env.initial_balance:.0f}")
axes[1].fill_between(range(len(bal)), bal, env.initial_balance,
    where=[b >= env.initial_balance for b in bal],
    alpha=0.2, color="green", label="Profit")
axes[1].fill_between(range(len(bal)), bal, env.initial_balance,
    where=[b < env.initial_balance for b in bal],
    alpha=0.2, color="red", label="Loss")
axes[1].set_title("Balance ($) — Week 2")
axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Balance ($)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "../visualize/result_week2.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\n  Grafik disimpan → {out}")