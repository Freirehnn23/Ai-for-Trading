import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from env.forex_env_pro import ForexEnv
from agent.dqn_agent import DQNAgent
from utils.data_loader import load_forex_data

# ── Load data & env ───────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
df        = load_forex_data(DATA_PATH)
env       = ForexEnv(df)

# ── Load model ────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_agent.pkl")
agent      = DQNAgent(state_size=2, action_size=3)

with open(model_path, "rb") as f:
    saved = pickle.load(f)
agent.model.set_weights(saved["weights"])
agent.epsilon = 0.0   # ← no exploration, pure exploitation

print("[OK] Model loaded")
print("=" * 55)
print("  EVALUASI AGENT (epsilon = 0)")
print("=" * 55)

# ── Jalankan 1 episode penuh ──────────────────────────────────────────
obs, _       = env.reset()
done         = False
balance_hist = [0.0]
action_log   = {"hold": 0, "buy": 0, "sell": 0}
action_names = ["hold", "buy", "sell"]

while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    action_log[action_names[action]] += 1
    balance_hist.append(balance_hist[-1] + reward)

# ── Hasil ─────────────────────────────────────────────────────────────
total = sum(action_log.values())
print(f"\n  Total steps      : {total}")
print(f"  Hold : {action_log['hold']:4d}  ({100*action_log['hold']/total:.1f}%)")
print(f"  Buy  : {action_log['buy']:4d}  ({100*action_log['buy']/total:.1f}%)")
print(f"  Sell : {action_log['sell']:4d}  ({100*action_log['sell']/total:.1f}%)")
print(f"\n  Profit kumulatif : {balance_hist[-1]:+.4f}")

if balance_hist[-1] > 0:
    print("  → PROFIT ✓")
else:
    print("  → LOSS  — agent perlu lebih banyak training")

# ── Load rewards history ──────────────────────────────────────────────
rewards_path = os.path.join(os.path.dirname(__file__), "../visualize/rewards_history.json")
with open(rewards_path) as f:
    rewards = json.load(f)

ma = [np.mean(rewards[max(0,i-10):i+1]) for i in range(len(rewards))]

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafik 1: Reward per episode
axes[0].plot(rewards, color="#aec6e8", alpha=0.5, linewidth=0.8, label="Reward")
axes[0].plot(ma, color="#2176ae", linewidth=2.0, label="Moving Avg (10)")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[0].set_title("Reward per Episode")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Grafik 2: Profit kumulatif saat eval
axes[1].plot(balance_hist, color="#2176ae", linewidth=1.5)
axes[1].fill_between(range(len(balance_hist)), balance_hist, 0,
    where=[v >= 0 for v in balance_hist], alpha=0.2, color="green", label="Profit")
axes[1].fill_between(range(len(balance_hist)), balance_hist, 0,
    where=[v < 0 for v in balance_hist], alpha=0.2, color="red", label="Loss")
axes[1].set_title("Profit Kumulatif (Eval)")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Profit")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "../visualize/result.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\n  Grafik disimpan → {out}")