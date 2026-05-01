"""
WEEK 6 — Comprehensive Evaluation
Day 1: Sharpe Ratio plot
Day 2: Max Drawdown chart
Day 3: Win Rate + P&L distribution
Day 4: Full history saved to JSON
Day 5: 4-panel performance dashboard
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_forex_data
from utils.metrics import full_report, save_history, max_drawdown, sharpe_ratio
from env.forex_env_pro_4 import ForexEnvWeek4
from agent.dqn_agent import DQNAgent

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/trained_agent_week4.pkl")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "../visualize")

df    = load_forex_data(DATA_PATH)
env   = ForexEnvWeek4(df)
agent = DQNAgent(state_size=9, action_size=3)

with open(MODEL_PATH, "rb") as f:
    agent.model.set_weights(pickle.load(f)["weights"])
agent.epsilon = 0.0

# ── Evaluate ──────────────────────────────────────────────────────────
obs, _ = env.reset(); done = False
action_log = {"hold": 0, "buy": 0, "sell": 0}
names = ["hold", "buy", "sell"]
while not done:
    action = agent.act(obs)
    obs, _, term, trunc, _ = env.step(action)
    done = term or trunc
    action_log[names[action]] += 1

# ── Full report (Week 6) ──────────────────────────────────────────────
report = full_report(env, label="Week 4 Agent")
save_history(env, OUT_DIR, label="week4")

total = sum(action_log.values())
print(f"\n  Hold: {action_log['hold']} ({100*action_log['hold']/total:.0f}%)  "
      f"Buy: {action_log['buy']} ({100*action_log['buy']/total:.0f}%)  "
      f"Sell: {action_log['sell']} ({100*action_log['sell']/total:.0f}%)")

# ── 4-panel dashboard ─────────────────────────────────────────────────
profits = [t["profit"] if isinstance(t, dict) else t for t in env.trade_history]
bal     = env.balance_history

rolling_sharpe = [
    sharpe_ratio(profits[max(0,i-20):i])
    for i in range(20, len(profits)+1)
]

bal_arr = np.array(bal)
peak    = np.maximum.accumulate(bal_arr)
dd_pct  = (peak - bal_arr) / np.where(peak > 0, peak, 1) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Week 6 — Performance Dashboard", fontsize=13)

# 1. Balance
ax = axes[0,0]
ax.plot(bal, color="#378ADD", linewidth=1.2)
ax.axhline(env.initial_balance, color="#888780", linestyle="--", linewidth=0.8)
ax.fill_between(range(len(bal)), bal, env.initial_balance,
    where=[b >= env.initial_balance for b in bal], alpha=0.2, color="#1D9E75")
ax.fill_between(range(len(bal)), bal, env.initial_balance,
    where=[b < env.initial_balance for b in bal], alpha=0.2, color="#E24B4A")
ax.set_title("Balance ($)"); ax.grid(alpha=0.3)

# 2. Drawdown
ax = axes[0,1]
ax.fill_between(range(len(dd_pct)), dd_pct, color="#E24B4A", alpha=0.4)
ax.plot(dd_pct, color="#A32D2D", linewidth=1)
ax.set_title(f"Drawdown (%) — max {dd_pct.max():.1f}%"); ax.grid(alpha=0.3)

# 3. Trade P&L distribution
ax = axes[1,0]
wins   = [p for p in profits if p > 0]
losses = [p for p in profits if p <= 0]
ax.hist(wins,   bins=20, color="#1D9E75", alpha=0.7, label=f"Wins ({len(wins)})")
ax.hist(losses, bins=20, color="#E24B4A", alpha=0.7, label=f"Losses ({len(losses)})")
ax.axvline(0, color="#888780", linestyle="--", linewidth=0.8)
ax.set_title("Trade P&L Distribution"); ax.legend(); ax.grid(alpha=0.3)

# 4. Rolling Sharpe
ax = axes[1,1]
ax.plot(rolling_sharpe, color="#378ADD", linewidth=1.2)
ax.axhline(1.0, color="#1D9E75", linestyle="--", linewidth=0.8, label="Sharpe=1 (good)")
ax.axhline(0.0, color="#888780", linestyle="--", linewidth=0.8)
ax.set_title("Rolling Sharpe (20-trade window)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "result_week6.png")
plt.savefig(out_path, dpi=150); plt.show()
print(f"\n  Dashboard → {out_path}")