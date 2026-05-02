"""
=============================================================
WEEK 8 — DAY 2: 4-Panel Performance Dashboard
=============================================================
Visualisasi lengkap:
  Panel 1: Balance growth + drawdown overlay
  Panel 2: Trade P&L distribution (win vs loss histogram)
  Panel 3: Rolling Sharpe Ratio (20-trade window)
  Panel 4: Reward per episode + moving average (training curve)
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from utils.data_loader import load_forex_data
from utils.metrics import (
    sharpe_ratio, max_drawdown, win_rate,
    profit_factor, full_report
)
from agent.dqn_agent import DQNAgent
from env.forex_env_pro_4 import ForexEnvWeek4

# ── Paths ─────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, "data", "Data_historis(23-26).csv")
MODEL_PATH = os.path.join(BASE, "models", "trained_agent_week4.pkl")
REWARDS_PATH = os.path.join(BASE, "visualize", "rewards_history.json")
OUT_PATH   = os.path.join(BASE, "visualize", "week8_dashboard.png")

# ── Try loading model, fallback to best available ─────────────────────
def find_model(base):
    for name in ["trained_agent_week4.pkl", "trained_agent_week7.pkl",
                 "trained_agent.pkl"]:
        p = os.path.join(base, "models", name)
        if os.path.exists(p):
            return p
    return None

model_path = find_model(BASE)
if model_path is None:
    raise FileNotFoundError("Tidak ada model ditemukan di folder models/")

print(f"[OK] Loading model: {os.path.basename(model_path)}")

df    = load_forex_data(DATA_PATH)
env   = ForexEnvWeek4(df)
agent = DQNAgent(state_size=9, action_size=3)

with open(model_path, "rb") as f:
    saved = pickle.load(f)
agent.model.set_weights(saved["weights"])
agent.epsilon = 0.0

# ── Run evaluation ────────────────────────────────────────────────────
obs, _ = env.reset()
done   = False
action_log = [0, 0, 0]   # hold, buy, sell

while not done:
    action = agent.act(obs)
    obs, _, term, trunc, _ = env.step(action)
    done = term or trunc
    action_log[action] += 1

# ── Full report ────────────────────────────────────────────────────────
report = full_report(env, label="Week 4 DQN Agent")

# ── Prepare data ───────────────────────────────────────────────────────
profits_raw = env.trade_history
profits = [t["profit"] if isinstance(t, dict) else t for t in profits_raw]
bal     = np.array(env.balance_history)

# Drawdown series
peak    = np.maximum.accumulate(bal)
dd_pct  = (peak - bal) / np.where(peak > 0, peak, 1) * 100

# Rolling Sharpe
window = 20
roll_sharpe = [
    sharpe_ratio(profits[max(0, i-window):i])
    for i in range(1, len(profits) + 1)
]

# Training rewards
rewards_hist = []
if os.path.exists(REWARDS_PATH):
    with open(REWARDS_PATH) as f:
        rewards_hist = json.load(f)

# ── Colors ────────────────────────────────────────────────────────────
C_BLUE   = "#378ADD"
C_GREEN  = "#1D9E75"
C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_ORANGE = "#E8894A"

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F8F8F7")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.96, top=0.90, bottom=0.08)

title = (f"RL Forex AI — Final Performance Dashboard  |  "
         f"Return: {report['total_return']}  |  "
         f"Win Rate: {report['win_rate']}  |  "
         f"Sharpe: {report['sharpe_ratio']}  |  "
         f"Max DD: {report['max_drawdown']}")
fig.suptitle(title, fontsize=11, fontweight='bold', color="#333331", y=0.96)

# ─────────────────────────────────────────────────────────────────────
# PANEL 1: Balance + Drawdown overlay
# ─────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1_dd = ax1.twinx()

ax1.plot(bal, color=C_BLUE, linewidth=1.4, zorder=3, label="Balance ($)")
ax1.axhline(env.initial_balance, color=C_GRAY, linestyle="--",
            linewidth=0.8, label=f"Initial ${env.initial_balance:.0f}")
ax1.fill_between(range(len(bal)), bal, env.initial_balance,
    where=bal >= env.initial_balance, alpha=0.15, color=C_GREEN)
ax1.fill_between(range(len(bal)), bal, env.initial_balance,
    where=bal < env.initial_balance, alpha=0.15, color=C_RED)

ax1_dd.fill_between(range(len(dd_pct)), dd_pct, alpha=0.18,
                    color=C_ORANGE, label="Drawdown %")
ax1_dd.set_ylabel("Drawdown (%)", fontsize=9, color=C_ORANGE)
ax1_dd.tick_params(axis='y', colors=C_ORANGE, labelsize=8)
ax1_dd.set_ylim(bottom=0)

ax1.set_title("Balance & Drawdown", fontsize=10, fontweight='bold')
ax1.set_xlabel("Timestep", fontsize=9)
ax1.set_ylabel("Balance ($)", fontsize=9)
ax1.tick_params(labelsize=8)
ax1.set_facecolor("#F0F0EF")
ax1.grid(True, alpha=0.25, color='white')
handles = [
    Patch(color=C_BLUE, label="Balance"),
    Patch(color=C_GREEN, alpha=0.5, label="Profit zone"),
    Patch(color=C_ORANGE, alpha=0.5, label="Drawdown"),
]
ax1.legend(handles=handles, fontsize=8, loc="upper left")

# ─────────────────────────────────────────────────────────────────────
# PANEL 2: P&L Distribution
# ─────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

wins   = [p for p in profits if p > 0]
losses = [p for p in profits if p <= 0]

if wins:
    ax2.hist(wins, bins=25, color=C_GREEN, alpha=0.75,
             label=f"Wins ({len(wins)})  avg=${np.mean(wins):.2f}", edgecolor='white', linewidth=0.3)
if losses:
    ax2.hist(losses, bins=25, color=C_RED, alpha=0.75,
             label=f"Loss ({len(losses)})  avg=${np.mean(losses):.2f}", edgecolor='white', linewidth=0.3)

ax2.axvline(0, color=C_GRAY, linestyle="--", linewidth=1.0)
pf = profit_factor(profits)
ax2.text(0.97, 0.96, f"Profit Factor: {pf:.2f}", transform=ax2.transAxes,
         ha='right', va='top', fontsize=9, color="#333331",
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax2.set_title("Trade P&L Distribution", fontsize=10, fontweight='bold')
ax2.set_xlabel("Profit per Trade ($)", fontsize=9)
ax2.set_ylabel("Frequency", fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_facecolor("#F0F0EF")
ax2.grid(True, alpha=0.25, color='white')
ax2.legend(fontsize=8)

# ─────────────────────────────────────────────────────────────────────
# PANEL 3: Rolling Sharpe
# ─────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

x_sharpe = range(1, len(roll_sharpe) + 1)
ax3.plot(x_sharpe, roll_sharpe, color=C_BLUE, linewidth=1.2, alpha=0.8)
ax3.fill_between(x_sharpe, roll_sharpe, 0,
    where=[s >= 0 for s in roll_sharpe], alpha=0.15, color=C_GREEN)
ax3.fill_between(x_sharpe, roll_sharpe, 0,
    where=[s < 0 for s in roll_sharpe], alpha=0.15, color=C_RED)
ax3.axhline(1.0, color=C_GREEN, linestyle="--", linewidth=0.9,
            label="Sharpe=1 (good)")
ax3.axhline(2.0, color=C_GREEN, linestyle=":", linewidth=0.9,
            label="Sharpe=2 (excellent)")
ax3.axhline(0.0, color=C_GRAY, linestyle="-", linewidth=0.5)

final_sr = float(report['sharpe_ratio'])
ax3.text(0.97, 0.96, f"Final Sharpe: {final_sr:.2f}",
         transform=ax3.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax3.set_title(f"Rolling Sharpe Ratio (window={window} trades)",
              fontsize=10, fontweight='bold')
ax3.set_xlabel("Trade #", fontsize=9)
ax3.set_ylabel("Sharpe Ratio", fontsize=9)
ax3.tick_params(labelsize=8)
ax3.set_facecolor("#F0F0EF")
ax3.grid(True, alpha=0.25, color='white')
ax3.legend(fontsize=8)

# ─────────────────────────────────────────────────────────────────────
# PANEL 4: Training Curve
# ─────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

if rewards_hist:
    rw = np.array(rewards_hist)
    ma_w = 15
    ma_rw = [np.mean(rw[max(0, i-ma_w):i+1]) for i in range(len(rw))]
    eps_x = range(len(rw))

    colors_bar = [C_GREEN if r >= 0 else C_RED for r in rw]
    ax4.bar(eps_x, rw, color=colors_bar, alpha=0.35, width=1.0)
    ax4.plot(eps_x, ma_rw, color=C_BLUE, linewidth=2.0,
             label=f"Moving Avg ({ma_w} ep)")
    ax4.axhline(0, color=C_GRAY, linestyle="--", linewidth=0.7)

    # Mark breakeven episode
    for i, v in enumerate(ma_rw):
        if v > 0:
            ax4.axvline(i, color=C_GREEN, linestyle=":", linewidth=0.8,
                        alpha=0.6, label=f"Break-even ep.{i}")
            break

    ax4.set_xlim(0, len(rw) - 1)
else:
    ax4.text(0.5, 0.5, "No training history found\nRun train/train_ppo.py first",
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=10, color=C_GRAY)

ax4.set_title("Training Reward Curve", fontsize=10, fontweight='bold')
ax4.set_xlabel("Episode", fontsize=9)
ax4.set_ylabel("Total Reward ($)", fontsize=9)
ax4.tick_params(labelsize=8)
ax4.set_facecolor("#F0F0EF")
ax4.grid(True, alpha=0.25, color='white')
ax4.legend(fontsize=8)

# ── Stats annotation ──────────────────────────────────────────────────
total_acts = sum(action_log)
ann = (f"Actions — Hold: {action_log[0]/total_acts*100:.0f}%  "
       f"Buy: {action_log[1]/total_acts*100:.0f}%  "
       f"Sell: {action_log[2]/total_acts*100:.0f}%  |  "
       f"Trades: {report['total_trades']}  |  "
       f"Final Balance: {report['final_balance']}")
fig.text(0.5, 0.01, ann, ha='center', fontsize=9, color="#555553",
         style='italic')

plt.savefig(OUT_PATH, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n  Dashboard saved → {OUT_PATH}")
plt.show()
