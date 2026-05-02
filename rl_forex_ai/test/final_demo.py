"""
=============================================================
WEEK 8 — DAY 5: Final Demo + Complete Test Suite
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, json, time
import numpy as np

from utils.data_loader import load_forex_data
from utils.metrics import (
    sharpe_ratio, max_drawdown, win_rate,
    profit_factor, calmar_ratio, full_report
)
from agent.dqn_agent import DQNAgent
from env.forex_env_pro_2 import ForexEnvWeek2
from env.forex_env_pro_3 import ForexEnvWeek3
from env.forex_env_pro_4 import ForexEnvWeek4

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "Data_historis(23-26).csv")
OUT_DIR   = os.path.join(BASE, "visualize")
MODEL_DIR = os.path.join(BASE, "models")

# ── Helper: cek state_size dari weight model ──────────────────────────
def get_model_state_size(model_path):
    """Baca dimensi input dari weight layer pertama."""
    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        weights = saved["weights"]
        # weights[0] = (w, b) untuk layer pertama
        # w.shape = (state_size, hidden_size)
        return weights[0][0].shape[0]
    except Exception:
        return None

def find_model_for_week(week):
    """
    Cari model yang state_size-nya cocok dengan week yang diminta.
    Tidak sembarang fallback ke model lain.
    """
    expected_size = {2: 4, 3: 5, 4: 9}.get(week)
    if expected_size is None:
        return None

    # Cari dari yang paling spesifik
    candidates = [
        os.path.join(MODEL_DIR, f"trained_agent_week{week}.pkl"),
        os.path.join(MODEL_DIR, f"trained_agent_week{week-1}.pkl"),
        os.path.join(MODEL_DIR, "trained_agent.pkl"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        sz = get_model_state_size(path)
        if sz == expected_size:
            return path

    return None   # tidak ada model yang cocok


def run_eval(env, agent, label):
    obs, _ = env.reset()
    done   = False
    action_log = [0, 0, 0]
    t0 = time.time()

    while not done:
        action = agent.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        action_log[action] += 1

    elapsed = time.time() - t0
    profits = [t["profit"] if isinstance(t, dict) else t
               for t in env.trade_history]
    bal     = env.balance_history
    total_steps = sum(action_log)

    dd  = max_drawdown(bal)
    wr  = win_rate(profits)
    pf  = profit_factor(profits)
    sr  = sharpe_ratio(profits)
    ret = (env.balance - env.initial_balance) / env.initial_balance
    cm  = calmar_ratio(ret, dd)

    return {
        "label"         : label,
        "final_balance" : env.balance,
        "total_return"  : ret,
        "total_trades"  : len(profits),
        "win_rate"      : wr,
        "profit_factor" : pf,
        "sharpe_ratio"  : sr,
        "max_drawdown"  : dd,
        "calmar_ratio"  : cm,
        "hold_pct"      : action_log[0] / total_steps * 100,
        "buy_pct"       : action_log[1] / total_steps * 100,
        "sell_pct"      : action_log[2] / total_steps * 100,
        "eval_time_s"   : elapsed,
    }


def load_agent(model_path, state_size):
    agent = DQNAgent(state_size=state_size, action_size=3)
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    agent.model.set_weights(saved["weights"])
    agent.epsilon = 0.0
    return agent


# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  🤖  RL FOREX AI — FINAL DEMO  (Week 8 Day 5)")
print("=" * 65)

df = load_forex_data(DATA_PATH)
print(f"  Data  : {DATA_PATH}")
print(f"  Output: {OUT_DIR}")
print("=" * 65)
print()

# ── Scan semua model yang tersedia ────────────────────────────────────
print("  Model scan:")
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl"):
        p = os.path.join(MODEL_DIR, fname)
        sz = get_model_state_size(p)
        week_tag = {4: "W2", 5: "W3", 9: "W4"}.get(sz, "?")
        print(f"    {fname:<35} state_size={sz}  [{week_tag}]")
print()

# ── Evaluasi per week ─────────────────────────────────────────────────
ENV_CONFIGS = [
    (2, ForexEnvWeek2, "Week 2 — Spread+Commission"),
    (3, ForexEnvWeek3, "Week 3 — +SL/TP/Risk"),
    (4, ForexEnvWeek4, "Week 4 — +RSI/MA/S&R  [FINAL]"),
]

results = []
for week, EnvClass, label in ENV_CONFIGS:
    model_path = find_model_for_week(week)
    if model_path is None:
        print(f"  [SKIP] {label}")
        print(f"         → Tidak ada model dengan state_size="
              f"{[4,5,9][[2,3,4].index(week)]}")
        print(f"           Jalankan: python train/train_ppo.py  (WEEK={week})")
        continue

    state_size = {2: 4, 3: 5, 4: 9}[week]
    env   = EnvClass(df)
    agent = load_agent(model_path, state_size)
    r     = run_eval(env, agent, label)
    results.append(r)
    print(f"  ✓ {label} ({r['eval_time_s']:.1f}s)")

if not results:
    print()
    print("  ⚠️  Tidak ada model yang cocok ditemukan.")
    print("  Training ulang dengan semua week:")
    print()
    print("    # Jalankan satu per satu:")
    for w in [2, 3, 4]:
        print(f"    # Edit train/train_ppo.py → WEEK={w}, lalu:")
        print(f"    python train/train_ppo.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  📊  PERBANDINGAN PERFORMA ANTAR WEEK")
print("=" * 65)
print(f"  {'Label':<34} {'Return':>8} {'Win%':>6} {'PF':>5} {'Sharpe':>7} {'MaxDD':>7}")
print("  " + "-" * 62)
for r in results:
    tag = "★ " if r == results[-1] else "  "
    print(f"{tag}{r['label']:<34}"
          f" {r['total_return']*100:>7.1f}%"
          f" {r['win_rate']*100:>5.1f}%"
          f" {r['profit_factor']:>5.2f}"
          f" {r['sharpe_ratio']:>7.2f}"
          f" {r['max_drawdown']*100:>6.1f}%")

# ─────────────────────────────────────────────────────────────────────
# FULL REPORT — best model
# ─────────────────────────────────────────────────────────────────────
best = results[-1]
print()
print("=" * 65)
print(f"  📋  FULL REPORT — {best['label']}")
print("=" * 65)
print(f"  Initial Balance : $1,000.00")
print(f"  Final Balance   : ${best['final_balance']:.2f}")
print(f"  Total Return    : {best['total_return']*100:.1f}%")
print(f"  Total Trades    : {best['total_trades']}")
print(f"  Win Rate        : {best['win_rate']*100:.1f}%")
print(f"  Profit Factor   : {best['profit_factor']:.2f}x")
print(f"  Sharpe Ratio    : {best['sharpe_ratio']:.2f}")
print(f"  Max Drawdown    : {best['max_drawdown']*100:.1f}%")
cm = best['calmar_ratio']
print(f"  Calmar Ratio    : {'inf' if cm == float('inf') else f'{cm:.2f}'}")
print()
print(f"  Action Mix:")
print(f"    Hold  : {best['hold_pct']:.1f}%")
print(f"    Buy   : {best['buy_pct']:.1f}%")
print(f"    Sell  : {best['sell_pct']:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# GRADING
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  🎓  PENILAIAN OTOMATIS")
print("=" * 65)

checks = [
    ("Win Rate > 50%",       best['win_rate'] > 0.50),
    ("Profit Factor > 1.0",  best['profit_factor'] > 1.0),
    ("Positive Return",      best['total_return'] > 0),
    ("Sharpe > 0",           best['sharpe_ratio'] > 0),
    ("Max Drawdown < 50%",   best['max_drawdown'] < 0.50),
    ("Trades > 50",          best['total_trades'] > 50),
    ("Mix: tidak 100% hold", best['hold_pct'] < 99.0),
]

passed = sum(ok for _, ok in checks)
for desc, ok in checks:
    print(f"  {'✅' if ok else '❌'}  {desc}")

grade_map = [(7,"EXCELLENT 🏆"),(6,"GOOD 🥇"),(5,"PASS ✓"),
             (4,"MARGINAL ⚠️"),(0,"FAIL ❌")]
grade = next(g for n,g in grade_map if passed >= n)
print()
print(f"  Score: {passed}/{len(checks)}  →  {grade}")

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
summary = {
    "results": [{k:v for k,v in r.items()} for r in results],
    "grade": grade, "score": f"{passed}/{len(checks)}"
}
out_json = os.path.join(OUT_DIR, "final_demo_report.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print()
print(f"  Report saved → {out_json}")
print("  Next: python visualize/week8_dashboard.py")
print("=" * 65)
