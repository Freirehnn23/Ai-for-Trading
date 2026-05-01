"""
WEEK 6 — Evaluation Metrics
Day 1: Sharpe Ratio
Day 2: Max Drawdown
Day 3: Win Rate + Profit Factor
Day 4: History saving
Day 5: Full performance report
"""
import numpy as np
import json
import os


def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    r = np.array(returns, dtype=float)
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    excess = r - risk_free
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def max_drawdown(balance_history):
    bal  = np.array(balance_history, dtype=float)
    peak = np.maximum.accumulate(bal)
    dd   = (peak - bal) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


def win_rate(trades):
    if not trades: return 0.0
    return len([t for t in trades if t > 0]) / len(trades)


def profit_factor(trades):
    wins   = [t for t in trades if t > 0]
    losses = [-t for t in trades if t < 0]
    if not losses: return float('inf')
    if not wins:   return 0.0
    return float(sum(wins) / sum(losses))


def calmar_ratio(total_return, max_dd):
    if max_dd == 0: return float('inf')
    return float(total_return / max_dd)


def full_report(env, label="Agent"):
    trades  = env.trade_history
    profits = [t["profit"] if isinstance(t, dict) else t for t in trades]
    bal     = env.balance_history

    dd        = max_drawdown(bal)
    wr        = win_rate(profits)
    pf        = profit_factor(profits)
    sr        = sharpe_ratio(profits)
    total_ret = (env.balance - env.initial_balance) / env.initial_balance
    calmar    = calmar_ratio(total_ret, dd)

    wins   = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    report = {
        "label"          : label,
        "initial_balance": f"${env.initial_balance:.2f}",
        "final_balance"  : f"${env.balance:.2f}",
        "total_return"   : f"{total_ret*100:.1f}%",
        "total_trades"   : len(profits),
        "win_rate"       : f"{wr*100:.1f}%",
        "profit_factor"  : f"{pf:.2f}",
        "sharpe_ratio"   : f"{sr:.2f}",
        "max_drawdown"   : f"{dd*100:.1f}%",
        "calmar_ratio"   : f"{calmar:.2f}" if calmar != float('inf') else "inf",
        "avg_win"        : f"${np.mean(wins):.2f}" if wins else "$0",
        "avg_loss"       : f"${np.mean(losses):.2f}" if losses else "$0",
        "max_win"        : f"${max(profits):.2f}" if profits else "$0",
        "max_loss"       : f"${min(profits):.2f}" if profits else "$0",
        "max_balance"    : f"${max(bal):.2f}",
        "min_balance"    : f"${min(bal):.2f}",
    }

    print("=" * 55)
    print(f"  {label} — Performance Report")
    print("=" * 55)
    for k, v in report.items():
        if k != "label":
            print(f"  {k:<20}: {v}")
    return report


def save_history(env, output_dir, label="week"):
    os.makedirs(output_dir, exist_ok=True)
    trades = [
        t if isinstance(t, dict) else {"profit": t, "type": "manual"}
        for t in env.trade_history
    ]
    data = {"balance_history": env.balance_history, "trades": trades}
    path = os.path.join(output_dir, f"history_{label}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  History saved → {path}")
    return path