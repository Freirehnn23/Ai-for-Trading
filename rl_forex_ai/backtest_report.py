import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_forex_data
from env.forex_env_pro_4 import ForexEnvWeek4
from agent.ppo_agent import PPOAgent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")

SELECTED_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "selected",
    "ppo_best_return_29_27pct.pt",
)

BEST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ppo_actor_critic_week4_best.pt",
)

REPORT_DIR = os.path.join(BASE_DIR, "visualize", "backtest_report")


def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def calculate_max_drawdown(equity_curve):
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) == 0:
        return 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-8)

    return float(drawdown.min())


def calculate_sharpe_from_equity(equity_curve):
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2:
        return 0.0

    returns = np.diff(equity) / (equity[:-1] + 1e-8)

    if returns.std() == 0:
        return 0.0

    # Approx daily Sharpe, asumsi data harian.
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

    return float(sharpe)


def evaluate_buy_and_hold(test_df, initial_balance=1000.0):
    first_close = float(test_df.iloc[0]["close"])
    last_close = float(test_df.iloc[-1]["close"])

    total_return = (last_close - first_close) / first_close
    final_balance = initial_balance * (1.0 + total_return)

    equity_curve = []

    for _, row in test_df.iterrows():
        price = float(row["close"])
        equity = initial_balance * (price / first_close)
        equity_curve.append(equity)

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "equity_curve": equity_curve,
    }


def evaluate_ppo(test_df, model_path):
    env = ForexEnvWeek4(test_df)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    agent.load(model_path)

    obs, _ = env.reset()
    done = False

    step_logs = []
    equity_curve = []
    action_count = {
        "hold": 0,
        "buy": 0,
        "sell": 0,
    }

    step = 0

    while not done:
        action, log_prob, value = agent.select_action(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if action == 0:
            action_count["hold"] += 1
            action_name = "HOLD"
        elif action == 1:
            action_count["buy"] += 1
            action_name = "BUY_OR_CLOSE_SELL"
        elif action == 2:
            action_count["sell"] += 1
            action_name = "SELL_OR_CLOSE_BUY"
        else:
            action_name = f"ACTION_{action}"

        balance = float(env.balance)
        equity_curve.append(balance)

        price = float(test_df.iloc[min(step + 1, len(test_df) - 1)]["close"])

        step_logs.append({
            "step": step,
            "price": price,
            "action": int(action),
            "action_name": action_name,
            "reward": float(reward),
            "balance": balance,
            "position": int(env.position),
            "entry_price": float(getattr(env, "entry_price", 0.0)),
            "sl": float(getattr(env, "sl_price", 0.0)),
            "tp": float(getattr(env, "tp_price", 0.0)),
            "log_prob": float(log_prob),
            "value_estimate": float(value),
        })

        obs = next_obs
        step += 1

    initial_balance = float(env.initial_balance)
    final_balance = float(env.balance)
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100

    trades = getattr(env, "trade_history", [])

    trade_df = pd.DataFrame(trades)

    if len(trades) > 0 and "profit" in trade_df.columns:
        profits = trade_df["profit"].astype(float).values

        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        total_trades = len(profits)
        win_rate_pct = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        best_trade = float(profits.max())
        worst_trade = float(profits.min())
    else:
        total_trades = 0
        win_rate_pct = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        best_trade = 0.0
        worst_trade = 0.0

    report = {
        "model_path": model_path,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "action_hold": action_count["hold"],
        "action_buy": action_count["buy"],
        "action_sell": action_count["sell"],
    }

    return report, step_logs, trade_df, equity_curve


def plot_equity_curve(ppo_equity, buy_hold_equity):
    plt.figure(figsize=(12, 6))

    plt.plot(ppo_equity, label="PPO Agent")
    plt.plot(buy_hold_equity[:len(ppo_equity)], label="Buy and Hold")

    plt.title("Equity Curve: PPO vs Buy and Hold")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)

    path = os.path.join(REPORT_DIR, "equity_curve_ppo_vs_buy_hold.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_actions(step_logs):
    df = pd.DataFrame(step_logs)

    plt.figure(figsize=(12, 4))

    plt.plot(df["action"])

    plt.title("PPO Actions Over Time")
    plt.xlabel("Step")
    plt.ylabel("Action: 0=Hold, 1=Buy/CloseSell, 2=Sell/CloseBuy")
    plt.grid(True)

    path = os.path.join(REPORT_DIR, "ppo_actions_over_time.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def print_report(ppo_report, buy_hold_report):
    print("=" * 80)
    print("BACKTEST REPORT — PPO VS BUY AND HOLD")
    print("=" * 80)

    print("PPO AGENT")
    print("-" * 80)
    print(f"Initial balance     : {ppo_report['initial_balance']:.2f}")
    print(f"Final balance       : {ppo_report['final_balance']:.2f}")
    print(f"Total return        : {ppo_report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {ppo_report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {ppo_report['sharpe']:.3f}")
    print(f"Total trades        : {ppo_report['total_trades']}")
    print(f"Win rate            : {ppo_report['win_rate_pct']:.2f}%")
    print(f"Profit factor       : {ppo_report['profit_factor']:.3f}")
    print(f"Gross profit        : {ppo_report['gross_profit']:.2f}")
    print(f"Gross loss          : {ppo_report['gross_loss']:.2f}")
    print(f"Average win         : {ppo_report['avg_win']:.2f}")
    print(f"Average loss        : {ppo_report['avg_loss']:.2f}")
    print(f"Best trade          : {ppo_report['best_trade']:.2f}")
    print(f"Worst trade         : {ppo_report['worst_trade']:.2f}")
    print(
        "Actions             : "
        f"Hold={ppo_report['action_hold']}, "
        f"Buy={ppo_report['action_buy']}, "
        f"Sell={ppo_report['action_sell']}"
    )

    print("=" * 80)

    print("BUY AND HOLD")
    print("-" * 80)
    print(f"Initial balance     : {buy_hold_report['initial_balance']:.2f}")
    print(f"Final balance       : {buy_hold_report['final_balance']:.2f}")
    print(f"Total return        : {buy_hold_report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {buy_hold_report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {buy_hold_report['sharpe']:.3f}")

    print("=" * 80)

    difference = ppo_report["total_return_pct"] - buy_hold_report["total_return_pct"]

    if difference > 0:
        print(f"RESULT              : PPO menang +{difference:.2f}% dari Buy and Hold")
    else:
        print(f"RESULT              : PPO kalah {difference:.2f}% dari Buy and Hold")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--split", type=float, default=0.8)

    args = parser.parse_args()

    ensure_report_dir()

    if args.model_path is not None:
        model_path = args.model_path
    elif os.path.exists(SELECTED_MODEL_PATH):
        model_path = SELECTED_MODEL_PATH
    else:
        model_path = BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}"
        )

    df = load_forex_data(args.data_path)

    split_idx = int(len(df) * args.split)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows : {len(test_df)}")
    print(f"[INFO] Model path: {model_path}")

    ppo_report, step_logs, trade_df, ppo_equity = evaluate_ppo(
        test_df=test_df,
        model_path=model_path,
    )

    buy_hold_report = evaluate_buy_and_hold(
        test_df=test_df,
        initial_balance=ppo_report["initial_balance"],
    )

    print_report(ppo_report, buy_hold_report)

    report_path = os.path.join(REPORT_DIR, "summary_report.csv")
    steps_path = os.path.join(REPORT_DIR, "step_logs.csv")
    trades_path = os.path.join(REPORT_DIR, "trade_logs.csv")

    pd.DataFrame([{
        **{f"ppo_{k}": v for k, v in ppo_report.items()},
        **{
            "buy_hold_initial_balance": buy_hold_report["initial_balance"],
            "buy_hold_final_balance": buy_hold_report["final_balance"],
            "buy_hold_total_return_pct": buy_hold_report["total_return_pct"],
            "buy_hold_max_drawdown_pct": buy_hold_report["max_drawdown_pct"],
            "buy_hold_sharpe": buy_hold_report["sharpe"],
        },
        "return_difference_pct": (
            ppo_report["total_return_pct"]
            - buy_hold_report["total_return_pct"]
        ),
    }]).to_csv(report_path, index=False)

    pd.DataFrame(step_logs).to_csv(steps_path, index=False)

    if len(trade_df) > 0:
        trade_df.to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["profit", "type"]).to_csv(trades_path, index=False)

    equity_path = plot_equity_curve(
        ppo_equity=ppo_equity,
        buy_hold_equity=buy_hold_report["equity_curve"],
    )

    actions_path = plot_actions(step_logs)

    print("FILES SAVED")
    print("-" * 80)
    print(f"Summary report      : {report_path}")
    print(f"Step logs           : {steps_path}")
    print(f"Trade logs          : {trades_path}")
    print(f"Equity curve chart  : {equity_path}")
    print(f"Actions chart       : {actions_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()