import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_forex_data
from env.forex_env_pro_4 import ForexEnvWeek4
from agent.ppo_agent import PPOAgent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_actor_critic_week4_best.pt")
REPORT_DIR = os.path.join(BASE_DIR, "visualize")

os.makedirs(REPORT_DIR, exist_ok=True)


def calculate_max_drawdown(equity_curve):
    equity = np.array(equity_curve, dtype=np.float64)

    if len(equity) == 0:
        return 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-8)

    return float(drawdown.min())


def evaluate_agent(env, agent):
    obs, _ = env.reset()
    done = False

    action_count = [0 for _ in range(env.action_space.n)]
    equity_curve = []
    step_logs = []

    step = 0

    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_count[action] += 1
        equity_curve.append(float(env.balance))

        step_logs.append({
            "step": step,
            "action": action,
            "reward": float(reward),
            "balance": float(env.balance),
            "position": int(env.position),
        })

        step += 1

    initial_balance = float(env.initial_balance)
    final_balance = float(env.balance)
    total_return = (final_balance - initial_balance) / initial_balance

    trades = getattr(env, "trade_history", [])

    max_drawdown = calculate_max_drawdown(equity_curve)

    report = {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "total_trades": len(trades),
        "action_hold": action_count[0],
        "action_buy": action_count[1],
        "action_sell": action_count[2],
    }

    return report, step_logs


def evaluate_buy_and_hold(df):
    initial_balance = 1000.0

    first_close = float(df.iloc[0]["close"])
    last_close = float(df.iloc[-1]["close"])

    total_return = (last_close - first_close) / first_close
    final_balance = initial_balance * (1.0 + total_return)

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return * 100,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", type=float, default=0.8)

    args = parser.parse_args()

    df = load_forex_data(args.data_path)

    split_idx = int(len(df) * args.split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    test_env = ForexEnvWeek4(test_df)

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    agent.load(args.model_path)

    report, step_logs = evaluate_agent(test_env, agent)
    buy_hold_report = evaluate_buy_and_hold(test_df)

    print("=" * 80)
    print("EVALUASI PPO MODEL")
    print("=" * 80)
    print(f"Model path          : {args.model_path}")
    print(f"Data path           : {args.data_path}")
    print(f"Train rows          : {len(train_df)}")
    print(f"Test rows           : {len(test_df)}")
    print("=" * 80)
    print("PPO AGENT")
    print("-" * 80)
    print(f"Initial balance     : {report['initial_balance']:.2f}")
    print(f"Final balance       : {report['final_balance']:.2f}")
    print(f"Total return        : {report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {report['max_drawdown_pct']:.2f}%")
    print(f"Total trades        : {report['total_trades']}")
    print(f"Actions             : Hold={report['action_hold']}, Buy={report['action_buy']}, Sell={report['action_sell']}")
    print("=" * 80)
    print("BUY AND HOLD BENCHMARK")
    print("-" * 80)
    print(f"Initial balance     : {buy_hold_report['initial_balance']:.2f}")
    print(f"Final balance       : {buy_hold_report['final_balance']:.2f}")
    print(f"Total return        : {buy_hold_report['total_return_pct']:+.2f}%")
    print("=" * 80)

    logs_path = os.path.join(REPORT_DIR, "ppo_best_evaluation_steps.csv")
    report_path = os.path.join(REPORT_DIR, "ppo_best_evaluation_report.csv")

    pd.DataFrame(step_logs).to_csv(logs_path, index=False)
    pd.DataFrame([report]).to_csv(report_path, index=False)

    print(f"Step logs saved     : {logs_path}")
    print(f"Report saved        : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()