"""
Evaluate PPO Sequence Models:
- CNN-1D PPO
- LSTM PPO
- Attention PPO
- Transformer PPO

Default:
    evaluate model CNN-1D terbaik yang sudah dibackup:
    models/selected/ppo_cnn1d_w32_best_return_30_29pct.pt

Run dari folder rl_forex_ai:

    python evaluate_sequence.py

Atau pilih encoder lain:

    python evaluate_sequence.py --encoder lstm
    python evaluate_sequence.py --encoder cnn1d
    python evaluate_sequence.py --encoder attention
    python evaluate_sequence.py --encoder transformer

Atau pakai model path custom:

    python evaluate_sequence.py --model-path models/sequence/ppo_cnn1d_w32_best.pt --encoder cnn1d
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ppo_sequence_agent import PPOSequenceAgent, PPOSequenceConfig
from env.forex_env_pro_4 import ForexEnvWeek4
from env.windowed_env import WindowedObservationWrapper
from utils.data_loader import load_forex_data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Data_historis(23-26).csv",
)

SELECTED_CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "selected",
    "ppo_cnn1d_w32_best_return_30_29pct.pt",
)

SEQUENCE_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "sequence",
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "visualize",
    "sequence",
    "evaluation",
)


ACTION_NAMES_3 = {
    0: "HOLD",
    1: "BUY_OR_CLOSE_SELL",
    2: "SELL_OR_CLOSE_BUY",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) == 0:
        return 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-8)

    return float(drawdown.min())


def calculate_sharpe_from_equity(equity_curve: List[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)

    if len(equity) < 2:
        return 0.0

    returns = np.diff(equity) / (equity[:-1] + 1e-8)

    if returns.std() == 0:
        return 0.0

    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

    return float(sharpe)


def evaluate_buy_and_hold(test_df: pd.DataFrame, initial_balance: float = 1000.0) -> Dict:
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
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100.0,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "equity_curve": equity_curve,
    }


def resolve_model_path(encoder: str, window_size: int, model_path: str | None) -> str:
    if model_path is not None:
        if os.path.isabs(model_path):
            return model_path

        return os.path.join(BASE_DIR, model_path)

    if encoder == "cnn1d" and os.path.exists(SELECTED_CNN_MODEL_PATH):
        return SELECTED_CNN_MODEL_PATH

    return os.path.join(
        SEQUENCE_MODEL_DIR,
        f"ppo_{encoder}_w{window_size}_best.pt",
    )


def load_agent_from_checkpoint(model_path: str, fallback_encoder: str, fallback_window_size: int):
    checkpoint = torch.load(model_path, map_location="cpu")

    input_shape = tuple(
        checkpoint.get(
            "input_shape",
            (fallback_window_size, 9),
        )
    )

    action_dim = int(
        checkpoint.get(
            "action_dim",
            3,
        )
    )

    config_dict = checkpoint.get("config", {})

    if not isinstance(config_dict, dict):
        config_dict = {}

    config_dict.setdefault("encoder_type", fallback_encoder)

    allowed_keys = PPOSequenceConfig.__dataclass_fields__.keys()

    clean_config = {
        key: value
        for key, value in config_dict.items()
        if key in allowed_keys
    }

    config = PPOSequenceConfig(**clean_config)

    agent = PPOSequenceAgent(
        input_shape=input_shape,
        action_dim=action_dim,
        config=config,
    )

    agent.load(model_path)

    return agent, input_shape, action_dim, config


def make_sequence_env(df: pd.DataFrame, window_size: int):
    base_env = ForexEnvWeek4(df)
    env = WindowedObservationWrapper(
        base_env,
        window_size=window_size,
    )
    return env


def evaluate_sequence_model(
    test_df: pd.DataFrame,
    model_path: str,
    encoder: str,
    window_size: int,
):
    agent, input_shape, action_dim, config = load_agent_from_checkpoint(
        model_path=model_path,
        fallback_encoder=encoder,
        fallback_window_size=window_size,
    )

    env = make_sequence_env(
        test_df,
        window_size=input_shape[0],
    )

    base_env = env.unwrapped

    obs, _ = env.reset()
    done = False

    step_logs = []
    equity_curve = []

    action_count = {
        ACTION_NAMES_3.get(i, f"ACTION_{i}"): 0
        for i in range(action_dim)
    }

    step = 0
    total_reward = 0.0

    while not done:
        action, log_prob, value_estimate = agent.select_action(
            obs,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        action_name = ACTION_NAMES_3.get(action, f"ACTION_{action}")
        action_count[action_name] += 1

        total_reward += float(reward)

        balance = float(base_env.balance)
        equity_curve.append(balance)

        price_idx = min(step + 1, len(test_df) - 1)
        price = float(test_df.iloc[price_idx]["close"])

        step_logs.append({
            "step": step,
            "price": price,
            "action": int(action),
            "action_name": action_name,
            "reward": float(reward),
            "balance": balance,
            "position": int(base_env.position),
            "entry_price": float(getattr(base_env, "entry_price", 0.0)),
            "sl": float(getattr(base_env, "sl_price", 0.0)),
            "tp": float(getattr(base_env, "tp_price", 0.0)),
            "log_prob": float(log_prob),
            "value_estimate": float(value_estimate),
        })

        obs = next_obs
        step += 1

    initial_balance = float(base_env.initial_balance)
    final_balance = float(base_env.balance)
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100.0

    trades = getattr(base_env, "trade_history", [])
    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0 and "profit" in trade_df.columns:
        profits = trade_df["profit"].astype(float).values

        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        total_trades = int(len(profits))
        win_rate_pct = float(len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0

        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        best_trade = float(profits.max())
        worst_trade = float(profits.min())

        tp_hits = int((trade_df["type"] == "TP").sum()) if "type" in trade_df.columns else 0
        sl_hits = int((trade_df["type"] == "SL").sum()) if "type" in trade_df.columns else 0
        manual_closes = int(total_trades - tp_hits - sl_hits)
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
        tp_hits = 0
        sl_hits = 0
        manual_closes = 0

    report = {
        "model_path": model_path,
        "encoder": config.encoder_type,
        "input_shape": str(input_shape),
        "action_dim": action_dim,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100.0,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "total_reward": total_reward,
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "manual_closes": manual_closes,
        "action_hold": action_count.get("HOLD", 0),
        "action_buy_or_close_sell": action_count.get("BUY_OR_CLOSE_SELL", 0),
        "action_sell_or_close_buy": action_count.get("SELL_OR_CLOSE_BUY", 0),
    }

    return report, step_logs, trade_df, equity_curve


def plot_equity_curve(ppo_equity, buy_hold_equity, output_path: str) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(ppo_equity, label="PPO Sequence")
    plt.plot(buy_hold_equity[:len(ppo_equity)], label="Buy and Hold")

    plt.title("Equity Curve: PPO Sequence vs Buy and Hold")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_actions(step_logs, output_path: str) -> None:
    df = pd.DataFrame(step_logs)

    plt.figure(figsize=(12, 4))

    plt.plot(df["action"])

    plt.title("PPO Sequence Actions Over Time")
    plt.xlabel("Step")
    plt.ylabel("Action: 0=Hold, 1=Buy/CloseSell, 2=Sell/CloseBuy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_report(sequence_report: Dict, buy_hold_report: Dict) -> None:
    print("=" * 80)
    print("EVALUASI PPO SEQUENCE MODEL")
    print("=" * 80)

    print(f"Model path          : {sequence_report['model_path']}")
    print(f"Encoder             : {sequence_report['encoder']}")
    print(f"Input shape         : {sequence_report['input_shape']}")
    print(f"Action dim          : {sequence_report['action_dim']}")

    print("=" * 80)
    print("PPO SEQUENCE AGENT")
    print("-" * 80)
    print(f"Initial balance     : {sequence_report['initial_balance']:.2f}")
    print(f"Final balance       : {sequence_report['final_balance']:.2f}")
    print(f"Total return        : {sequence_report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {sequence_report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {sequence_report['sharpe']:.3f}")
    print(f"Total trades        : {sequence_report['total_trades']}")
    print(f"Win rate            : {sequence_report['win_rate_pct']:.2f}%")
    print(f"Profit factor       : {sequence_report['profit_factor']:.3f}")
    print(f"Gross profit        : {sequence_report['gross_profit']:.2f}")
    print(f"Gross loss          : {sequence_report['gross_loss']:.2f}")
    print(f"Average win         : {sequence_report['avg_win']:.2f}")
    print(f"Average loss        : {sequence_report['avg_loss']:.2f}")
    print(f"Best trade          : {sequence_report['best_trade']:.2f}")
    print(f"Worst trade         : {sequence_report['worst_trade']:.2f}")
    print(f"TP hits             : {sequence_report['tp_hits']}")
    print(f"SL hits             : {sequence_report['sl_hits']}")
    print(f"Manual closes       : {sequence_report['manual_closes']}")
    print(
        "Actions             : "
        f"Hold={sequence_report['action_hold']}, "
        f"Buy/CloseSell={sequence_report['action_buy_or_close_sell']}, "
        f"Sell/CloseBuy={sequence_report['action_sell_or_close_buy']}"
    )

    print("=" * 80)
    print("BUY AND HOLD BENCHMARK")
    print("-" * 80)
    print(f"Initial balance     : {buy_hold_report['initial_balance']:.2f}")
    print(f"Final balance       : {buy_hold_report['final_balance']:.2f}")
    print(f"Total return        : {buy_hold_report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {buy_hold_report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {buy_hold_report['sharpe']:.3f}")

    print("=" * 80)

    diff = sequence_report["total_return_pct"] - buy_hold_report["total_return_pct"]

    if diff > 0:
        print(f"RESULT              : PPO Sequence menang +{diff:.2f}% dari Buy and Hold")
    else:
        print(f"RESULT              : PPO Sequence kalah {diff:.2f}% dari Buy and Hold")

    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)

    parser.add_argument(
        "--encoder",
        type=str,
        default="cnn1d",
        choices=["cnn1d", "lstm", "attention", "transformer"],
    )

    parser.add_argument("--window-size", type=int, default=32)

    parser.add_argument("--split", type=float, default=0.80)

    parser.add_argument("--model-path", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(REPORT_DIR)

    model_path = resolve_model_path(
        encoder=args.encoder,
        window_size=args.window_size,
        model_path=args.model_path,
    )

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

    sequence_report, step_logs, trade_df, sequence_equity = evaluate_sequence_model(
        test_df=test_df,
        model_path=model_path,
        encoder=args.encoder,
        window_size=args.window_size,
    )

    buy_hold_report = evaluate_buy_and_hold(
        test_df=test_df,
        initial_balance=sequence_report["initial_balance"],
    )

    print_report(
        sequence_report=sequence_report,
        buy_hold_report=buy_hold_report,
    )

    safe_encoder = sequence_report["encoder"]

    summary_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_summary_report.csv",
    )

    steps_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_step_logs.csv",
    )

    trades_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_trade_logs.csv",
    )

    equity_chart_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_equity_curve_vs_buy_hold.png",
    )

    actions_chart_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_actions_over_time.png",
    )

    merged_summary = {
        **{f"sequence_{k}": v for k, v in sequence_report.items()},
        "buy_hold_initial_balance": buy_hold_report["initial_balance"],
        "buy_hold_final_balance": buy_hold_report["final_balance"],
        "buy_hold_total_return_pct": buy_hold_report["total_return_pct"],
        "buy_hold_max_drawdown_pct": buy_hold_report["max_drawdown_pct"],
        "buy_hold_sharpe": buy_hold_report["sharpe"],
        "return_difference_pct": (
            sequence_report["total_return_pct"]
            - buy_hold_report["total_return_pct"]
        ),
    }

    pd.DataFrame([merged_summary]).to_csv(summary_path, index=False)
    pd.DataFrame(step_logs).to_csv(steps_path, index=False)

    if len(trade_df) > 0:
        trade_df.to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["profit", "type"]).to_csv(trades_path, index=False)

    plot_equity_curve(
        ppo_equity=sequence_equity,
        buy_hold_equity=buy_hold_report["equity_curve"],
        output_path=equity_chart_path,
    )

    plot_actions(
        step_logs=step_logs,
        output_path=actions_chart_path,
    )

    print("FILES SAVED")
    print("-" * 80)
    print(f"Summary report      : {summary_path}")
    print(f"Step logs           : {steps_path}")
    print(f"Trade logs          : {trades_path}")
    print(f"Equity curve chart  : {equity_chart_path}")
    print(f"Actions chart       : {actions_chart_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()