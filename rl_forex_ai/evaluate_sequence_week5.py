"""
Evaluate PPO Sequence Model — WEEK5 + ACTION MASKING.

Default mengevaluasi:
    models/sequence/week5/ppo_cnn1d_week5_w32_best.pt

Run dari folder rl_forex_ai:
    python evaluate_sequence_week5.py --encoder cnn1d

Custom:
    python evaluate_sequence_week5.py --encoder cnn1d --model-path models/sequence/week5/ppo_cnn1d_week5_w32_best.pt
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
from env.forex_env_pro_5 import ForexEnvWeek5
from env.windowed_env import WindowedObservationWrapper
from utils.data_loader import load_forex_data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
SEQUENCE_WEEK5_MODEL_DIR = os.path.join(BASE_DIR, "models", "sequence", "week5")
REPORT_DIR = os.path.join(BASE_DIR, "visualize", "sequence", "week5", "evaluation")

ACTION_NAMES_4 = {
    0: "HOLD",
    1: "OPEN_BUY",
    2: "OPEN_SELL",
    3: "CLOSE_POSITION",
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
    return float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-8))


def evaluate_buy_and_hold(test_df: pd.DataFrame, initial_balance: float = 1000.0) -> Dict:
    first_close = float(test_df.iloc[0]["close"])
    last_close = float(test_df.iloc[-1]["close"])
    total_return = (last_close - first_close) / first_close
    final_balance = initial_balance * (1.0 + total_return)
    equity_curve = [initial_balance * (float(row["close"]) / first_close) for _, row in test_df.iterrows()]

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100.0,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "equity_curve": equity_curve,
    }


def split_train_val_test(df, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def resolve_model_path(encoder: str, window_size: int, model_path: str | None) -> str:
    if model_path is not None:
        if os.path.isabs(model_path):
            return model_path
        return os.path.join(BASE_DIR, model_path)

    return os.path.join(
        SEQUENCE_WEEK5_MODEL_DIR,
        f"ppo_{encoder}_week5_w{window_size}_best.pt",
    )


def load_agent_from_checkpoint(model_path: str, fallback_encoder: str, fallback_window_size: int):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    input_shape = tuple(checkpoint.get("input_shape", (fallback_window_size, 9)))
    action_dim = int(checkpoint.get("action_dim", 4))

    config_dict = checkpoint.get("config", {})
    if not isinstance(config_dict, dict):
        config_dict = {}
    config_dict.setdefault("encoder_type", fallback_encoder)

    allowed_keys = PPOSequenceConfig.__dataclass_fields__.keys()
    clean_config = {k: v for k, v in config_dict.items() if k in allowed_keys}
    config = PPOSequenceConfig(**clean_config)

    agent = PPOSequenceAgent(input_shape=input_shape, action_dim=action_dim, config=config)
    agent.load(model_path)
    agent.net.eval()

    return agent, input_shape, action_dim, config


def make_sequence_env(df: pd.DataFrame, window_size: int, args):
    base_env = ForexEnvWeek5(
        df=df,
        spread=args.spread,
        slippage=args.slippage,
        commission=args.commission,
        contract_size=args.contract_size,
        risk_per_trade=args.risk_per_trade,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        initial_balance=args.initial_balance,
        trade_penalty=args.trade_penalty,
        action_penalty=args.action_penalty,
        invalid_action_penalty=args.invalid_action_penalty,
        drawdown_penalty=args.drawdown_penalty,
    )
    return WindowedObservationWrapper(base_env, window_size=window_size)


def get_action_mask(env):
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return None


def evaluate_sequence_model(test_df: pd.DataFrame, model_path: str, encoder: str, window_size: int, args):
    agent, input_shape, action_dim, config = load_agent_from_checkpoint(
        model_path=model_path,
        fallback_encoder=encoder,
        fallback_window_size=window_size,
    )

    env = make_sequence_env(test_df, window_size=input_shape[0], args=args)
    base_env = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    done = False

    step_logs = []
    action_count = {ACTION_NAMES_4.get(i, f"ACTION_{i}"): 0 for i in range(action_dim)}
    total_reward = 0.0
    step = 0

    while not done:
        mask = get_action_mask(env)
        action, log_prob, value_estimate = agent.select_action(
            obs,
            deterministic=True,
            action_mask=mask,
            min_confidence=args.min_confidence,
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_name = ACTION_NAMES_4.get(action, f"ACTION_{action}")
        action_count[action_name] += 1
        total_reward += float(reward)

        price_idx = min(step + 1, len(test_df) - 1)
        price = float(test_df.iloc[price_idx]["close"])

        equity = float(base_env.equity_history[-1]) if getattr(base_env, "equity_history", None) else float(base_env.balance)

        step_logs.append({
            "step": step,
            "price": price,
            "action": int(action),
            "action_name": action_name,
            "action_mask": "".join(["1" if x else "0" for x in np.asarray(mask, dtype=bool)]),
            "reward": float(reward),
            "balance": float(base_env.balance),
            "equity": equity,
            "position": int(base_env.position),
            "entry_price": float(getattr(base_env, "entry_price", 0.0)),
            "sl": float(getattr(base_env, "sl_price", 0.0)),
            "tp": float(getattr(base_env, "tp_price", 0.0)),
            "log_prob": float(log_prob),
            "value_estimate": float(value_estimate),
            **{f"info_{k}": v for k, v in info.items() if isinstance(v, (int, float, str, bool, type(None)))},
        })

        obs = next_obs
        step += 1

    initial_balance = float(base_env.initial_balance)
    final_balance = float(base_env.balance)
    equity_curve = [float(x) for x in getattr(base_env, "equity_history", [])]
    final_equity = float(equity_curve[-1]) if equity_curve else final_balance
    total_return_pct = ((final_equity - initial_balance) / initial_balance) * 100.0

    trade_df = pd.DataFrame(getattr(base_env, "trade_history", []))

    if len(trade_df) > 0 and "profit" in trade_df.columns:
        profits = trade_df["profit"].astype(float).values
        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        total_trades = int(len(profits))
        win_rate_pct = float(len(wins) / total_trades * 100.0) if total_trades else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        best_trade = float(profits.max())
        worst_trade = float(profits.min())

        if "type" in trade_df.columns:
            tp_hits = int((trade_df["type"] == "TP").sum())
            sl_hits = int((trade_df["type"] == "SL").sum())
            manual_closes = int(trade_df["type"].astype(str).str.contains("manual_close", case=False, na=False).sum())
        else:
            tp_hits = 0
            sl_hits = 0
            manual_closes = 0
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
        "final_equity": final_equity,
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
        "action_open_buy": action_count.get("OPEN_BUY", 0),
        "action_open_sell": action_count.get("OPEN_SELL", 0),
        "action_close_position": action_count.get("CLOSE_POSITION", 0),
        "equity_curve": equity_curve,
        "step_logs": step_logs,
        "trade_df": trade_df,
        "action_count": action_count,
    }

    return report


def plot_equity_curve(report: Dict, buy_hold_report: Dict, output_path: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(report["equity_curve"], label="PPO Week5")
    plt.plot(buy_hold_report["equity_curve"], label="Buy and Hold")
    plt.title("PPO Week5 Equity Curve vs Buy and Hold")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_actions(step_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.scatter(step_df["step"], step_df["action"], s=18)
    plt.yticks([0, 1, 2, 3], ["HOLD", "OPEN_BUY", "OPEN_SELL", "CLOSE"])
    plt.title("PPO Week5 Actions Over Time")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_report(report: Dict, buy_hold_report: Dict) -> None:
    print("=" * 80)
    print("EVALUASI PPO SEQUENCE WEEK5 — ACTION MASK ON")
    print("=" * 80)
    print(f"Encoder             : {report['encoder']}")
    print(f"Input shape         : {report['input_shape']}")
    print(f"Action dim          : {report['action_dim']}")
    print("=" * 80)
    print("PPO WEEK5 AGENT")
    print("-" * 80)
    print(f"Initial balance     : {report['initial_balance']:.2f}")
    print(f"Final equity        : {report['final_equity']:.2f}")
    print(f"Total return        : {report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {report['sharpe']:.3f}")
    print(f"Total trades        : {report['total_trades']}")
    print(f"Win rate            : {report['win_rate_pct']:.2f}%")
    print(f"Profit factor       : {report['profit_factor']:.3f}")
    print(f"TP hits             : {report['tp_hits']}")
    print(f"SL hits             : {report['sl_hits']}")
    print(f"Manual closes       : {report['manual_closes']}")
    print(f"Actions             : {report['action_count']}")
    print("=" * 80)
    print("BUY AND HOLD BENCHMARK")
    print("-" * 80)
    print(f"Final balance       : {buy_hold_report['final_balance']:.2f}")
    print(f"Total return        : {buy_hold_report['total_return_pct']:+.2f}%")
    print(f"Max drawdown        : {buy_hold_report['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio        : {buy_hold_report['sharpe']:.3f}")
    print("=" * 80)
    diff = report["total_return_pct"] - buy_hold_report["total_return_pct"]
    if diff > 0:
        print(f"RESULT              : PPO Week5 menang {diff:+.2f}% dari Buy and Hold")
    else:
        print(f"RESULT              : PPO Week5 kalah {diff:+.2f}% dari Buy and Hold")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--encoder", type=str, default="cnn1d", choices=["cnn1d", "lstm", "attention", "transformer"])
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=None)

    parser.add_argument("--spread", type=float, default=0.20)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--commission", type=float, default=0.50)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--sl-pct", type=float, default=0.010)
    parser.add_argument("--tp-pct", type=float, default=0.020)
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--trade-penalty", type=float, default=0.01)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.05)
    parser.add_argument("--drawdown-penalty", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(REPORT_DIR)

    model_path = resolve_model_path(args.encoder, args.window_size, args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model Week5 tidak ditemukan: {model_path}")

    df = load_forex_data(args.data_path)
    _, _, test_df = split_train_val_test(df, args.train_ratio, args.val_ratio)

    buy_hold_report = evaluate_buy_and_hold(test_df, initial_balance=args.initial_balance)
    report = evaluate_sequence_model(
        test_df=test_df,
        model_path=model_path,
        encoder=args.encoder,
        window_size=args.window_size,
        args=args,
    )

    print_report(report, buy_hold_report)

    prefix = f"{args.encoder}_week5"
    summary_path = os.path.join(REPORT_DIR, f"{prefix}_summary_report.csv")
    step_path = os.path.join(REPORT_DIR, f"{prefix}_step_logs.csv")
    trade_path = os.path.join(REPORT_DIR, f"{prefix}_trade_logs.csv")
    equity_path = os.path.join(REPORT_DIR, f"{prefix}_equity_curve_vs_buy_hold.png")
    actions_path = os.path.join(REPORT_DIR, f"{prefix}_actions_over_time.png")

    summary = {k: v for k, v in report.items() if k not in ["equity_curve", "step_logs", "trade_df", "action_count"]}
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    step_df = pd.DataFrame(report["step_logs"])
    step_df.to_csv(step_path, index=False)
    report["trade_df"].to_csv(trade_path, index=False)
    plot_equity_curve(report, buy_hold_report, equity_path)
    plot_actions(step_df, actions_path)

    print("FILES SAVED")
    print("-" * 80)
    print(f"Summary report      : {summary_path}")
    print(f"Step logs           : {step_path}")
    print(f"Trade logs          : {trade_path}")
    print(f"Equity curve chart  : {equity_path}")
    print(f"Actions chart       : {actions_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
