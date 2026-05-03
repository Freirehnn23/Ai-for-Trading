"""
Stress Test PPO Sequence Model.

UPDATE:
- Tetap bisa mengetes Week4 champion sebagai baseline.
- Bisa mengetes Week5 action-masking model dengan --env-version week5.

Contoh:
    python stress_test_sequence.py --env-version week4 --encoder cnn1d
    python stress_test_sequence.py --env-version week5 --encoder cnn1d
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ppo_sequence_agent import PPOSequenceAgent, PPOSequenceConfig
from env.forex_env_pro_4 import ForexEnvWeek4
from env.forex_env_pro_5 import ForexEnvWeek5
from env.windowed_env import WindowedObservationWrapper
from utils.data_loader import load_forex_data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
REPORT_DIR = os.path.join(BASE_DIR, "visualize", "sequence", "stress_test")

DEFAULT_WEEK4_MODEL = os.path.join(BASE_DIR, "models", "selected", "ppo_cnn1d_w32_best_return_30_29pct.pt")
DEFAULT_WEEK5_MODEL = os.path.join(BASE_DIR, "models", "sequence", "week5", "ppo_cnn1d_week5_w32_best.pt")

SCENARIOS = [
    {"name": "normal", "spread": 0.20, "slippage": 0.05, "commission": 0.50},
    {"name": "medium", "spread": 0.40, "slippage": 0.10, "commission": 0.75},
    {"name": "hard", "spread": 0.80, "slippage": 0.20, "commission": 1.00},
    {"name": "very_hard", "spread": 1.20, "slippage": 0.30, "commission": 1.50},
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_df(df: pd.DataFrame, env_version: str, split: float, train_ratio: float, val_ratio: float):
    if env_version == "week5":
        n = len(df)
        val_end = int(n * (train_ratio + val_ratio))
        return df.iloc[val_end:].reset_index(drop=True)

    idx = int(len(df) * split)
    return df.iloc[idx:].reset_index(drop=True)


def max_drawdown(equity) -> float:
    arr = np.asarray(equity, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (peak + 1e-8)
    return float(dd.min())


def buy_hold_return(test_df: pd.DataFrame, initial_balance: float = 1000.0):
    first = float(test_df.iloc[0]["close"])
    curve = [initial_balance * (float(row["close"]) / first) for _, row in test_df.iterrows()]
    return {
        "return_pct": (curve[-1] - initial_balance) / initial_balance * 100.0,
        "max_drawdown_pct": max_drawdown(curve) * 100.0,
    }


def load_agent(model_path, fallback_encoder="cnn1d", fallback_window_size=32):
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    input_shape = tuple(ckpt.get("input_shape", (fallback_window_size, 9)))
    action_dim = int(ckpt.get("action_dim", 3))
    config_dict = ckpt.get("config", {})
    if not isinstance(config_dict, dict):
        config_dict = {}
    config_dict.setdefault("encoder_type", fallback_encoder)

    allowed = PPOSequenceConfig.__dataclass_fields__.keys()
    config = PPOSequenceConfig(**{k: v for k, v in config_dict.items() if k in allowed})
    agent = PPOSequenceAgent(input_shape=input_shape, action_dim=action_dim, config=config)
    agent.load(model_path)
    agent.net.eval()
    return agent, input_shape, action_dim, config


def make_env(env_version, df, window_size, scenario, args):
    if env_version == "week4":
        base = ForexEnvWeek4(df)
        # Override cost params if attributes exist.
        for key in ["spread", "slippage", "commission"]:
            if hasattr(base, key):
                setattr(base, key, float(scenario[key]))
    elif env_version == "week5":
        base = ForexEnvWeek5(
            df,
            spread=float(scenario["spread"]),
            slippage=float(scenario["slippage"]),
            commission=float(scenario["commission"]),
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
    else:
        raise ValueError(env_version)

    return WindowedObservationWrapper(base, window_size=window_size)


def get_mask(env, env_version: str):
    if env_version == "week5" and hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return None


def evaluate_once(env_version, test_df, agent, input_shape, action_dim, scenario, seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_version, test_df, input_shape[0], scenario, args)
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    done = False
    action_count = {i: 0 for i in range(action_dim)}

    while not done:
        mask = get_mask(env, env_version)
        action, _, _ = agent.select_action(
            obs,
            deterministic=True,
            action_mask=mask,
            min_confidence=args.min_confidence if env_version == "week5" else None,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action_count[action] = action_count.get(action, 0) + 1

    equity = getattr(base, "equity_history", getattr(base, "balance_history", [base.balance]))
    final_equity = float(equity[-1]) if len(equity) else float(base.balance)
    ret = (final_equity - base.initial_balance) / base.initial_balance * 100.0

    trades = getattr(base, "trade_history", [])
    profits = [float(t.get("profit", 0.0)) for t in trades if isinstance(t, dict)]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    wr = len(wins) / len(profits) * 100.0 if profits else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0 else (float("inf") if wins else 0.0)

    return {
        "scenario": scenario["name"],
        "seed": seed,
        "spread": scenario["spread"],
        "slippage": scenario["slippage"],
        "commission": scenario["commission"],
        "return_pct": ret,
        "max_drawdown_pct": max_drawdown(equity) * 100.0,
        "total_trades": len(profits),
        "win_rate_pct": wr,
        "profit_factor": pf,
        "action_count": str(action_count),
    }


def summarize(detail_df: pd.DataFrame, buy_hold_pct: float) -> pd.DataFrame:
    rows = []
    for scenario, g in detail_df.groupby("scenario"):
        returns = g["return_pct"].astype(float)
        rows.append({
            "scenario": scenario,
            "runs": len(g),
            "mean_return_pct": returns.mean(),
            "min_return_pct": returns.min(),
            "max_return_pct": returns.max(),
            "mean_edge_vs_buy_hold_pct": (returns - buy_hold_pct).mean(),
            "mean_max_drawdown_pct": g["max_drawdown_pct"].astype(float).mean(),
            "mean_trades": g["total_trades"].astype(float).mean(),
            "mean_win_rate_pct": g["win_rate_pct"].astype(float).mean(),
            "mean_profit_factor": g["profit_factor"].replace([np.inf, -np.inf], np.nan).astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values("scenario")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--env-version", default="week4", choices=["week4", "week5"])
    p.add_argument("--encoder", default="cnn1d", choices=["cnn1d", "lstm", "attention", "transformer"])
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--model-path", default=None)
    p.add_argument("--runs-per-scenario", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=2001)
    p.add_argument("--split", type=float, default=0.80)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--min-confidence", type=float, default=None)

    p.add_argument("--contract-size", type=float, default=100.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--sl-pct", type=float, default=0.010)
    p.add_argument("--tp-pct", type=float, default=0.020)
    p.add_argument("--initial-balance", type=float, default=1000.0)
    p.add_argument("--trade-penalty", type=float, default=0.01)
    p.add_argument("--action-penalty", type=float, default=0.002)
    p.add_argument("--invalid-action-penalty", type=float, default=0.05)
    p.add_argument("--drawdown-penalty", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(REPORT_DIR)

    if args.model_path:
        model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(BASE_DIR, args.model_path)
    else:
        model_path = DEFAULT_WEEK4_MODEL if args.env_version == "week4" else DEFAULT_WEEK5_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    df = load_forex_data(args.data_path)
    test_df = split_df(df, args.env_version, args.split, args.train_ratio, args.val_ratio)
    bh = buy_hold_return(test_df, initial_balance=args.initial_balance)

    agent, input_shape, action_dim, config = load_agent(model_path, args.encoder, args.window_size)

    print("=" * 80)
    print("STRESS TEST PPO SEQUENCE")
    print("=" * 80)
    print(f"Env version       : {args.env_version}")
    print(f"Model             : {model_path}")
    print(f"Encoder           : {config.encoder_type}")
    print(f"Input shape       : {input_shape}")
    print(f"Action dim        : {action_dim}")
    print(f"Test rows         : {len(test_df)}")
    print(f"Buy&Hold return   : {bh['return_pct']:+.2f}%")
    print("Action mask       : " + ("ON" if args.env_version == "week5" else "OFF"))
    print("=" * 80)

    rows = []
    seed = args.seed_start
    for scenario in SCENARIOS:
        for _ in range(args.runs_per_scenario):
            row = evaluate_once(args.env_version, test_df, agent, input_shape, action_dim, scenario, seed, args)
            row["edge_vs_buy_hold_pct"] = row["return_pct"] - bh["return_pct"]
            rows.append(row)
            print(
                f"{scenario['name']:<10} | seed={seed}"
                f" | return={row['return_pct']:+7.2f}%"
                f" | edge={row['edge_vs_buy_hold_pct']:+7.2f}%"
                f" | DD={row['max_drawdown_pct']:7.2f}%"
                f" | trades={row['total_trades']:3d}"
                f" | WR={row['win_rate_pct']:6.2f}%"
                f" | PF={row['profit_factor']:.3f}"
            )
            seed += 1

    detail_df = pd.DataFrame(rows)
    summary_df = summarize(detail_df, bh["return_pct"])

    detail_path = os.path.join(REPORT_DIR, f"stress_test_{args.env_version}_detail.csv")
    summary_path = os.path.join(REPORT_DIR, f"stress_test_{args.env_version}_summary.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("SUMMARY BY SCENARIO")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("FILES SAVED")
    print(f"Detail  : {detail_path}")
    print(f"Summary : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
