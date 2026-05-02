"""
Monte Carlo Evaluation untuk PPO Sequence Model.

Tujuan:
- Menguji model CNN-1D PPO berkali-kali dengan seed/slippage berbeda.
- Mengecek apakah return +30.29% stabil atau hanya kebetulan.
- Default mengevaluasi model champion:

    models/selected/ppo_cnn1d_w32_best_return_30_29pct.pt

Cara run dari folder rl_forex_ai:

    python monte_carlo_evaluate_sequence.py

Run 50 kali:

    python monte_carlo_evaluate_sequence.py --runs 50

Evaluasi model lain:

    python monte_carlo_evaluate_sequence.py --encoder lstm
    python monte_carlo_evaluate_sequence.py --encoder attention
    python monte_carlo_evaluate_sequence.py --encoder transformer
"""

from __future__ import annotations

import argparse
import os
import random
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
    "monte_carlo",
)


ACTION_NAMES_3 = {
    0: "HOLD",
    1: "BUY_OR_CLOSE_SELL",
    2: "SELL_OR_CLOSE_BUY",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    agent.net.eval()

    return agent, input_shape, action_dim, config


def make_sequence_env(df: pd.DataFrame, window_size: int):
    base_env = ForexEnvWeek4(df)

    env = WindowedObservationWrapper(
        base_env,
        window_size=window_size,
    )

    return env


def evaluate_one_run(
    test_df: pd.DataFrame,
    agent: PPOSequenceAgent,
    input_shape,
    action_dim: int,
    run_id: int,
    seed: int,
) -> Dict:
    set_seed(seed)

    window_size = int(input_shape[0])

    env = make_sequence_env(
        test_df,
        window_size=window_size,
    )

    base_env = env.unwrapped

    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()

    done = False

    action_count = {
        ACTION_NAMES_3.get(i, f"ACTION_{i}"): 0
        for i in range(action_dim)
    }

    equity_curve = [float(base_env.balance)]
    total_reward = 0.0
    step = 0

    while not done:
        action, _, _ = agent.select_action(
            obs,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_name = ACTION_NAMES_3.get(action, f"ACTION_{action}")
        action_count[action_name] += 1

        total_reward += float(reward)
        equity_curve.append(float(base_env.balance))

        obs = next_obs
        step += 1

    initial_balance = float(base_env.initial_balance)
    final_balance = float(base_env.balance)

    total_return_pct = (
        (final_balance - initial_balance)
        / initial_balance
        * 100.0
    )

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

        if "type" in trade_df.columns:
            tp_hits = int((trade_df["type"] == "TP").sum())
            sl_hits = int((trade_df["type"] == "SL").sum())
        else:
            tp_hits = 0
            sl_hits = 0

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

    return {
        "run_id": run_id,
        "seed": seed,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": calculate_max_drawdown(equity_curve) * 100.0,
        "sharpe": calculate_sharpe_from_equity(equity_curve),
        "total_reward": total_reward,
        "steps": step,
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


def summarize_results(results_df: pd.DataFrame, buy_hold_report: Dict) -> Dict:
    returns = results_df["total_return_pct"].astype(float)
    drawdowns = results_df["max_drawdown_pct"].astype(float)
    sharpes = results_df["sharpe"].astype(float)
    final_balances = results_df["final_balance"].astype(float)

    buy_hold_return = float(buy_hold_report["total_return_pct"])

    edges = returns - buy_hold_return
    beat_mask = returns > buy_hold_return

    stats = {
        "runs": int(len(results_df)),
        "buy_hold_return_pct": buy_hold_return,
        "buy_hold_final_balance": float(buy_hold_report["final_balance"]),
        "buy_hold_max_drawdown_pct": float(buy_hold_report["max_drawdown_pct"]),
        "buy_hold_sharpe": float(buy_hold_report["sharpe"]),

        "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "std_return_pct": float(returns.std(ddof=0)),
        "min_return_pct": float(returns.min()),
        "max_return_pct": float(returns.max()),

        "mean_final_balance": float(final_balances.mean()),
        "min_final_balance": float(final_balances.min()),
        "max_final_balance": float(final_balances.max()),

        "mean_edge_vs_buy_hold_pct": float(edges.mean()),
        "min_edge_vs_buy_hold_pct": float(edges.min()),
        "max_edge_vs_buy_hold_pct": float(edges.max()),

        "beat_buy_hold_count": int(beat_mask.sum()),
        "beat_buy_hold_rate_pct": float(beat_mask.mean() * 100.0),

        "mean_max_drawdown_pct": float(drawdowns.mean()),
        "worst_max_drawdown_pct": float(drawdowns.min()),
        "best_max_drawdown_pct": float(drawdowns.max()),

        "mean_sharpe": float(sharpes.mean()),
        "min_sharpe": float(sharpes.min()),
        "max_sharpe": float(sharpes.max()),

        "mean_trades": float(results_df["total_trades"].mean()),
        "mean_win_rate_pct": float(results_df["win_rate_pct"].mean()),
        "mean_profit_factor": float(results_df["profit_factor"].replace([np.inf, -np.inf], np.nan).mean()),

        "mean_action_hold": float(results_df["action_hold"].mean()),
        "mean_action_buy_or_close_sell": float(results_df["action_buy_or_close_sell"].mean()),
        "mean_action_sell_or_close_buy": float(results_df["action_sell_or_close_buy"].mean()),
    }

    return stats


def plot_return_distribution(results_df: pd.DataFrame, buy_hold_return: float, output_path: str) -> None:
    plt.figure(figsize=(10, 6))

    plt.hist(results_df["total_return_pct"], bins=10)
    plt.axvline(buy_hold_return, linestyle="--", label="Buy and Hold")

    plt.title("Monte Carlo Return Distribution")
    plt.xlabel("Total Return (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_returns_by_run(results_df: pd.DataFrame, buy_hold_return: float, output_path: str) -> None:
    plt.figure(figsize=(12, 5))

    plt.plot(results_df["run_id"], results_df["total_return_pct"], marker="o")
    plt.axhline(buy_hold_return, linestyle="--", label="Buy and Hold")

    plt.title("Monte Carlo Return by Run")
    plt.xlabel("Run")
    plt.ylabel("Total Return (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_stats(stats: Dict, model_path: str, encoder: str, input_shape) -> None:
    print("=" * 80)
    print("MONTE CARLO EVALUATION — PPO SEQUENCE")
    print("=" * 80)
    print(f"Model path              : {model_path}")
    print(f"Encoder                 : {encoder}")
    print(f"Input shape             : {input_shape}")
    print(f"Runs                    : {stats['runs']}")
    print("=" * 80)

    print("BUY AND HOLD")
    print("-" * 80)
    print(f"Return                  : {stats['buy_hold_return_pct']:+.2f}%")
    print(f"Final balance           : {stats['buy_hold_final_balance']:.2f}")
    print(f"Max drawdown            : {stats['buy_hold_max_drawdown_pct']:.2f}%")
    print(f"Sharpe                  : {stats['buy_hold_sharpe']:.3f}")

    print("=" * 80)

    print("PPO SEQUENCE MONTE CARLO")
    print("-" * 80)
    print(f"Mean return             : {stats['mean_return_pct']:+.2f}%")
    print(f"Median return           : {stats['median_return_pct']:+.2f}%")
    print(f"Std return              : {stats['std_return_pct']:.3f}%")
    print(f"Min return              : {stats['min_return_pct']:+.2f}%")
    print(f"Max return              : {stats['max_return_pct']:+.2f}%")
    print(f"Mean final balance      : {stats['mean_final_balance']:.2f}")
    print(f"Min final balance       : {stats['min_final_balance']:.2f}")
    print(f"Max final balance       : {stats['max_final_balance']:.2f}")
    print(f"Mean edge vs B&H        : {stats['mean_edge_vs_buy_hold_pct']:+.2f}%")
    print(f"Worst edge vs B&H       : {stats['min_edge_vs_buy_hold_pct']:+.2f}%")
    print(f"Best edge vs B&H        : {stats['max_edge_vs_buy_hold_pct']:+.2f}%")
    print(
        f"Beat Buy&Hold           : "
        f"{stats['beat_buy_hold_count']}/{stats['runs']} "
        f"({stats['beat_buy_hold_rate_pct']:.2f}%)"
    )
    print(f"Mean max drawdown       : {stats['mean_max_drawdown_pct']:.2f}%")
    print(f"Worst max drawdown      : {stats['worst_max_drawdown_pct']:.2f}%")
    print(f"Mean Sharpe             : {stats['mean_sharpe']:.3f}")
    print(f"Mean trades             : {stats['mean_trades']:.2f}")
    print(f"Mean win rate           : {stats['mean_win_rate_pct']:.2f}%")
    print(f"Mean profit factor      : {stats['mean_profit_factor']:.3f}")
    print(
        "Mean actions            : "
        f"Hold={stats['mean_action_hold']:.2f}, "
        f"Buy/CloseSell={stats['mean_action_buy_or_close_sell']:.2f}, "
        f"Sell/CloseBuy={stats['mean_action_sell_or_close_buy']:.2f}"
    )

    print("=" * 80)

    if stats["beat_buy_hold_rate_pct"] >= 80 and stats["mean_edge_vs_buy_hold_pct"] > 0:
        print("KESIMPULAN              : Model cukup robust terhadap variasi slippage.")
    elif stats["mean_edge_vs_buy_hold_pct"] > 0:
        print("KESIMPULAN              : Model menang rata-rata, tapi perlu validasi lanjut.")
    else:
        print("KESIMPULAN              : Model belum robust, perlu tuning/validasi ulang.")

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

    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)

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

    agent, input_shape, action_dim, config = load_agent_from_checkpoint(
        model_path=model_path,
        fallback_encoder=args.encoder,
        fallback_window_size=args.window_size,
    )

    buy_hold_report = evaluate_buy_and_hold(
        test_df=test_df,
        initial_balance=1000.0,
    )

    print("=" * 80)
    print("START MONTE CARLO EVALUATION")
    print("=" * 80)
    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")
    print(f"Model      : {model_path}")
    print(f"Encoder    : {config.encoder_type}")
    print(f"Input shape: {input_shape}")
    print(f"Runs       : {args.runs}")
    print("=" * 80)

    results = []

    for i in range(args.runs):
        run_id = i + 1
        seed = args.seed_start + i

        result = evaluate_one_run(
            test_df=test_df,
            agent=agent,
            input_shape=input_shape,
            action_dim=action_dim,
            run_id=run_id,
            seed=seed,
        )

        results.append(result)

        print(
            f"Run {run_id:03d}/{args.runs}"
            f" | Seed: {seed}"
            f" | Return: {result['total_return_pct']:+7.2f}%"
            f" | Final: {result['final_balance']:,.2f}"
            f" | DD: {result['max_drawdown_pct']:6.2f}%"
            f" | Trades: {result['total_trades']:3d}"
            f" | WR: {result['win_rate_pct']:5.1f}%"
            f" | PF: {result['profit_factor']:.3f}"
            f" | Actions: H={result['action_hold']}, "
            f"B={result['action_buy_or_close_sell']}, "
            f"S={result['action_sell_or_close_buy']}"
        )

    results_df = pd.DataFrame(results)

    stats = summarize_results(
        results_df=results_df,
        buy_hold_report=buy_hold_report,
    )

    print_stats(
        stats=stats,
        model_path=model_path,
        encoder=config.encoder_type,
        input_shape=input_shape,
    )

    safe_encoder = config.encoder_type

    runs_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_monte_carlo_runs.csv",
    )

    stats_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_monte_carlo_stats.csv",
    )

    hist_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_monte_carlo_return_distribution.png",
    )

    line_path = os.path.join(
        REPORT_DIR,
        f"{safe_encoder}_monte_carlo_returns_by_run.png",
    )

    results_df.to_csv(runs_path, index=False)
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    plot_return_distribution(
        results_df=results_df,
        buy_hold_return=buy_hold_report["total_return_pct"],
        output_path=hist_path,
    )

    plot_returns_by_run(
        results_df=results_df,
        buy_hold_return=buy_hold_report["total_return_pct"],
        output_path=line_path,
    )

    print("FILES SAVED")
    print("-" * 80)
    print(f"Runs detail             : {runs_path}")
    print(f"Stats summary           : {stats_path}")
    print(f"Return distribution     : {hist_path}")
    print(f"Returns by run          : {line_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()