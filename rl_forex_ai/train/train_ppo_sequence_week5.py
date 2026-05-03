"""
Training PPO Sequence Agent — WEEK5 Environment + ACTION MASKING.

File ini menggantikan versi Week5 sebelumnya.
Jangan hapus train/train_ppo_sequence.py lama karena itu baseline Week4.

Perbedaan penting:
1. Pakai env.forex_env_pro_5.ForexEnvWeek5.
2. Action space 4 action:
   0 HOLD, 1 OPEN_BUY, 2 OPEN_SELL, 3 CLOSE_POSITION.
3. Menggunakan ACTION MASKING:
   - posisi flat  : valid HOLD, OPEN_BUY, OPEN_SELL
   - posisi aktif : valid HOLD, CLOSE_POSITION
4. Split data menjadi train/validation/test.
5. Best model dipilih dari VALIDATION dengan skor yang menolak no-trade model.
6. Test set hanya dipakai sekali di akhir training.

Run dari folder rl_forex_ai:

Smoke test:
    python train/train_ppo_sequence_week5.py --encoder cnn1d --episodes 2 --rollout-steps 64 --print-every 1 --update-epochs 2 --batch-size 32

Training utama:
    python train/train_ppo_sequence_week5.py --encoder cnn1d --episodes 300 --rollout-steps 512 --print-every 10 --update-epochs 5 --batch-size 64 --entropy-coef 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo_sequence_agent import PPOSequenceAgent, PPOSequenceConfig
from env.forex_env_pro_5 import ForexEnvWeek5
from env.windowed_env import WindowedObservationWrapper
from utils.data_loader import load_forex_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sequence", "week5")
VIS_DIR = os.path.join(BASE_DIR, "visualize", "sequence", "week5")


ACTION_NAME = {
    0: "HOLD",
    1: "OPEN_BUY",
    2: "OPEN_SELL",
    3: "CLOSE_POSITION",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_val_test_split(df, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if len(train_df) < 100 or len(val_df) < 20 or len(test_df) < 20:
        raise ValueError(
            f"Dataset terlalu kecil untuk split train/val/test. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    return train_df, val_df, test_df


def make_env(df, window_size: int, args):
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

    env = WindowedObservationWrapper(
        base_env,
        window_size=window_size,
    )

    return env


def get_action_mask(env):
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return None


def collect_rollout(
    env,
    agent: PPOSequenceAgent,
    rollout_steps: int,
    reward_scale: float,
) -> Tuple[Dict, float, int]:
    obs, _ = env.reset()

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    action_masks = []

    total_raw_reward = 0.0
    completed_episodes = 0

    for _ in range(rollout_steps):
        mask = get_action_mask(env)

        action, log_prob, value = agent.select_action(
            obs,
            deterministic=False,
            action_mask=mask,
        )

        next_obs, raw_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        scaled_reward = float(raw_reward) / reward_scale

        states.append(obs.copy())
        actions.append(action)
        rewards.append(scaled_reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)
        action_masks.append(np.asarray(mask, dtype=bool).copy())

        total_raw_reward += float(raw_reward)
        obs = next_obs

        if done:
            completed_episodes += 1
            obs, _ = env.reset()

    last_mask = get_action_mask(env)
    _, _, last_value = agent.select_action(
        obs,
        deterministic=True,
        action_mask=last_mask,
    )

    advantages, returns = agent.compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
    )

    rollout = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "log_probs": log_probs,
        "values": values,
        "action_masks": action_masks,
        "advantages": advantages,
        "returns": returns,
    }

    return rollout, total_raw_reward, completed_episodes


def calculate_max_drawdown(equity_history) -> float:
    equity = np.asarray(equity_history, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-8)
    return float(dd.min())


def evaluate(env, agent: PPOSequenceAgent, min_confidence: float | None = None) -> Dict:
    obs, _ = env.reset()
    done = False

    base_env = env.unwrapped

    action_count = {ACTION_NAME[i]: 0 for i in range(base_env.action_space.n)}
    total_reward = 0.0

    while not done:
        mask = get_action_mask(env)
        action, _, _ = agent.select_action(
            obs,
            deterministic=True,
            action_mask=mask,
            min_confidence=min_confidence,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        action_count[ACTION_NAME.get(action, f"ACTION_{action}")] += 1

    trades = getattr(base_env, "trade_history", [])
    profits = [float(t.get("profit", 0.0)) for t in trades if isinstance(t, dict)]

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    if profits:
        win_rate = len(wins) / len(profits)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = float(sum(wins))
        gross_loss = float(abs(sum(losses)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    equity_history = getattr(base_env, "equity_history", getattr(base_env, "balance_history", []))
    max_drawdown = calculate_max_drawdown(equity_history)

    final_equity = float(equity_history[-1]) if len(equity_history) else float(base_env.balance)

    return {
        "final_balance": float(base_env.balance),
        "final_equity": final_equity,
        "total_return": float((final_equity - base_env.initial_balance) / base_env.initial_balance),
        "total_reward": total_reward,
        "total_trades": len(profits),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "action_count": action_count,
    }


def validation_score(report: Dict) -> float:
    """
    Skor validation untuk memilih best model.

    Penting:
    - Model no-trade jangan dipilih sebagai best.
    - Skor tidak hanya return, tapi juga drawdown, PF, winrate, dan jumlah trade.
    """
    total_trades = int(report["total_trades"])

    if total_trades < 5:
        return -1e9 + float(report["final_equity"])

    ret_pct = float(report["total_return"] * 100.0)
    dd_pct = abs(float(report["max_drawdown"] * 100.0))
    wr_pct = float(report["win_rate"] * 100.0)
    pf = float(report["profit_factor"])
    if not np.isfinite(pf):
        pf = 5.0
    pf = min(pf, 5.0)

    return (
        ret_pct
        - 0.75 * dd_pct
        + 2.0 * pf
        + 0.05 * wr_pct
        + min(total_trades, 30) * 0.05
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--encoder", type=str, default="cnn1d", choices=["cnn1d", "lstm", "attention", "transformer"])
    parser.add_argument("--window-size", type=int, default=32)

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    # Environment params: dibuat lebih ringan dari patch Week5 awal agar tidak collapse ke no-trade.
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
    parser.add_argument("--min-confidence", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    df = load_forex_data(args.data_path)
    train_df, val_df, test_df = make_train_val_test_split(df, args.train_ratio, args.val_ratio)

    train_env = make_env(train_df, args.window_size, args)
    val_env = make_env(val_df, args.window_size, args)
    test_env = make_env(test_df, args.window_size, args)

    input_shape = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    config = PPOSequenceConfig(
        encoder_type=args.encoder,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    agent = PPOSequenceAgent(input_shape=input_shape, action_dim=action_dim, config=config)

    model_prefix = f"ppo_{args.encoder}_week5_w{args.window_size}"
    best_model_path = os.path.join(MODEL_DIR, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(MODEL_DIR, f"{model_prefix}_final.pt")
    history_path = os.path.join(VIS_DIR, f"{model_prefix}_history.json")

    print("=" * 80)
    print(f"PPO Sequence Agent — WEEK5 ACTION MASK — {args.encoder.upper()}")
    print("=" * 80)
    print(f"Device       : {agent.device}")
    print(f"Data path    : {args.data_path}")
    print(f"Rows total   : {len(df)}")
    print(f"Train rows   : {len(train_df)}")
    print(f"Val rows     : {len(val_df)}")
    print(f"Test rows    : {len(test_df)}")
    print(f"Input shape  : {input_shape}")
    print(f"Action dim   : {action_dim}")
    print(f"Window size  : {args.window_size}")
    print(f"Episodes     : {args.episodes}")
    print("Action mask  : ON")
    print("=" * 80)

    history = []
    best_val_score = -float("inf")

    for episode in range(1, args.episodes + 1):
        rollout, train_raw_reward, completed_episodes = collect_rollout(
            env=train_env,
            agent=agent,
            rollout_steps=args.rollout_steps,
            reward_scale=args.reward_scale,
        )

        losses = agent.update(rollout)

        should_print = episode == 1 or episode % args.print_every == 0

        if should_print:
            val_report = evaluate(val_env, agent, min_confidence=args.min_confidence)
            val_score = validation_score(val_report)

            row = {
                "episode": episode,
                "encoder": args.encoder,
                "train_raw_reward": float(train_raw_reward),
                "completed_episodes_in_rollout": completed_episodes,
                "val_score": float(val_score),
                **losses,
                **{f"val_{k}": v for k, v in val_report.items()},
            }
            history.append(row)

            if val_score > best_val_score:
                best_val_score = val_score
                agent.save(best_model_path)

            print(
                f"Ep {episode:04d}/{args.episodes}"
                f" | Encoder: {args.encoder}"
                f" | TrainR(raw): {train_raw_reward:+9.2f}"
                f" | ValEq: {val_report['final_equity']:,.2f}"
                f" | ValRet: {val_report['total_return'] * 100:+7.2f}%"
                f" | ValScore: {val_score:+8.2f}"
                f" | Trades: {val_report['total_trades']:4d}"
                f" | WR: {val_report['win_rate'] * 100:5.1f}%"
                f" | DD: {val_report['max_drawdown'] * 100:6.2f}%"
                f" | Actions: {val_report['action_count']}"
                f" | PiLoss: {losses['policy_loss']:+.4f}"
                f" | VLoss: {losses['value_loss']:.4f}"
                f" | Entropy: {losses['entropy']:.4f}"
            )

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, default=str)

    agent.save(final_model_path)

    print("=" * 80)
    print("Training WEEK5 ACTION MASK selesai")
    print(f"Final model : {final_model_path}")
    print(f"Best model  : {best_model_path}")
    print(f"Best score  : {best_val_score:.4f}")
    print(f"History     : {history_path}")

    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        test_report = evaluate(test_env, agent, min_confidence=args.min_confidence)
        print("=" * 80)
        print("FINAL TEST RESULT — BEST VALIDATION MODEL")
        print("-" * 80)
        print(f"Test equity       : {test_report['final_equity']:.2f}")
        print(f"Test return       : {test_report['total_return'] * 100:+.2f}%")
        print(f"Test max drawdown : {test_report['max_drawdown'] * 100:.2f}%")
        print(f"Test trades       : {test_report['total_trades']}")
        print(f"Test win rate     : {test_report['win_rate'] * 100:.2f}%")
        print(f"Profit factor     : {test_report['profit_factor']:.3f}")
        print(f"Actions           : {test_report['action_count']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
