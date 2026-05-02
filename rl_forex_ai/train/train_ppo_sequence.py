"""
Training PPO Sequence Agent.

Mendukung:
- CNN-1D PPO
- LSTM PPO
- Attention PPO
- Transformer PPO

Cara run dari folder rl_forex_ai:

Smoke test:
python train/train_ppo_sequence.py --encoder lstm --episodes 2 --rollout-steps 64 --print-every 1 --update-epochs 2 --batch-size 32

Training LSTM:
python train/train_ppo_sequence.py --encoder lstm --episodes 300 --rollout-steps 512 --print-every 10 --update-epochs 5 --batch-size 64 --entropy-coef 0.03

Training CNN:
python train/train_ppo_sequence.py --encoder cnn1d --episodes 300 --rollout-steps 512 --print-every 10 --update-epochs 5 --batch-size 64 --entropy-coef 0.03

Training Attention:
python train/train_ppo_sequence.py --encoder attention --episodes 300 --rollout-steps 512 --print-every 10 --update-epochs 5 --batch-size 64 --entropy-coef 0.03

Training Transformer:
python train/train_ppo_sequence.py --encoder transformer --episodes 300 --rollout-steps 512 --print-every 10 --update-epochs 5 --batch-size 64 --entropy-coef 0.03
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
from env.forex_env_pro_4 import ForexEnvWeek4
from env.windowed_env import WindowedObservationWrapper
from utils.data_loader import load_forex_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sequence")
VIS_DIR = os.path.join(BASE_DIR, "visualize", "sequence")


ACTION_NAME = {
    0: "HOLD",
    1: "BUY_OR_CLOSE_SELL",
    2: "SELL_OR_CLOSE_BUY",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_test_split(df, train_ratio: float = 0.80):
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    if len(train_df) < 50 or len(test_df) < 20:
        raise ValueError(
            f"Dataset terlalu kecil. train={len(train_df)}, test={len(test_df)}"
        )

    return train_df, test_df


def make_env(df, window_size: int):
    base_env = ForexEnvWeek4(df)
    env = WindowedObservationWrapper(
        base_env,
        window_size=window_size,
    )
    return env


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

    total_raw_reward = 0.0
    completed_episodes = 0

    for _ in range(rollout_steps):
        action, log_prob, value = agent.select_action(
            obs,
            deterministic=False,
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

        total_raw_reward += float(raw_reward)
        obs = next_obs

        if done:
            completed_episodes += 1
            obs, _ = env.reset()

    _, _, last_value = agent.select_action(
        obs,
        deterministic=True,
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
        "advantages": advantages,
        "returns": returns,
    }

    return rollout, total_raw_reward, completed_episodes


def evaluate(env, agent: PPOSequenceAgent) -> Dict:
    obs, _ = env.reset()
    done = False

    base_env = env.unwrapped

    action_count = {
        ACTION_NAME[i]: 0
        for i in range(base_env.action_space.n)
    }

    total_reward = 0.0

    while not done:
        action, _, _ = agent.select_action(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        action_count[ACTION_NAME[action]] += 1

    trades = getattr(base_env, "trade_history", [])

    profits = [
        float(t.get("profit", 0.0))
        for t in trades
        if isinstance(t, dict)
    ]

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    if len(profits) > 0:
        win_rate = len(wins) / len(profits)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        )
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    balance_history = np.asarray(
        getattr(base_env, "balance_history", []),
        dtype=np.float64,
    )

    if len(balance_history) > 0:
        running_peak = np.maximum.accumulate(balance_history)
        drawdown = (
            running_peak - balance_history
        ) / np.maximum(running_peak, 1e-8)

        max_drawdown = float(np.max(drawdown))
    else:
        max_drawdown = 0.0

    return {
        "final_balance": float(base_env.balance),
        "total_return": float(
            (base_env.balance - base_env.initial_balance)
            / base_env.initial_balance
        ),
        "total_reward": total_reward,
        "total_trades": len(profits),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "action_count": action_count,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)

    parser.add_argument(
        "--encoder",
        type=str,
        default="lstm",
        choices=["cnn1d", "lstm", "attention", "transformer"],
    )

    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=1000.0)

    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    df = load_forex_data(args.data_path)

    train_df, test_df = make_train_test_split(
        df,
        train_ratio=args.train_ratio,
    )

    train_env = make_env(
        train_df,
        window_size=args.window_size,
    )

    test_env = make_env(
        test_df,
        window_size=args.window_size,
    )

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

    agent = PPOSequenceAgent(
        input_shape=input_shape,
        action_dim=action_dim,
        config=config,
    )

    print("=" * 80)
    print(f"PPO Sequence Agent — {args.encoder.upper()}")
    print("=" * 80)
    print(f"Device       : {agent.device}")
    print(f"Data path    : {args.data_path}")
    print(f"Rows total   : {len(df)}")
    print(f"Train rows   : {len(train_df)}")
    print(f"Test rows    : {len(test_df)}")
    print(f"Input shape  : {input_shape}")
    print(f"Action dim   : {action_dim}")
    print(f"Window size  : {args.window_size}")
    print(f"Episodes     : {args.episodes}")
    print(f"Rollout steps: {args.rollout_steps}")
    print("=" * 80)

    history = []
    best_balance = -float("inf")

    model_prefix = f"ppo_{args.encoder}_w{args.window_size}"

    best_model_path = os.path.join(
        MODEL_DIR,
        f"{model_prefix}_best.pt",
    )

    final_model_path = os.path.join(
        MODEL_DIR,
        f"{model_prefix}_final.pt",
    )

    history_path = os.path.join(
        VIS_DIR,
        f"{model_prefix}_history.json",
    )

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
            eval_report = evaluate(test_env, agent)

            row = {
                "episode": episode,
                "encoder": args.encoder,
                "train_raw_reward": float(train_raw_reward),
                "completed_episodes_in_rollout": completed_episodes,
                **losses,
                **{f"eval_{k}": v for k, v in eval_report.items()},
            }

            history.append(row)

            if eval_report["final_balance"] > best_balance:
                best_balance = eval_report["final_balance"]
                agent.save(best_model_path)

            action_count = eval_report["action_count"]

            print(
                f"Ep {episode:04d}/{args.episodes}"
                f" | Encoder: {args.encoder}"
                f" | TrainR(raw): {train_raw_reward:+9.2f}"
                f" | TestBal: {eval_report['final_balance']:,.2f}"
                f" | TestRet: {eval_report['total_return'] * 100:+7.2f}%"
                f" | Trades: {eval_report['total_trades']:4d}"
                f" | WR: {eval_report['win_rate'] * 100:5.1f}%"
                f" | DD: {eval_report['max_drawdown'] * 100:5.1f}%"
                f" | Actions: {action_count}"
                f" | PiLoss: {losses['policy_loss']:+.4f}"
                f" | VLoss: {losses['value_loss']:.4f}"
                f" | Entropy: {losses['entropy']:.4f}"
            )

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    agent.save(final_model_path)

    print("=" * 80)
    print("Training sequence selesai")
    print(f"Encoder     : {args.encoder}")
    print(f"Final model : {final_model_path}")
    print(f"Best model  : {best_model_path}")
    print(f"History     : {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()