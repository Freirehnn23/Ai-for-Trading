import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_forex_data
from env.forex_env_pro_4 import ForexEnvWeek4
from agent.ppo_agent import PPOAgent


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Data_historis(23-26).csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualize")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def collect_rollout(env, agent, rollout_steps):
    obs, _ = env.reset()

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    total_reward_raw = 0.0

    for _ in range(rollout_steps):
        action, log_prob, value = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        reward_scaled = reward / env.initial_balance

        states.append(obs)
        actions.append(action)
        rewards.append(reward_scaled)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        total_reward_raw += reward
        obs = next_obs

        if done:
            obs, _ = env.reset()

    _, _, last_value = agent.select_action(obs, deterministic=True)

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

    return rollout, total_reward_raw


def evaluate(env, agent):
    obs, _ = env.reset()
    done = False

    action_count = [0 for _ in range(env.action_space.n)]

    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_count[action] += 1

    trades = getattr(env, "trade_history", [])

    return {
        "final_balance": float(env.balance),
        "total_return": float((env.balance - env.initial_balance) / env.initial_balance),
        "total_trades": len(trades),
        "action_count": action_count,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)

    args = parser.parse_args()

    df = load_forex_data(args.data_path)

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    train_env = ForexEnvWeek4(train_df)
    test_env = ForexEnvWeek4(test_df)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
    )

    print("=" * 80)
    print("PPO Actor-Critic PyTorch - ForexEnvWeek4")
    print("=" * 80)
    print(f"Data path      : {args.data_path}")
    print(f"Rows total     : {len(df)}")
    print(f"Train rows     : {len(train_df)}")
    print(f"Test rows      : {len(test_df)}")
    print(f"State dim      : {state_dim}")
    print(f"Action dim     : {action_dim}")
    print(f"Episodes       : {args.episodes}")
    print(f"Rollout steps  : {args.rollout_steps}")
    print("=" * 80)

    history = []
    best_test_balance = -np.inf

    for episode in range(1, args.episodes + 1):
        rollout, train_reward_raw = collect_rollout(
            env=train_env,
            agent=agent,
            rollout_steps=args.rollout_steps,
        )

        losses = agent.update(rollout)

        if episode % args.print_every == 0 or episode == 1:
            eval_report = evaluate(test_env, agent)

            row = {
                "episode": episode,
                "train_reward_raw": float(train_reward_raw),
                "eval_final_balance": eval_report["final_balance"],
                "eval_total_return": eval_report["total_return"],
                "eval_total_trades": eval_report["total_trades"],
                "action_count": eval_report["action_count"],
                **losses,
            }

            history.append(row)

            print(
                f"Episode {episode:04d}/{args.episodes}"
                f" | TrainRewardRaw: {train_reward_raw:+.2f}"
                f" | TestBalance: {eval_report['final_balance']:.2f}"
                f" | TestReturn: {eval_report['total_return'] * 100:+.2f}%"
                f" | Trades: {eval_report['total_trades']}"
                f" | Actions: {eval_report['action_count']}"
                f" | PiLoss: {losses['policy_loss']:+.5f}"
                f" | VLoss: {losses['value_loss']:.5f}"
                f" | Entropy: {losses['entropy']:.5f}"
            )

            if eval_report["final_balance"] > best_test_balance:
                best_test_balance = eval_report["final_balance"]

                best_model_path = os.path.join(
                    MODEL_DIR,
                    "ppo_actor_critic_week4_best.pt",
                )

                agent.save(best_model_path)

    model_path = os.path.join(MODEL_DIR, "ppo_actor_critic_week4.pt")
    history_path = os.path.join(VIS_DIR, "ppo_training_history.json")

    agent.save(model_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 80)
    print(f"Final model saved : {model_path}")
    print(f"Best model saved  : {os.path.join(MODEL_DIR, 'ppo_actor_critic_week4_best.pt')}")
    print(f"History saved     : {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()