import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from env.forex_env_pro import ForexEnv
from agent.dqn_agent import DQNAgent
from utils.data_loader import load_forex_data

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
EPISODES    = 100
PRINT_EVERY = 10
TRAIN_EVERY = 4

# ── Init ──────────────────────────────────────────────────────────────
df    = load_forex_data(DATA_PATH)
env   = ForexEnv(df)
agent = DQNAgent(
    state_size     = 2,
    action_size    = 3,
    epsilon_decay  = 0.9995,  # lebih lambat (dari 0.995)
    buffer_capacity= 20_000,  # lebih besar (dari 10_000)
)

EPISODES    = 300   # lebih banyak (dari 100)
TRAIN_EVERY = 2     # lebih sering train (dari 4)

print("=" * 55)
print("  WEEK 1 — TRAINING DQN AGENT")
print("=" * 55)
print(f"  Data    : {len(df)} baris")
print(f"  Episodes: {EPISODES}")
print("=" * 55)

rewards_history = []

# ── Training Loop ─────────────────────────────────────────────────────
for episode in range(1, EPISODES + 1):
    obs, _ = env.reset()
    total_reward = 0.0
    step_count   = 0
    done         = False

    while not done:
        action                          = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        obs           = next_obs
        total_reward += reward
        step_count   += 1

        if step_count % TRAIN_EVERY == 0:
            agent.train()

    rewards_history.append(float(total_reward))

    if episode % PRINT_EVERY == 0 or episode == 1:
        avg = sum(rewards_history[-PRINT_EVERY:]) / min(len(rewards_history), PRINT_EVERY)
        print(
            f"  Episode {episode:4d}/{EPISODES}"
            f"  |  Reward: {total_reward:+8.4f}"
            f"  |  Avg: {avg:+8.4f}"
            f"  |  Epsilon: {agent.epsilon:.3f}"
        )

print(f"\n  Training selesai!")
print(f"  Reward terbaik: {max(rewards_history):+.4f}  (ep. {rewards_history.index(max(rewards_history))+1})")

# ── Simpan rewards ────────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), "../visualize"), exist_ok=True)
rewards_path = os.path.join(os.path.dirname(__file__), "../visualize/rewards_history.json")
with open(rewards_path, "w") as f:
    json.dump(rewards_history, f)

# ── Simpan model ──────────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), "../models"), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_agent.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"weights": agent.model.get_weights()}, f)

print(f"  Model disimpan  → {model_path}")
print(f"  Rewards disimpan → {rewards_path}")