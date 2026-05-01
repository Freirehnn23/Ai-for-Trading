"""
Universal training script — Week 2, 3, 4
Ganti WEEK untuk switch environment.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle, importlib
from utils.data_loader import load_forex_data
from agent.dqn_agent import DQNAgent

# ── CONFIG (ubah WEEK untuk switch) ──────────────────────────────────
WEEK        = 4        # 2, 3, atau 4
EPISODES    = 500
PRINT_EVERY = 25
TRAIN_EVERY = 4
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/Data_historis(23-26).csv")
# ─────────────────────────────────────────────────────────────────────

ENV_MAP = {
    2: ("env.forex_env_week2", "ForexEnvWeek2"),
    3: ("env.forex_env_week3", "ForexEnvWeek3"),
    4: ("env.forex_env_week4", "ForexEnvWeek4"),
}
module_name, class_name = ENV_MAP[WEEK]
Env = getattr(importlib.import_module(module_name), class_name)
print(f">> Environment: {class_name}  (Week {WEEK})")

df    = load_forex_data(DATA_PATH)
env   = Env(df)
agent = DQNAgent(
    state_size      = env.observation_space.shape[0],
    action_size     = 3,
    epsilon         = 1.0,
    epsilon_min     = 0.05,
    epsilon_decay   = 0.98,
    buffer_capacity = 20_000,
    batch_size      = 64,
    lr              = 5e-4,
)

print("=" * 55)
print(f"  State sz : {env.observation_space.shape[0]}")
print(f"  Episodes : {EPISODES}")
print("=" * 55)

rewards_history = []
for ep in range(1, EPISODES + 1):
    obs, _ = env.reset()
    total_reward, steps, done = 0.0, 0, False
    while not done:
        action = agent.act(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs; total_reward += reward; steps += 1
        if steps % TRAIN_EVERY == 0:
            agent.train()
    agent.decay_epsilon()
    rewards_history.append(float(total_reward))

    if ep % PRINT_EVERY == 0 or ep == 1:
        avg = sum(rewards_history[-PRINT_EVERY:]) / min(len(rewards_history), PRINT_EVERY)
        print(
            f"  Ep {ep:4d}/{EPISODES}"
            f"  |  Reward: {total_reward:+8.2f}"
            f"  |  Avg: {avg:+8.2f}"
            f"  |  Eps: {agent.epsilon:.3f}"
            f"  |  Balance: ${env.balance:.0f}"
        )

print(f"\n  Training selesai!")

out = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(f"{out}/visualize", exist_ok=True)
os.makedirs(f"{out}/models", exist_ok=True)

with open(f"{out}/visualize/rewards_history.json", "w") as f:
    json.dump(rewards_history, f)

with open(f"{out}/models/trained_agent_week{WEEK}.pkl", "wb") as f:
    pickle.dump({"weights": agent.model.get_weights(), "week": WEEK}, f)

print(f"  Model → models/trained_agent_week{WEEK}.pkl")