import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.forex_env_pro import ForexEnv
from utils.data_loader import load_forex_data

def main():
    df = load_forex_data("data/forex.csv")

    env = ForexEnv(df)

    model = PPO.load("models/ppo_forex")

    obs, _ = env.reset()

    balances = []

    for _ in range(len(df) - 1):
        action, _ = model.predict(obs)

        obs, reward, done, _, info = env.step(action)

        balances.append(info["balance"])

        if done:
            break

    print("FINAL BALANCE:", balances[-1])

    plt.plot(balances)
    plt.title("Balance Over Time")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.show()


if __name__ == "__main__":
    main()