import gymnasium as gym
import numpy as np

class ForexEnv(gym.Env):
    def __init__(self, df):
        super(ForexEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0

        # === BALANCE SYSTEM ===
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.position = 0  # 1 = buy, -1 = sell, 0 = none

        # === ACTION SPACE ===
        self.action_space = gym.spaces.Discrete(3)  # hold, buy, sell

        # === OBSERVATION SPACE ===
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def _get_obs(self):
        row = self.df.iloc[self.current_step]

        return np.array([
            row["close"],
            row["rsi"],
            self.balance
        ], dtype=np.float32)

    def step(self, action):
        done = False

        # === UPDATE POSITION ===
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1

        # === PRICE ===
        current_price = self.df.iloc[self.current_step]["close"]
        next_price = self.df.iloc[self.current_step + 1]["close"]

        price_change = next_price - current_price

        # === REWARD ===
        reward = self.position * price_change * 1000

        # === UPDATE BALANCE ===
        self.balance += reward

        # === NEXT STEP ===
        self.current_step += 1

        # === TERMINATION ===
        if self.current_step >= len(self.df) - 1:
            done = True

        if self.balance <= 0:
            done = True

        obs = self._get_obs()

        terminated = done
        truncated = False

        info = {
            "balance": self.balance,
            "position": self.position
        }

        return obs, reward, terminated, truncated, info