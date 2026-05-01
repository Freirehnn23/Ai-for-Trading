import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


class ForexEnv(gym.Env):
    """
    Environment Week 1 — sederhana, gymnasium-compatible.
    State  : [harga_sekarang, harga_sebelumnya]
    Action : 0=hold, 1=buy, 2=sell
    Reward : profit/loss saat close posisi
    """

    def __init__(self, df):
        super().__init__()

        # Terima DataFrame langsung (bukan path)
        self.prices = df["close"].values.astype(np.float32)

        # Normalisasi relatif ke harga pertama
        self.prices = self.prices / self.prices[0]

        self.n_steps = len(self.prices)

        # Wajib ada di gymnasium.Env
        self.action_space = spaces.Discrete(3)  # hold, buy, sell
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  = np.inf,
            shape = (2,),           # [harga_kini, harga_sebelumnya]
            dtype = np.float32
        )

        # State internal
        self.current_step = 1
        self.position     = 0
        self.entry_price  = 0.0
        self.total_profit = 0.0

    # ── Reset (wajib return obs, info) ────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 1
        self.position     = 0
        self.entry_price  = 0.0
        self.total_profit = 0.0

        return self._get_obs(), {}

    # ── Step (wajib return obs, reward, terminated, truncated, info) ──
    def step(self, action):
        current_price = self.prices[self.current_step]
        reward        = 0.0

        if action == 1:   # BUY
            if self.position == 0:
                self.position    = 1
                self.entry_price = current_price

        elif action == 2:  # SELL
            if self.position == 1:
                reward             = float(current_price - self.entry_price)
                self.total_profit += reward
                self.position      = 0
                self.entry_price   = 0.0

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        info = {
            "price"   : float(current_price),
            "position": self.position,
            "profit"  : self.total_profit,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Obs ───────────────────────────────────────────────────────────
    def _get_obs(self):
        return np.array([
            self.prices[self.current_step],
            self.prices[self.current_step - 1]
        ], dtype=np.float32)