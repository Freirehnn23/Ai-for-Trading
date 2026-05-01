"""
=============================================================
WEEK 2 — Environment Realistis
=============================================================
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForexEnvWeek2(gym.Env):

    def __init__(
        self,
        df,
        spread          = 0.20,
        slippage        = 0.05,
        commission      = 0.50,
        leverage        = 100,
        lot_size        = 0.01,
        contract_size   = 100,
        initial_balance = 1000.0,
    ):
        super().__init__()

        self.df     = df.reset_index(drop=True)
        self.prices = self.df["close"].values.astype(np.float64)
        self.n_steps = len(self.prices)

        self.spread        = spread
        self.slippage      = slippage
        self.commission    = commission
        self.leverage      = leverage
        self.lot_size      = lot_size
        self.contract_size = contract_size

        self.initial_balance = initial_balance
        self.balance         = initial_balance

        self.current_step = 1
        self.position     = 0
        self.entry_price  = 0.0

        self.trade_history   = []
        self.balance_history = [initial_balance]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = 1
        self.position        = 0
        self.entry_price     = 0.0
        self.balance         = self.initial_balance
        self.trade_history   = []
        self.balance_history = [self.initial_balance]
        return self._get_obs(), {}

    def step(self, action):
        raw_price = self.prices[self.current_step]
        reward    = 0.0

        slipped_price = raw_price + np.random.uniform(
            -self.slippage, self.slippage
        )
        ask_price = slipped_price + self.spread / 2
        bid_price = slipped_price - self.spread / 2

        if action == 1 and self.position == 0:        # OPEN BUY
            self.position    = 1
            self.entry_price = ask_price
            self.balance    -= self.commission

        elif action == 2 and self.position == 0:      # OPEN SELL
            self.position    = -1
            self.entry_price = bid_price
            self.balance    -= self.commission

        elif action == 2 and self.position == 1:      # CLOSE BUY
            raw_profit = (
                (bid_price - self.entry_price)
                * self.contract_size
                * self.lot_size
            )
            profit = raw_profit - self.commission
            self.balance  += profit
            reward         = profit
            self.position  = 0
            self.entry_price = 0.0
            self.trade_history.append(profit)

        elif action == 1 and self.position == -1:     # CLOSE SELL
            raw_profit = (
                (self.entry_price - ask_price)
                * self.contract_size
                * self.lot_size
            )
            profit = raw_profit - self.commission
            self.balance  += profit
            reward         = profit
            self.position  = 0
            self.entry_price = 0.0
            self.trade_history.append(profit)

        else:  # HOLD
            # === FIX: Unrealized P&L sebagai reward kecil ===
            # Agent dapat sinyal real-time apakah posisinya untung/rugi
            if self.position == 1:
                unrealized = (
                    (bid_price - self.entry_price)
                    * self.contract_size * self.lot_size
                )
                reward = unrealized * 0.01  # skala kecil, hanya sinyal
            elif self.position == -1:
                unrealized = (
                    (self.entry_price - ask_price)
                    * self.contract_size * self.lot_size
                )
                reward = unrealized * 0.01

        self.current_step += 1
        terminated = (
            self.current_step >= self.n_steps - 1
            or self.balance <= 0
        )
        truncated = False

        self.balance_history.append(self.balance)

        info = {
            "step"    : self.current_step,
            "ask"     : round(ask_price, 5),
            "bid"     : round(bid_price, 5),
            "position": self.position,
            "balance" : round(self.balance, 2),
            "reward"  : round(reward, 4),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        price_now  = self.prices[self.current_step]
        price_prev = self.prices[self.current_step - 1]
        base       = self.prices[0]
        return np.array([
            price_now  / base,
            price_prev / base,
            float(self.position),
            self.balance / self.initial_balance,
        ], dtype=np.float32)

    def get_stats(self):
        trades = self.trade_history
        if not trades:
            return {"error": "Tidak ada trade yang terjadi"}
        wins     = [t for t in trades if t > 0]
        losses   = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100
        return {
            "total_trades" : len(trades),
            "win_rate"     : f"{win_rate:.1f}%",
            "avg_profit"   : f"${np.mean(wins):.2f}"   if wins   else "$0",
            "avg_loss"     : f"${np.mean(losses):.2f}" if losses else "$0",
            "total_profit" : f"${sum(trades):.2f}",
            "final_balance": f"${self.balance:.2f}",
            "max_balance"  : f"${max(self.balance_history):.2f}",
            "min_balance"  : f"${min(self.balance_history):.2f}",
        }