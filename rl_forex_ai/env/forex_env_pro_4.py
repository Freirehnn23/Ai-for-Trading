"""
WEEK 4 — Feature Engineering
Day 1: RSI          → observation feature (normalized [-1,1])
Day 2: Moving Avg   → fast/slow cross signal
Day 3: Support & Resistance → rolling min/max detection
Day 4: Distance to S/R      → how far price is from key levels
Day 5: All features normalized to stable [-1, 1] range
Observation shape: (9,)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.forex_env_pro_3 import ForexEnvWeek3


class ForexEnvWeek4(ForexEnvWeek3):

    def __init__(
        self, df,
        spread=0.20, slippage=0.05, commission=0.50,
        contract_size=100, risk_per_trade=0.01,
        sl_pct=0.010, tp_pct=0.020, initial_balance=1000.0,
        # Week 4 params
        rsi_period  = 14,
        ma_fast     = 10,
        ma_slow     = 20,
        sr_lookback = 20,
    ):
        self.rsi_period  = rsi_period
        self.ma_fast_n   = ma_fast
        self.ma_slow_n   = ma_slow
        self.sr_lookback = sr_lookback

        super().__init__(
            df=df, spread=spread, slippage=slippage, commission=commission,
            contract_size=contract_size, risk_per_trade=risk_per_trade,
            sl_pct=sl_pct, tp_pct=tp_pct, initial_balance=initial_balance,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self._precompute()
        self.current_step = max(self.ma_slow_n, self.rsi_period) + 1

    def _precompute(self):
        p = self.prices; n = len(p)

        # RSI (Wilder smoothing)
        delta = np.diff(p, prepend=p[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        rsi   = np.full(n, 50.0)
        ag    = np.mean(gain[1:self.rsi_period+1])
        al    = np.mean(loss[1:self.rsi_period+1])
        for i in range(self.rsi_period, n):
            if i > self.rsi_period:
                ag = (ag * (self.rsi_period-1) + gain[i]) / self.rsi_period
                al = (al * (self.rsi_period-1) + loss[i]) / self.rsi_period
            rs       = ag / al if al > 0 else 100
            rsi[i]   = 100 - 100 / (1 + rs)
        self.rsi = rsi

        # Moving averages (vectorized cumsum)
        def rollmean(arr, w):
            out = np.full(n, np.nan)
            cs  = np.cumsum(arr)
            out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-(w)]])) / w
            return out

        self.ma_fast_vals = rollmean(p, self.ma_fast_n)
        self.ma_slow_vals = rollmean(p, self.ma_slow_n)

    def _reset_state(self):
        super()._reset_state()
        start = max(getattr(self, 'ma_slow_n', 20), getattr(self, 'rsi_period', 14)) + 1
        self.current_step = start

    def _get_sr(self, step):
        start = max(1, step - self.sr_lookback)
        win   = self.prices[start:step]
        return float(np.min(win)), float(np.max(win))

    def _get_obs(self):
        i     = self.current_step; base = self.prices[0]; price = self.prices[i]

        # Unrealized P&L (from Week 3)
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (price - self.entry_price) / self.entry_price

        # DAY 1: RSI normalized [0,100] → [-1, 1]
        rsi_norm = (self.rsi[i] - 50) / 50

        # DAY 2: MA cross signal → [-1, 1]
        mf = self.ma_fast_vals[i] if not np.isnan(self.ma_fast_vals[i]) else price
        ms = self.ma_slow_vals[i] if not np.isnan(self.ma_slow_vals[i]) else price
        ma_sig = np.clip((mf - ms) / base * 10, -1, 1)

        # DAY 3 & 4: Distance to support / resistance
        support, resistance = self._get_sr(i)
        dist_sup = float(np.clip((price - support) / base * 2, 0, 1))
        dist_res = float(np.clip((resistance - price) / base * 2, 0, 1))

        return np.array([
            price / base,
            self.prices[i-1] / base,
            float(self.position),
            self.balance / self.initial_balance,
            float(np.clip(unrealized, -0.1, 0.1)),
            float(np.clip(rsi_norm, -1, 1)),       # RSI
            float(np.clip(ma_sig, -1, 1)),          # MA cross
            dist_sup,                               # distance to support
            dist_res,                               # distance to resistance
        ], dtype=np.float32)