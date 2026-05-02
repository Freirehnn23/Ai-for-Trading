"""
WEEK 3 — Risk Management
Day 1: risk_per_trade  -> dynamic lot sizing (risk 1% balance per trade)
Day 2: stop_loss_pct   -> auto-close when loss hits limit
Day 3: take_profit_pct -> auto-close at profit target (RR 1:2)
Day 4: get_stats()     -> max_drawdown, sl_hits, tp_hits
Day 5: clean refactor  -> unrealized P&L in observation

MODIFIED:
- Added reward shaping to reduce overtrading.
- Added trade_penalty.
- Added action_penalty.
- Added hold_position_penalty.
- Fixed trade_history:
  open_buy/open_sell are NOT counted as trades anymore.
  trade_history only records closed trades:
  SL, TP, manual_close_buy, manual_close_sell.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForexEnvWeek3(gym.Env):

    def __init__(
        self,
        df,
        spread=0.20,
        slippage=0.05,
        commission=0.50,
        contract_size=100,
        risk_per_trade=0.01,
        sl_pct=0.010,
        tp_pct=0.020,
        initial_balance=1000.0,

        # Reward shaping tambahan
        trade_penalty=0.01,
        action_penalty=0.002,
        hold_position_penalty=0.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.prices = self.df["close"].values.astype(np.float64)
        self.n_steps = len(self.prices)

        self.spread = spread
        self.slippage = slippage
        self.commission = commission
        self.contract_size = contract_size
        self.risk_per_trade = risk_per_trade
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.initial_balance = initial_balance

        self.trade_penalty = trade_penalty
        self.action_penalty = action_penalty
        self.hold_position_penalty = hold_position_penalty

        # Shape (5,): price, prev_price, position, balance, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        # 0 = Hold
        # 1 = Buy / Close Sell
        # 2 = Sell / Close Buy
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.balance = self.initial_balance
        self.current_step = 1

        self.position = 0
        self.entry_price = 0.0

        self.lot_size = 0.01
        self.sl_price = 0.0
        self.tp_price = 0.0

        self.trade_history = []
        self.balance_history = [self.initial_balance]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _calc_lot(self, entry):
        risk_amt = self.balance * self.risk_per_trade
        risk_per_lot = self.sl_pct * entry * self.contract_size

        lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01

        return float(np.clip(lot, 0.001, 1.0))

    def step(self, action):
        raw = self.prices[self.current_step]
        reward = 0.0

        slipped = raw + np.random.uniform(-self.slippage, self.slippage)
        ask = slipped + self.spread / 2
        bid = slipped - self.spread / 2

        trade_executed = False
        force_close = False

        # ==============================
        # SL / TP check
        # ==============================
        force = None

        if self.position == 1:
            if bid <= self.sl_price:
                force = ("SL", self.sl_price)
            elif bid >= self.tp_price:
                force = ("TP", self.tp_price)

        elif self.position == -1:
            if ask >= self.sl_price:
                force = ("SL", self.sl_price)
            elif ask <= self.tp_price:
                force = ("TP", self.tp_price)

        # ==============================
        # Forced close by SL / TP
        # ==============================
        if force:
            tag, px = force

            raw_p = (
                (px - self.entry_price)
                if self.position == 1
                else (self.entry_price - px)
            ) * self.contract_size * self.lot_size

            profit = raw_p - self.commission

            self.balance += profit
            reward = profit

            self.position = 0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0

            self.trade_history.append({
                "profit": profit,
                "type": tag,
            })

            trade_executed = True
            force_close = True

        # ==============================
        # OPEN BUY
        # ==============================
        elif action == 1 and self.position == 0:
            self.lot_size = self._calc_lot(ask)

            self.entry_price = ask
            self.position = 1

            self.sl_price = ask * (1 - self.sl_pct)
            self.tp_price = ask * (1 + self.tp_pct)

            self.balance -= self.commission

            trade_executed = True

            # Penting:
            # Jangan append open_buy ke trade_history.
            # Entry bukan closed trade.
            # Trade baru dihitung saat SL, TP, atau manual close.

        # ==============================
        # OPEN SELL
        # ==============================
        elif action == 2 and self.position == 0:
            self.lot_size = self._calc_lot(bid)

            self.entry_price = bid
            self.position = -1

            self.sl_price = bid * (1 + self.sl_pct)
            self.tp_price = bid * (1 - self.tp_pct)

            self.balance -= self.commission

            trade_executed = True

            # Penting:
            # Jangan append open_sell ke trade_history.
            # Entry bukan closed trade.
            # Trade baru dihitung saat SL, TP, atau manual close.

        # ==============================
        # CLOSE BUY
        # ==============================
        elif action == 2 and self.position == 1:
            profit = (
                (bid - self.entry_price)
                * self.contract_size
                * self.lot_size
                - self.commission
            )

            self.balance += profit
            reward = profit

            self.position = 0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0

            self.trade_history.append({
                "profit": profit,
                "type": "manual_close_buy",
            })

            trade_executed = True

        # ==============================
        # CLOSE SELL
        # ==============================
        elif action == 1 and self.position == -1:
            profit = (
                (self.entry_price - ask)
                * self.contract_size
                * self.lot_size
                - self.commission
            )

            self.balance += profit
            reward = profit

            self.position = 0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0

            self.trade_history.append({
                "profit": profit,
                "type": "manual_close_sell",
            })

            trade_executed = True

        # ==============================
        # HOLD / NO VALID TRADE
        # ==============================
        else:
            if self.position == 1:
                reward = (
                    (bid - self.entry_price)
                    * self.contract_size
                    * self.lot_size
                    * 0.01
                )

            elif self.position == -1:
                reward = (
                    (self.entry_price - ask)
                    * self.contract_size
                    * self.lot_size
                    * 0.01
                )

        # ==============================
        # Reward shaping tambahan
        # ==============================

        # Penalti jika agent memilih action aktif Buy/Sell.
        # Tujuannya agar action 0/Hold punya peluang dipilih.
        if action in [1, 2]:
            reward -= self.action_penalty

        # Penalti tambahan jika benar-benar terjadi transaksi.
        # Ini mengurangi overtrade.
        if trade_executed and not force_close:
            reward -= self.trade_penalty

        # Penalti kecil kalau sedang punya posisi terbuka.
        # Default 0.0 agar agent tidak terlalu takut membuka posisi.
        if self.position != 0:
            reward -= self.hold_position_penalty

        self.current_step += 1

        terminated = self.current_step >= self.n_steps - 1 or self.balance <= 0
        truncated = False

        self.balance_history.append(self.balance)

        info = {
            "balance": round(self.balance, 2),
            "position": int(self.position),
            "entry_price": round(self.entry_price, 4),
            "sl": round(self.sl_price, 2),
            "tp": round(self.tp_price, 2),
            "trade_executed": trade_executed,
            "force_close": force_close,
            "reward": round(float(reward), 6),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        i = self.current_step
        base = self.prices[0]
        price = self.prices[i]

        unrealized = 0.0

        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (price - self.entry_price) / self.entry_price

        return np.array([
            price / base,
            self.prices[i - 1] / base,
            float(self.position),
            self.balance / self.initial_balance,
            float(np.clip(unrealized, -0.1, 0.1)),
        ], dtype=np.float32)

    def get_stats(self):
        trades = self.trade_history

        if not trades:
            return {"error": "Tidak ada trade"}

        # trade_history sekarang hanya berisi closed trade.
        profits = [t["profit"] for t in trades]

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        bal = self.balance_history
        peak = bal[0]
        max_dd = 0.0

        for b in bal:
            if b > peak:
                peak = b

            dd = (peak - b) / peak if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd

        sl_n = sum(1 for t in trades if t["type"] == "SL")
        tp_n = sum(1 for t in trades if t["type"] == "TP")
        manual_n = len(trades) - sl_n - tp_n

        return {
            "total_trades": len(trades),
            "win_rate": f"{len(wins) / len(trades) * 100:.1f}%",
            "avg_profit": f"${np.mean(wins):.2f}" if wins else "$0",
            "avg_loss": f"${np.mean(losses):.2f}" if losses else "$0",
            "total_profit": f"${sum(profits):.2f}",
            "final_balance": f"${self.balance:.2f}",
            "max_drawdown": f"{max_dd * 100:.1f}%",
            "sl_hits": f"{sl_n} ({sl_n / len(trades) * 100:.0f}%)",
            "tp_hits": f"{tp_n} ({tp_n / len(trades) * 100:.0f}%)",
            "manual_close": manual_n,
        }