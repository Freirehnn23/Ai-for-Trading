"""
WEEK 3 — Risk Management
Day 1: risk_per_trade  → dynamic lot sizing (risk 1% balance per trade)
Day 2: stop_loss_pct   → auto-close when loss hits limit
Day 3: take_profit_pct → auto-close at profit target (RR 1:2)
Day 4: get_stats()     → max_drawdown, sl_hits, tp_hits
Day 5: clean refactor  → unrealized P&L in observation
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForexEnvWeek3(gym.Env):

    def __init__(
        self, df,
        spread          = 0.20,
        slippage        = 0.05,
        commission      = 0.50,
        contract_size   = 100,
        risk_per_trade  = 0.01,   # 1% of balance risked per trade
        sl_pct          = 0.010,  # stop loss: 1% below entry
        tp_pct          = 0.020,  # take profit: 2% above entry (RR 1:2)
        initial_balance = 1000.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.prices = self.df["close"].values.astype(np.float64)
        self.n_steps = len(self.prices)

        self.spread = spread; self.slippage = slippage
        self.commission = commission; self.contract_size = contract_size
        self.risk_per_trade = risk_per_trade
        self.sl_pct = sl_pct; self.tp_pct = tp_pct
        self.initial_balance = initial_balance

        # Shape (5,): price, prev_price, position, balance, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    def _reset_state(self):
        self.balance = self.initial_balance
        self.current_step = 1
        self.position = 0; self.entry_price = 0.0
        self.lot_size = 0.01; self.sl_price = 0.0; self.tp_price = 0.0
        self.trade_history = []; self.balance_history = [self.initial_balance]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _calc_lot(self, entry):
        # DAY 1: lot sized so SL hit = risk_per_trade% of balance
        risk_amt = self.balance * self.risk_per_trade
        risk_per_lot = self.sl_pct * entry * self.contract_size
        lot = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.01
        return float(np.clip(lot, 0.001, 1.0))

    def step(self, action):
        raw = self.prices[self.current_step]; reward = 0.0
        slipped = raw + np.random.uniform(-self.slippage, self.slippage)
        ask = slipped + self.spread / 2
        bid = slipped - self.spread / 2

        # DAY 2 & 3: SL / TP check (priority over manual action)
        force = None
        if self.position == 1:
            if bid <= self.sl_price: force = ("SL", self.sl_price)
            elif bid >= self.tp_price: force = ("TP", self.tp_price)
        elif self.position == -1:
            if ask >= self.sl_price: force = ("SL", self.sl_price)
            elif ask <= self.tp_price: force = ("TP", self.tp_price)

        if force:
            tag, px = force
            raw_p = (px - self.entry_price if self.position == 1
                     else self.entry_price - px) * self.contract_size * self.lot_size
            profit = raw_p - self.commission
            self.balance += profit; reward = profit
            self.position = 0; self.entry_price = 0.0
            self.trade_history.append({"profit": profit, "type": tag})

        elif action == 1 and self.position == 0:      # OPEN BUY
            self.lot_size = self._calc_lot(ask)
            self.entry_price = ask; self.position = 1
            self.sl_price = ask * (1 - self.sl_pct)
            self.tp_price = ask * (1 + self.tp_pct)
            self.balance -= self.commission

        elif action == 2 and self.position == 0:      # OPEN SELL
            self.lot_size = self._calc_lot(bid)
            self.entry_price = bid; self.position = -1
            self.sl_price = bid * (1 + self.sl_pct)
            self.tp_price = bid * (1 - self.tp_pct)
            self.balance -= self.commission

        elif action == 2 and self.position == 1:      # CLOSE BUY
            p = (bid - self.entry_price) * self.contract_size * self.lot_size - self.commission
            self.balance += p; reward = p; self.position = 0; self.entry_price = 0.0
            self.trade_history.append({"profit": p, "type": "manual"})

        elif action == 1 and self.position == -1:     # CLOSE SELL
            p = (self.entry_price - ask) * self.contract_size * self.lot_size - self.commission
            self.balance += p; reward = p; self.position = 0; self.entry_price = 0.0
            self.trade_history.append({"profit": p, "type": "manual"})

        else:  # HOLD — small unrealized signal
            if self.position == 1:
                reward = (bid - self.entry_price) * self.contract_size * self.lot_size * 0.01
            elif self.position == -1:
                reward = (self.entry_price - ask) * self.contract_size * self.lot_size * 0.01

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1 or self.balance <= 0
        self.balance_history.append(self.balance)
        return self._get_obs(), reward, terminated, False, {
            "balance": round(self.balance, 2),
            "sl": round(self.sl_price, 2), "tp": round(self.tp_price, 2),
        }

    def _get_obs(self):
        i = self.current_step; base = self.prices[0]; price = self.prices[i]
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (price - self.entry_price) / self.entry_price
        return np.array([
            price / base, self.prices[i-1] / base,
            float(self.position), self.balance / self.initial_balance,
            float(np.clip(unrealized, -0.1, 0.1)),
        ], dtype=np.float32)

    def get_stats(self):
        trades = self.trade_history
        if not trades: return {"error": "Tidak ada trade"}
        profits = [t["profit"] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        bal = self.balance_history; peak = bal[0]; max_dd = 0.0
        for b in bal:
            if b > peak: peak = b
            dd = (peak - b) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
        sl_n = sum(1 for t in trades if t["type"] == "SL")
        tp_n = sum(1 for t in trades if t["type"] == "TP")
        return {
            "total_trades" : len(trades),
            "win_rate"     : f"{len(wins)/len(trades)*100:.1f}%",
            "avg_profit"   : f"${np.mean(wins):.2f}" if wins else "$0",
            "avg_loss"     : f"${np.mean(losses):.2f}" if losses else "$0",
            "total_profit" : f"${sum(profits):.2f}",
            "final_balance": f"${self.balance:.2f}",
            "max_drawdown" : f"{max_dd*100:.1f}%",
            "sl_hits"      : f"{sl_n} ({sl_n/len(trades)*100:.0f}%)",
            "tp_hits"      : f"{tp_n} ({tp_n/len(trades)*100:.0f}%)",
            "manual_close" : len(trades) - sl_n - tp_n,
        }