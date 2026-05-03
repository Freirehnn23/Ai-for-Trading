"""
WEEK 5 — Realistic Trading Environment

PERUBAHAN UTAMA DARI WEEK4/WEEK3:
1. Action space lebih jelas:
   0 = HOLD
   1 = OPEN_BUY
   2 = OPEN_SELL
   3 = CLOSE_POSITION

2. Action masking tersedia lewat get_action_mask().

3. Reward berbasis perubahan equity:
   reward = equity_now - equity_before

4. Default penalty dibuat lebih ringan agar agent tidak collapse ke no-trade.

5. SL/TP memakai high/low candle, bukan close saja.

6. Eksekusi action memakai candle berikutnya:
   observation pada candle t -> order dieksekusi di open candle t+1.

7. Drawdown dihitung dari equity, bukan hanya balance.

8. Trade log lebih lengkap.

Catatan:
- Observation tetap shape (9,) agar kompatibel dengan WindowedObservationWrapper.
- Untuk CNN-1D, input akhir tetap (window_size, 9), contoh (32, 9).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForexEnvWeek5(gym.Env):
    metadata = {"render_modes": []}

    ACTION_HOLD = 0
    ACTION_OPEN_BUY = 1
    ACTION_OPEN_SELL = 2
    ACTION_CLOSE = 3

    def __init__(
        self,
        df,
        spread: float = 0.20,
        slippage: float = 0.05,
        commission: float = 0.50,
        contract_size: float = 100.0,
        risk_per_trade: float = 0.01,
        sl_pct: float = 0.010,
        tp_pct: float = 0.020,
        initial_balance: float = 1000.0,
        trade_penalty: float = 0.01,
        action_penalty: float = 0.002,
        invalid_action_penalty: float = 0.05,
        drawdown_penalty: float = 0.10,
        rsi_period: int = 14,
        ma_fast: int = 10,
        ma_slow: int = 20,
        sr_lookback: int = 20,
        conservative_intrabar: bool = True,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()

        required = ["close"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

        # Kalau open/high/low tidak ada, fallback ke close agar script tetap jalan.
        for col in ["open", "high", "low"]:
            if col not in self.df.columns:
                self.df[col] = self.df["close"]

        self.opens = self.df["open"].astype(float).values
        self.highs = self.df["high"].astype(float).values
        self.lows = self.df["low"].astype(float).values
        self.closes = self.df["close"].astype(float).values
        self.prices = self.closes
        self.n_steps = len(self.closes)

        self.spread = float(spread)
        self.slippage = float(slippage)
        self.commission = float(commission)
        self.contract_size = float(contract_size)
        self.risk_per_trade = float(risk_per_trade)
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self.initial_balance = float(initial_balance)

        self.trade_penalty = float(trade_penalty)
        self.action_penalty = float(action_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.drawdown_penalty = float(drawdown_penalty)

        self.rsi_period = int(rsi_period)
        self.ma_fast_n = int(ma_fast)
        self.ma_slow_n = int(ma_slow)
        self.sr_lookback = int(sr_lookback)
        self.conservative_intrabar = bool(conservative_intrabar)

        # 0 Hold, 1 Open Buy, 2 Open Sell, 3 Close Position
        self.action_space = spaces.Discrete(4)

        # Shape tetap 9 agar pipeline sequence lama tetap kompatibel.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self._precompute_features()
        self._reset_state()

    # ======================================================================
    # Precompute indicators
    # ======================================================================
    def _precompute_features(self) -> None:
        p = self.closes
        n = len(p)

        delta = np.diff(p, prepend=p[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        rsi = np.full(n, 50.0, dtype=np.float64)
        if n > self.rsi_period + 1:
            avg_gain = np.mean(gain[1:self.rsi_period + 1])
            avg_loss = np.mean(loss[1:self.rsi_period + 1])

            for i in range(self.rsi_period, n):
                if i > self.rsi_period:
                    avg_gain = ((avg_gain * (self.rsi_period - 1)) + gain[i]) / self.rsi_period
                    avg_loss = ((avg_loss * (self.rsi_period - 1)) + loss[i]) / self.rsi_period

                rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        self.rsi = rsi

        def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
            out = np.full(len(arr), np.nan, dtype=np.float64)
            if len(arr) < window:
                return out
            csum = np.cumsum(arr, dtype=np.float64)
            out[window - 1:] = (
                csum[window - 1:] - np.concatenate([[0.0], csum[:-window]])
            ) / window
            return out

        self.ma_fast_vals = rolling_mean(p, self.ma_fast_n)
        self.ma_slow_vals = rolling_mean(p, self.ma_slow_n)

    # ======================================================================
    # Reset and quote helpers
    # ======================================================================
    def _start_step(self) -> int:
        return max(self.ma_slow_n, self.rsi_period, self.sr_lookback) + 1

    def _reset_state(self) -> None:
        self.balance = float(self.initial_balance)
        self.current_step = min(self._start_step(), max(1, self.n_steps - 2))

        self.position = 0  # 0 flat, 1 long, -1 short
        self.entry_price = 0.0
        self.entry_step = -1
        self.entry_time = None

        self.lot_size = 0.01
        self.sl_price = 0.0
        self.tp_price = 0.0

        self.trade_history = []
        self.balance_history = [float(self.balance)]
        self.equity_history = [float(self.balance)]
        self.max_equity = float(self.balance)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _quote_from_price(self, price: float, apply_slippage: bool = False) -> Tuple[float, float, float]:
        px = float(price)
        if apply_slippage:
            px += float(self.np_random.uniform(-self.slippage, self.slippage))
        ask = px + (self.spread / 2.0)
        bid = px - (self.spread / 2.0)
        return px, bid, ask

    def _calc_lot(self, entry_price: float) -> float:
        risk_amount = max(self.balance, 0.0) * self.risk_per_trade
        risk_per_lot = max(self.sl_pct * entry_price * self.contract_size, 1e-8)
        lot = risk_amount / risk_per_lot
        return float(np.clip(lot, 0.001, 1.0))

    def _floating_pnl(self, bid: float, ask: float) -> float:
        if self.position == 1 and self.entry_price > 0:
            return (bid - self.entry_price) * self.contract_size * self.lot_size
        if self.position == -1 and self.entry_price > 0:
            return (self.entry_price - ask) * self.contract_size * self.lot_size
        return 0.0

    def _equity(self, bid: float, ask: float) -> float:
        return float(self.balance + self._floating_pnl(bid, ask))

    def _drawdown_from_equity(self, equity: float) -> float:
        peak = max(self.max_equity, 1e-8)
        return float(max(0.0, (peak - equity) / peak))

    def _get_time_value(self, step: int):
        if "time" in self.df.columns:
            try:
                return str(self.df.loc[step, "time"])
            except Exception:
                return None
        return None

    # ======================================================================
    # Trading helpers
    # ======================================================================
    def _open_long(self, ask: float, step: int) -> None:
        self.lot_size = self._calc_lot(ask)
        self.entry_price = float(ask)
        self.position = 1
        self.entry_step = int(step)
        self.entry_time = self._get_time_value(step)
        self.sl_price = self.entry_price * (1.0 - self.sl_pct)
        self.tp_price = self.entry_price * (1.0 + self.tp_pct)
        self.balance -= self.commission

    def _open_short(self, bid: float, step: int) -> None:
        self.lot_size = self._calc_lot(bid)
        self.entry_price = float(bid)
        self.position = -1
        self.entry_step = int(step)
        self.entry_time = self._get_time_value(step)
        self.sl_price = self.entry_price * (1.0 + self.sl_pct)
        self.tp_price = self.entry_price * (1.0 - self.tp_pct)
        self.balance -= self.commission

    def _close_position(self, exit_price: float, step: int, reason: str) -> float:
        if self.position == 0:
            return 0.0

        side = "long" if self.position == 1 else "short"

        if self.position == 1:
            gross_profit = (exit_price - self.entry_price) * self.contract_size * self.lot_size
        else:
            gross_profit = (self.entry_price - exit_price) * self.contract_size * self.lot_size

        net_profit = gross_profit - self.commission
        self.balance += net_profit

        holding_period = int(step - self.entry_step) if self.entry_step >= 0 else 0

        self.trade_history.append({
            "entry_step": int(self.entry_step),
            "exit_step": int(step),
            "entry_time": self.entry_time,
            "exit_time": self._get_time_value(step),
            "side": side,
            "entry_price": float(self.entry_price),
            "exit_price": float(exit_price),
            "sl_price": float(self.sl_price),
            "tp_price": float(self.tp_price),
            "lot_size": float(self.lot_size),
            "gross_profit": float(gross_profit),
            "entry_commission": float(self.commission),
            "exit_commission": float(self.commission),
            "profit": float(net_profit),
            "net_profit": float(net_profit),
            "type": str(reason),
            "exit_reason": str(reason),
            "holding_period": holding_period,
            "balance_after_trade": float(self.balance),
        })

        self.position = 0
        self.entry_price = 0.0
        self.entry_step = -1
        self.entry_time = None
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.lot_size = 0.01

        return float(net_profit)

    def _check_intrabar_sl_tp(self, high: float, low: float) -> Optional[Tuple[str, float]]:
        if self.position == 0:
            return None

        high_bid = float(high - self.spread / 2.0)
        low_bid = float(low - self.spread / 2.0)
        high_ask = float(high + self.spread / 2.0)
        low_ask = float(low + self.spread / 2.0)

        if self.position == 1:
            hit_sl = low_bid <= self.sl_price
            hit_tp = high_bid >= self.tp_price

            if hit_sl and hit_tp:
                return ("SL", self.sl_price) if self.conservative_intrabar else ("TP", self.tp_price)
            if hit_sl:
                return "SL", self.sl_price
            if hit_tp:
                return "TP", self.tp_price

        elif self.position == -1:
            hit_sl = high_ask >= self.sl_price
            hit_tp = low_ask <= self.tp_price

            if hit_sl and hit_tp:
                return ("SL", self.sl_price) if self.conservative_intrabar else ("TP", self.tp_price)
            if hit_sl:
                return "SL", self.sl_price
            if hit_tp:
                return "TP", self.tp_price

        return None

    def get_action_mask(self) -> np.ndarray:
        """
        Mask action valid untuk action masking Week5.
        True  = boleh dipilih
        False = invalid
        """
        if self.position == 0:
            return np.array([True, True, True, False], dtype=bool)
        return np.array([True, False, False, True], dtype=bool)

    # ======================================================================
    # Main step
    # ======================================================================
    def step(self, action: int):
        action = int(action)

        if self.current_step >= self.n_steps - 1:
            return self._get_obs(), 0.0, True, False, {"warning": "end_of_data"}

        exec_step = self.current_step + 1

        current_close = float(self.closes[self.current_step])
        _, prev_bid, prev_ask = self._quote_from_price(current_close, apply_slippage=False)
        prev_equity = self._equity(prev_bid, prev_ask)
        prev_drawdown = self._drawdown_from_equity(prev_equity)

        raw_open = float(self.opens[exec_step])
        raw_high = float(self.highs[exec_step])
        raw_low = float(self.lows[exec_step])
        raw_close = float(self.closes[exec_step])

        _, open_bid, open_ask = self._quote_from_price(raw_open, apply_slippage=True)
        _, close_bid, close_ask = self._quote_from_price(raw_close, apply_slippage=False)

        trade_executed = False
        invalid_action = False
        force_close = False
        close_reason = None

        # --------------------------------------------------
        # 1. Eksekusi action di open candle berikutnya.
        # --------------------------------------------------
        if action == self.ACTION_HOLD:
            pass

        elif action == self.ACTION_OPEN_BUY:
            if self.position == 0:
                self._open_long(open_ask, exec_step)
                trade_executed = True
            else:
                invalid_action = True

        elif action == self.ACTION_OPEN_SELL:
            if self.position == 0:
                self._open_short(open_bid, exec_step)
                trade_executed = True
            else:
                invalid_action = True

        elif action == self.ACTION_CLOSE:
            if self.position == 1:
                self._close_position(open_bid, exec_step, reason="manual_close_buy")
                trade_executed = True
                close_reason = "manual_close_buy"
            elif self.position == -1:
                self._close_position(open_ask, exec_step, reason="manual_close_sell")
                trade_executed = True
                close_reason = "manual_close_sell"
            else:
                invalid_action = True

        else:
            invalid_action = True

        # --------------------------------------------------
        # 2. Jika posisi masih terbuka, cek SL/TP intrabar.
        # --------------------------------------------------
        if self.position != 0:
            force = self._check_intrabar_sl_tp(raw_high, raw_low)
            if force is not None:
                tag, px = force
                self._close_position(px, exec_step, reason=tag)
                trade_executed = True
                force_close = True
                close_reason = tag

        # --------------------------------------------------
        # 3. Advance step, force close di akhir data.
        # --------------------------------------------------
        self.current_step = exec_step
        terminated = self.current_step >= self.n_steps - 1 or self.balance <= 0
        truncated = False

        if terminated and self.position != 0:
            if self.position == 1:
                self._close_position(close_bid, exec_step, reason="END")
            else:
                self._close_position(close_ask, exec_step, reason="END")
            trade_executed = True
            close_reason = "END"

        new_equity = self._equity(close_bid, close_ask)
        current_drawdown = self._drawdown_from_equity(new_equity)

        reward = new_equity - prev_equity

        if action in [self.ACTION_OPEN_BUY, self.ACTION_OPEN_SELL, self.ACTION_CLOSE]:
            reward -= self.action_penalty

        if trade_executed:
            reward -= self.trade_penalty

        if invalid_action:
            reward -= self.invalid_action_penalty

        if current_drawdown > prev_drawdown:
            reward -= self.drawdown_penalty * (current_drawdown - prev_drawdown) * self.initial_balance

        self.max_equity = max(self.max_equity, new_equity)
        self.balance_history.append(float(self.balance))
        self.equity_history.append(float(new_equity))

        info = {
            "balance": float(round(self.balance, 6)),
            "equity": float(round(new_equity, 6)),
            "position": int(self.position),
            "entry_price": float(round(self.entry_price, 6)),
            "sl": float(round(self.sl_price, 6)),
            "tp": float(round(self.tp_price, 6)),
            "trade_executed": bool(trade_executed),
            "invalid_action": bool(invalid_action),
            "force_close": bool(force_close),
            "close_reason": close_reason,
            "reward": float(round(reward, 8)),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ======================================================================
    # Observation and stats
    # ======================================================================
    def _get_sr(self, step: int) -> Tuple[float, float]:
        start = max(1, step - self.sr_lookback)
        window = self.closes[start:step]
        if len(window) == 0:
            px = float(self.closes[step])
            return px, px
        return float(np.min(window)), float(np.max(window))

    def _get_obs(self):
        i = int(np.clip(self.current_step, 1, self.n_steps - 1))
        base = max(float(self.closes[0]), 1e-8)
        price = float(self.closes[i])
        prev_price = float(self.closes[i - 1])

        _, bid, ask = self._quote_from_price(price, apply_slippage=False)
        equity = self._equity(bid, ask)

        unrealized_ratio = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_ratio = self._floating_pnl(bid, ask) / max(self.initial_balance, 1e-8)

        rsi_norm = (float(self.rsi[i]) - 50.0) / 50.0

        mf = self.ma_fast_vals[i] if not np.isnan(self.ma_fast_vals[i]) else price
        ms = self.ma_slow_vals[i] if not np.isnan(self.ma_slow_vals[i]) else price
        ma_sig = np.clip((mf - ms) / base * 10.0, -1.0, 1.0)

        support, resistance = self._get_sr(i)
        dist_sup = float(np.clip((price - support) / base * 2.0, 0.0, 1.0))
        dist_res = float(np.clip((resistance - price) / base * 2.0, 0.0, 1.0))

        return np.array([
            price / base,
            prev_price / base,
            float(self.position),
            equity / self.initial_balance,
            float(np.clip(unrealized_ratio, -0.25, 0.25)),
            float(np.clip(rsi_norm, -1.0, 1.0)),
            float(np.clip(ma_sig, -1.0, 1.0)),
            dist_sup,
            dist_res,
        ], dtype=np.float32)

    def get_stats(self) -> Dict:
        trades = self.trade_history
        equity = np.asarray(self.equity_history, dtype=np.float64)

        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / (peak + 1e-8)
            max_dd = float(dd.min())
        else:
            max_dd = 0.0

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": "0.0%",
                "final_balance": f"${self.balance:.2f}",
                "final_equity": f"${equity[-1]:.2f}" if len(equity) else f"${self.balance:.2f}",
                "max_drawdown": f"{max_dd * 100:.2f}%",
            }

        profits = [float(t.get("profit", 0.0)) for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sl_n = sum(1 for t in trades if t.get("type") == "SL")
        tp_n = sum(1 for t in trades if t.get("type") == "TP")
        manual_n = len(trades) - sl_n - tp_n

        return {
            "total_trades": len(trades),
            "win_rate": f"{len(wins) / len(trades) * 100:.1f}%",
            "avg_profit": f"${np.mean(wins):.2f}" if wins else "$0.00",
            "avg_loss": f"${np.mean(losses):.2f}" if losses else "$0.00",
            "total_profit": f"${sum(profits):.2f}",
            "profit_factor": f"{profit_factor:.3f}",
            "final_balance": f"${self.balance:.2f}",
            "max_drawdown": f"{max_dd * 100:.2f}%",
            "sl_hits": f"{sl_n}",
            "tp_hits": f"{tp_n}",
            "manual_close": manual_n,
        }
