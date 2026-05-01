"""
WEEK 7 — Multi-Pair Environment
Day 1-2: Load multiple pair CSVs
Day 3: Random pair per episode (forces generalization)
Day 5: get_pair_stats() for comparison
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from env.forex_env_pro_4 import ForexEnvWeek4
from utils.data_loader import load_forex_data


class MultiPairEnv(gym.Env):
    """
    Wraps multiple ForexEnvWeek4 instances.
    Each episode randomly picks one pair.
    Forces agent to learn general market patterns.
    """
    def __init__(self, pair_dfs, env_kwargs=None):
        super().__init__()
        env_kwargs = env_kwargs or {}

        self.pair_names = list(pair_dfs.keys())
        self.envs = {
            name: ForexEnvWeek4(df, **env_kwargs)
            for name, df in pair_dfs.items()
        }
        self.active_pair = self.pair_names[0]
        self.active_env  = self.envs[self.active_pair]

        self.observation_space = self.active_env.observation_space
        self.action_space      = self.active_env.action_space
        self.episode_count     = 0
        self.pair_history      = []

    def reset(self, seed=None, options=None):
        self.active_pair = np.random.choice(self.pair_names)
        self.active_env  = self.envs[self.active_pair]
        self.pair_history.append(self.active_pair)
        self.episode_count += 1
        return self.active_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.active_env.step(action)

    @property
    def balance(self):         return self.active_env.balance
    @property
    def balance_history(self): return self.active_env.balance_history
    @property
    def trade_history(self):   return self.active_env.trade_history
    @property
    def initial_balance(self): return self.active_env.initial_balance

    def get_pair_stats(self):
        stats = {}
        for name, env in self.envs.items():
            trades = [t["profit"] if isinstance(t, dict) else t for t in env.trade_history]
            ep_count = self.pair_history.count(name)
            if trades:
                wins = [t for t in trades if t > 0]
                stats[name] = {
                    "episodes" : ep_count,
                    "trades"   : len(trades),
                    "win_rate" : f"{len(wins)/len(trades)*100:.1f}%",
                    "total_pnl": f"${sum(trades):.2f}",
                }
            else:
                stats[name] = {"episodes": ep_count, "trades": 0}
        return stats


def load_multi_pair(data_dir, pairs=None):
    """
    Load multiple pair CSVs.
    Falls back to main XAUUSD file if pair-specific file not found.
    Add EUR/USD and GBP/USD files named: Data_EURUSD.csv, Data_GBPUSD.csv
    """
    if pairs is None:
        pairs = ["XAUUSD", "EURUSD", "GBPUSD"]

    pair_dfs = {}
    for pair in pairs:
        candidates = [
            os.path.join(data_dir, f"Data_{pair}.csv"),
            os.path.join(data_dir, f"{pair}.csv"),
            os.path.join(data_dir, "Data_historis(23-26).csv"),  # fallback
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    df = load_forex_data(path)
                    pair_dfs[pair] = df
                    print(f"  [OK] {pair}: {len(df)} baris")
                    break
                except Exception as e:
                    print(f"  [skip] {pair}: {e}")
        else:
            print(f"  [WARN] {pair}: tidak ditemukan, skip")

    if not pair_dfs:
        raise FileNotFoundError("Tidak ada pair yang berhasil dimuat")
    return pair_dfs