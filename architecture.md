# RL Forex AI — Clean Architecture

## Struktur Folder

```
rl_forex_ai/
│
├── data/
│   └── Data_historis(23-26).csv     ← Data XAUUSD Jan 2023 – Mei 2026 (853 baris)
│
├── env/                             ← Trading environments (1 per week)
│   ├── forex_env_pro.py             ← Week 1: State(2), no costs
│   ├── forex_env_pro_2.py           ← Week 2: State(4), spread+slippage+commission
│   ├── forex_env_pro_3.py           ← Week 3: State(5), +SL/TP/risk_per_trade
│   ├── forex_env_pro_4.py           ← Week 4: State(9), +RSI/MA/S&R (FINAL)
│   └── multi_pair_env.py            ← Week 7: Random pair per episode
│
├── agent/
│   └── dqn_agent.py                 ← DQN: ReplayBuffer + NumpyMLP + epsilon-greedy
│
├── utils/
│   ├── data_loader.py               ← CSV parser (format Indonesia → float)
│   ├── metrics.py                   ← Sharpe, MaxDD, WinRate, ProfitFactor
│   ├── indicators.py                ← MA, RSI helpers
│   └── sr_indicator.py              ← Support/Resistance
│
├── train/
│   ├── train_ppo.py                 ← Universal trainer (WEEK=2/3/4)
│   ├── train_ppo_1.py               ← Week 5: Hyperparameter grid search
│   └── train_ppo_2.py               ← Week 7: Multi-pair + out-of-sample
│
├── test/
│   ├── test_model.py                ← Standard evaluation
│   └── final_demo.py                ← Week 8: Final demo + full report
│
├── visualize/
│   ├── evaluate.py                  ← Week 1 evaluator
│   ├── week8_dashboard.py           ← Week 8: 4-panel performance dashboard
│   └── *.png                        ← Saved charts
│
├── models/
│   └── trained_agent_week4.pkl      ← Best trained model
│
├── docs/
│   └── methodology.md               ← Technical documentation
│
├── README.md                        ← GitHub README
├── ARCHITECTURE.md                  ← This file
└── requirements.txt
```

## Evolution Environment (State Space)

| Week | Class           | State Size | Fitur Tambahan                    |
|------|-----------------|-----------|-----------------------------------|
| 1    | ForexEnv        | 2         | price_now, price_prev             |
| 2    | ForexEnvWeek2   | 4         | +position, +balance               |
| 3    | ForexEnvWeek3   | 5         | +unrealized_pnl                   |
| 4    | ForexEnvWeek4   | 9         | +RSI, +MA_cross, +dist_sup, +dist_res |

## Agent Architecture

```
Input (9,) → Linear(64) → ReLU → Linear(64) → ReLU → Output(3,)
                                                         [hold, buy, sell]
```

- Optimizer: SGD (custom NumPy backprop)
- Loss: MSE (Q-learning target)
- Replay Buffer: deque(maxlen=20,000)
- Epsilon: 1.0 → 0.05 (decay 0.98/episode)
