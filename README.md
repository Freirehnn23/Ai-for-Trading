# Ai-for-Trading

# 🤖 RL Forex AI — Reinforcement Learning Trading Agent

> **Deep Q-Network (DQN) agent yang belajar trading XAUUSD (Gold) dari nol menggunakan Pure NumPy — tanpa PyTorch, tanpa library RL.**

---

## 📈 Hasil Final

| Metrik | Nilai |
|--------|-------|
| 💰 Initial Balance | $1,000 |
| 📈 Final Balance | **$2,250** |
| 🚀 Total Return | **+144%** |
| 🎯 Win Rate | **55.5%** |
| 📊 Profit Factor | **1.19×** |
| 📉 Total Trades | 382 |

---

## 🏗 Arsitektur

```
Market Data (XAUUSD Daily)
        ↓
ForexEnvWeek4 — simulasi market realistis
  ├── Spread ($0.20)
  ├── Slippage ($0.05)
  ├── Commission ($0.50/trade)
  ├── Stop Loss (1%)
  ├── Take Profit (2%)
  └── Observation: [price, prev_price, position, balance,
                    unrealized_pnl, RSI, MA_cross,
                    dist_support, dist_resistance]
        ↓
DQN Agent (Pure NumPy)
  ├── NumpyMLP: 9 → 64 → 64 → 3
  ├── Replay Buffer: 20,000 transitions
  ├── Epsilon-greedy: 1.0 → 0.05 (decay per episode)
  └── Q-Learning: Q(s,a) ← r + γ·max Q(s',a')
        ↓
Actions: [Hold | Buy | Sell]
```

---

## 📅 Roadmap (8 Minggu)

| Week | Fokus | Pencapaian |
|------|-------|-----------|
| **W1** | Foundation | DQN berjalan, environment bersih |
| **W2** | Realistic Market | Spread, slippage, commission → Balance +144% |
| **W3** | Risk Management | Stop Loss, Take Profit, dynamic lot sizing |
| **W4** | Feature Engineering | RSI, Moving Average, Support & Resistance |
| **W5** | Hyperparameter Tuning | Grid search: lr, batch, epsilon_decay |
| **W6** | Evaluation Metrics | Sharpe Ratio, Max Drawdown, Profit Factor |
| **W7** | Multi-Pair + Robustness | 80/20 train/test split, out-of-sample |
| **W8** | Finalisasi | Dashboard, docs, clean architecture |

---

## 🚀 Cara Pakai

### Setup
```bash
git clone https://github.com/username/rl_forex_ai
cd rl_forex_ai
pip install numpy pandas matplotlib gymnasium
```

### Training
```bash
# Training utama (Week 4 environment)
python train/train_ppo.py

# Hyperparameter search (Week 5)
python train/train_ppo_1.py

# Multi-pair + out-of-sample (Week 7)
python train/train_ppo_2.py
```

### Evaluasi & Visualisasi
```bash
# Evaluasi standard
python test/test_model.py

# Full demo + report
python test/final_demo.py

# 4-panel dashboard
python visualize/week8_dashboard.py
```

---

## 📁 Struktur Project

```
rl_forex_ai/
├── data/                    # Dataset XAUUSD 2023–2026
├── env/                     # Trading environments (W1→W4)
│   ├── forex_env_pro.py     # Week 1: state(2)
│   ├── forex_env_pro_2.py   # Week 2: state(4) + spread/commission
│   ├── forex_env_pro_3.py   # Week 3: state(5) + SL/TP
│   └── forex_env_pro_4.py   # Week 4: state(9) + RSI/MA/S&R ← FINAL
├── agent/
│   └── dqn_agent.py         # DQN: ReplayBuffer + NumpyMLP
├── utils/
│   ├── data_loader.py       # Parser CSV Indonesia
│   └── metrics.py           # Sharpe, MaxDD, WinRate
├── train/
│   ├── train_ppo.py         # Universal trainer (WEEK=2/3/4)
│   ├── train_ppo_1.py       # Week 5: hyperparameter tuning
│   └── train_ppo_2.py       # Week 7: multi-pair
├── test/
│   ├── test_model.py        # Standard evaluation
│   └── final_demo.py        # Full demo
├── visualize/
│   └── week8_dashboard.py   # 4-panel performance dashboard
├── docs/
│   └── methodology.md       # Technical documentation
├── models/                  # Saved model weights (.pkl)
├── ARCHITECTURE.md
└── README.md
```

---

## 🧠 Konsep Kunci

### Kenapa DQN?
- Cocok untuk discrete action space (hold/buy/sell)
- Experience Replay mengurangi korelasi data
- Lebih mudah debug dibanding PPO untuk belajar

### Kenapa Pure NumPy?
- Tidak ada dependency berat (PyTorch/TensorFlow)
- Transparansi: setiap forward/backward pass bisa dilacak
- Cukup untuk dataset kecil (< 10K timestep)

### Reward Design
```python
# Saat close posisi (realized)
reward = profit - commission

# Saat hold dengan posisi terbuka (sinyal kecil)
reward = unrealized_pnl * 0.01

# Auto SL/TP
reward = ±fixed_amount
```

---

## ⚠️ Disclaimer

> Project ini dibuat untuk **tujuan edukasi dan riset**. Hasil backtest tidak menjamin profit di live trading. Pasar forex sangat kompleks dan tidak terprediksi. **Jangan gunakan uang nyata tanpa pemahaman mendalam tentang risk management.**

---

## 📚 Referensi

- Mnih et al. (2015) — Human-level control through deep reinforcement learning
- Schulman et al. (2017) — Proximal Policy Optimization Algorithms
- Wilder (1978) — New Concepts in Technical Trading Systems (RSI)

---

## 👤 Author

Dibuat sebagai project riset AI Trading menggunakan Reinforcement Learning.

**Tech Stack**: Python · NumPy · Pandas · Matplotlib · Gymnasium
