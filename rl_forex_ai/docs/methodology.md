# RL Forex AI — Dokumentasi Teknis

## 1. Cara Kerja Sistem

### 1.1 Reinforcement Learning Framework

Sistem ini menggunakan **Deep Q-Network (DQN)** dengan arsitektur Pure NumPy (tanpa PyTorch/TensorFlow) yang ditraining menggunakan metode Q-Learning.

**Loop interaksi agent-environment:**
```
obs, _ = env.reset()
while not done:
    action = agent.act(obs)          # epsilon-greedy
    next_obs, reward, done = env.step(action)
    agent.store(obs, action, reward, next_obs, done)
    agent.train()                    # sample dari replay buffer
agent.decay_epsilon()               # satu kali per episode
```

### 1.2 Komponen Utama

| Komponen | File | Fungsi |
|----------|------|--------|
| Environment | `env/forex_env_pro_4.py` | Simulasi market dengan spread, SL, TP, RSI, MA, S/R |
| Agent | `agent/dqn_agent.py` | DQN dengan ReplayBuffer + NumpyMLP |
| Data Loader | `utils/data_loader.py` | Parse CSV format Indonesia |
| Metrics | `utils/metrics.py` | Sharpe, MaxDD, WinRate, ProfitFactor |

---

## 2. Metode

### 2.1 Q-Learning Update Rule

```
Q(s,a) ← reward + γ · max Q(s', a')
```

- **γ (gamma)** = 0.95 — discount factor
- **α (lr)** = 5e-4 — learning rate
- **ε (epsilon)** = 1.0 → 0.05 — exploration decay per episode

### 2.2 Neural Network (NumpyMLP)

```
Input(9) → Linear(64) → ReLU → Linear(64) → ReLU → Output(3)
```

Diinisialisasi dengan **He initialization**: `w ~ N(0, √(2/fan_in))`

### 2.3 Experience Replay

- Buffer capacity: 20,000 transitions
- Batch size: 64
- Sampling: random (uniform)
- Reward clipping: `[-200, 200]` untuk stabilitas numerik

### 2.4 Observation Space (9 fitur)

| Index | Fitur | Normalisasi |
|-------|-------|-------------|
| 0 | price_now / price_base | relative |
| 1 | price_prev / price_base | relative |
| 2 | position (-1/0/1) | raw |
| 3 | balance / initial_balance | relative |
| 4 | unrealized_pnl | clip(-0.1, 0.1) |
| 5 | RSI_norm = (RSI-50)/50 | [-1, 1] |
| 6 | MA_cross signal | clip(-1, 1) |
| 7 | dist_to_support | [0, 1] |
| 8 | dist_to_resistance | [0, 1] |

### 2.5 Action Space

| Action | Kode | Kondisi |
|--------|------|---------|
| Hold | 0 | Selalu valid |
| Buy | 1 | Open buy (position=0) atau close sell (position=-1) |
| Sell | 2 | Open sell (position=0) atau close buy (position=1) |

### 2.6 Reward Function

```python
# Realized P&L (saat close posisi)
profit = (price_close - price_open) * contract_size * lot_size - commission

# Unrealized signal (saat hold)
reward_hold = unrealized_pnl * 0.01   # skala kecil, hanya sinyal arah

# SL/TP auto-close
reward_sl = -loss_amount
reward_tp = +profit_amount
```

### 2.7 Risk Management (Week 3)

- **Risk per trade**: 1% dari balance
- **Stop Loss**: 1% di bawah/atas entry price
- **Take Profit**: 2% di atas/bawah entry price (RR ratio 1:2)
- **Dynamic lot sizing**: `lot = (balance × 0.01) / (sl_pct × entry × contract_size)`

---

## 3. Hasil Evaluasi

### 3.1 Progression Antar Week

| Week | Environment | State | Hasil Utama |
|------|-------------|-------|-------------|
| 1 | ForexEnv | (2,) | Profit kumulatif +1.3 (normalized) |
| 2 | ForexEnvWeek2 | (4,) | Balance $1,000 → $2,250 (+144%) |
| 3 | ForexEnvWeek3 | (5,) | +SL/TP, loss terkontrol |
| 4 | ForexEnvWeek4 | (9,) | +RSI/MA/S&R, lebih informatif |
| 7 | Multi-pair | (9,) | Out-of-sample test (80/20 split) |

### 3.2 Metrik Performa (Week 2 — best evaluated)

| Metrik | Nilai |
|--------|-------|
| Initial Balance | $1,000 |
| Final Balance | $2,250.14 |
| Total Return | +144% |
| Total Trades | 382 |
| Win Rate | 55.5% |
| Avg Win | $20.81 |
| Avg Loss | -$17.47 |
| Profit Factor | 1.19× |

### 3.3 Training Progression (500 episodes)

- **Ep 1–150**: Eksplorasi aktif, reward negatif (normal)
- **Ep 150–300**: Agent mulai menemukan strategi
- **Ep 300–500**: Konvergen, reward +$2,000–$5,000 per episode

---

## 4. Cara Menjalankan

### 4.1 Setup
```bash
pip install numpy pandas matplotlib gymnasium
cd rl_forex_ai
```

### 4.2 Training
```bash
# Week 4 (default, terbaik)
python train/train_ppo.py          # WEEK=4 di dalam file

# Week 5: Hyperparameter tuning
python train/train_ppo_1.py

# Week 7: Multi-pair
python train/train_ppo_2.py
```

### 4.3 Evaluasi
```bash
# Standard test
python test/test_model.py

# Final demo lengkap
python test/final_demo.py
```

### 4.4 Dashboard
```bash
python visualize/week8_dashboard.py
```

---

## 5. Keterbatasan & Catatan

1. **Data**: Hanya 853 candle harian (Jan 2023 – Mei 2026). Lebih banyak data = lebih baik.
2. **Overfitting**: Model di-test pada data yang sama dengan training. Gunakan `train_ppo_2.py` untuk out-of-sample test yang benar.
3. **NumPy DQN**: Tanpa GPU, training lambat untuk > 1000 episode. Upgrade ke PyTorch untuk produksi.
4. **Market realism**: Tidak ada news impact, liquidity risk, atau weekend gap.
5. **Live trading**: JANGAN langsung live trading tanpa paper trading minimal 3 bulan.

---

## 6. Referensi

- Mnih et al. (2015) — "Human-level control through deep reinforcement learning" (DQN paper)
- Schulman et al. (2017) — "Proximal Policy Optimization Algorithms" (PPO)
- Wilder (1978) — RSI indicator
- Q-Learning: Watkins & Dayan (1992)
