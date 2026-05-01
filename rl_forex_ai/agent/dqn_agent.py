"""
=============================================================
WEEK 1 - DAY 3: DQN Agent (Pure NumPy — tanpa PyTorch)
=============================================================
Konsep identik dengan versi PyTorch:
  - MLP dengan 2 hidden layers (64 → 64)
  - Epsilon-greedy policy
  - Experience Replay Buffer
  - Q-Learning update: Q(s,a) ← reward + γ·max Q(s',a')
=============================================================
"""

import random
import numpy as np
from collections import deque


# ====================================================================
# 1. REPLAY BUFFER
# ====================================================================
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ====================================================================
# 2. NEURAL NETWORK (NumPy MLP)
#    Arsitektur: state_size → 64 → ReLU → 64 → ReLU → action_size
# ====================================================================
class NumpyMLP:
    def __init__(self, layer_sizes, lr=1e-3):
        self.lr = lr
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialization
            w = np.random.randn(fan_in, layer_sizes[i+1]).astype(np.float32) * np.sqrt(2.0/fan_in)
            b = np.zeros(layer_sizes[i+1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
        self.n_layers = len(self.weights)

    def _relu(self, x):      return np.maximum(0, x)
    def _relu_d(self, x):    return (x > 0).astype(np.float32)

    def forward(self, x):
        """Forward pass, simpan aktivasi untuk backward."""
        self._acts, self._pre = [x], []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self._acts[-1] @ w + b
            self._pre.append(z)
            a = self._relu(z) if i < self.n_layers - 1 else z
            self._acts.append(a)
        return self._acts[-1]

    def backward(self, y_pred, y_true):
        """Backward pass — MSE loss."""
        batch = y_pred.shape[0]
        delta = (y_pred - y_true) * (2.0 / batch)
        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                delta = delta * self._relu_d(self._pre[i])
            dw = self._acts[i].T @ delta / batch
            db = delta.mean(axis=0)
            self.weights[i] -= self.lr * dw
            self.biases[i]  -= self.lr * db
            if i > 0:
                delta = delta @ self.weights[i].T

    def get_weights(self):
        return [(w.copy(), b.copy()) for w, b in zip(self.weights, self.biases)]

    def set_weights(self, wb):
        for i, (w, b) in enumerate(wb):
            self.weights[i] = w.copy()
            self.biases[i] = b.copy()

    def predict(self, x):
        """Inference tanpa menyimpan aktivasi."""
        out = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out = out @ w + b
            if i < self.n_layers - 1:
                out = self._relu(out)
        return out


# ====================================================================
# 3. DQN AGENT
# ====================================================================
class DQNAgent:
    def __init__(self, state_size, action_size,
                 lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_capacity=10_000, batch_size=64):
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size

        self.model  = NumpyMLP([state_size, 64, 64, action_size], lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    def act(self, state):
        """Epsilon-greedy: explore (random) atau exploit (argmax Q)."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis, :])
        return int(np.argmax(q[0]))

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def train(self):
        """Q-learning update dari batch pengalaman."""
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_current = self.model.forward(states)
        q_next    = self.model.predict(next_states)
        max_next  = q_next.max(axis=1)

        targets = q_current.copy()
        for i in range(self.batch_size):
            t = rewards[i]
            if not dones[i]:
                t += self.gamma * max_next[i]
            targets[i, actions[i]] = t

        self.model.backward(q_current, targets)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Add to NumpyMLP class (patched via DQNAgent convenience methods below)
