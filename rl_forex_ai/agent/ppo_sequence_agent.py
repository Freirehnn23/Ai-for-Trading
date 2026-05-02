"""
PPO Sequence Agent untuk trading time-series.

Mendukung encoder JST:
- cnn1d
- lstm
- attention
- transformer

Input:
    observation shape = (window_size, feature_dim)
    contoh: (32, 9)

Output:
    policy logits untuk action
    value estimation untuk PPO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class CNN1DEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        x = x.transpose(1, 2)  # (batch, features, window)
        x = self.net(x)
        return x.squeeze(-1)


class LSTMEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        output, _ = self.lstm(x)
        return output[:, -1, :]


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()

        self.proj = nn.Linear(feature_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        x = self.proj(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # mean pooling across time
        return x.mean(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, window_size, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        x = self.input_proj(x)

        window = x.shape[1]
        x = x + self.pos_embedding[:, :window, :]

        x = self.transformer(x)
        x = self.norm(x)

        # ambil token terakhir sebagai representasi state terbaru
        return x[:, -1, :]


class SequenceActorCritic(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        action_dim: int,
        encoder_type: str = "lstm",
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.window_size = int(input_shape[0])
        self.feature_dim = int(input_shape[1])
        self.action_dim = int(action_dim)
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "cnn1d":
            self.encoder = CNN1DEncoder(
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
            )

        elif self.encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
            )

        elif self.encoder_type == "attention":
            self.encoder = AttentionEncoder(
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )

        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                window_size=self.window_size,
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

        else:
            raise ValueError(
                f"encoder_type tidak dikenal: {encoder_type}. "
                "Pilih: cnn1d, lstm, attention, transformer"
            )

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(state)
        x = self.shared(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value


@dataclass
class PPOSequenceConfig:
    encoder_type: str = "lstm"
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    update_epochs: int = 5
    batch_size: int = 64
    hidden_dim: int = 128
    value_coef: float = 0.50
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.50
    dropout: float = 0.10
    num_heads: int = 4
    num_layers: int = 2
    device: str = "auto"


class PPOSequenceAgent:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        action_dim: int,
        config: PPOSequenceConfig | None = None,
    ):
        self.input_shape = tuple(input_shape)
        self.action_dim = int(action_dim)
        self.config = config or PPOSequenceConfig()

        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

        self.net = SequenceActorCritic(
            input_shape=self.input_shape,
            action_dim=self.action_dim,
            encoder_type=self.config.encoder_type,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.config.lr,
        )

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 2:
            obs = obs[None, :, :]

        return torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        self.net.eval()

        state_t = self._obs_to_tensor(obs)
        logits, value = self.net(state_t)

        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze(-1).item()),
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.net(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze(-1)

        return log_probs, entropy, values

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float = 0.0,
    ) -> Tuple[List[float], List[float]]:
        advantages: List[float] = []
        gae = 0.0
        values_ext = values + [last_value]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])

            delta = (
                rewards[t]
                + self.config.gamma * values_ext[t + 1] * mask
                - values_ext[t]
            )

            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * mask
                * gae
            )

            advantages.insert(0, gae)

        returns = [
            adv + val
            for adv, val in zip(advantages, values)
        ]

        return advantages, returns

    def update(self, rollout: Dict[str, List]) -> Dict[str, float]:
        states = torch.tensor(
            np.asarray(rollout["states"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            rollout["actions"],
            dtype=torch.long,
            device=self.device,
        )

        old_log_probs = torch.tensor(
            rollout["log_probs"],
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.tensor(
            rollout["returns"],
            dtype=torch.float32,
            device=self.device,
        )

        advantages = torch.tensor(
            rollout["advantages"],
            dtype=torch.float32,
            device=self.device,
        )

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        n_samples = states.shape[0]
        indices = np.arange(n_samples)

        last_policy_loss = torch.tensor(0.0, device=self.device)
        last_value_loss = torch.tensor(0.0, device=self.device)
        last_entropy = torch.tensor(0.0, device=self.device)

        self.net.train()

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                mb_idx = indices[start:start + self.config.batch_size]
                mb_idx_t = torch.tensor(
                    mb_idx,
                    dtype=torch.long,
                    device=self.device,
                )

                mb_states = states[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_returns = returns[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]

                new_log_probs, entropy, values = self.evaluate_actions(
                    mb_states,
                    mb_actions,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages

                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy = entropy_bonus.detach()

        return {
            "policy_loss": float(last_policy_loss.cpu().item()),
            "value_loss": float(last_value_loss.cpu().item()),
            "entropy": float(last_entropy.cpu().item()),
        }

    def save(self, path: str) -> None:
        payload = {
            "input_shape": self.input_shape,
            "action_dim": self.action_dim,
            "config": self.config.__dict__,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool = False) -> None:
        payload = torch.load(path, map_location=self.device)

        self.net.load_state_dict(payload["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )