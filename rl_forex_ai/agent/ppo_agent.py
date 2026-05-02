import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic untuk ForexEnvWeek4.

    Input:
        state: (batch, state_dim)

    Output:
        policy logits: (batch, action_dim)
        value:         (batch, 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        x = self.shared(state)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def act(self, state: np.ndarray, deterministic: bool = False):
        if state.ndim == 1:
            state = state[None, :]

        state_t = torch.tensor(state, dtype=torch.float32)
        logits, value = self.forward(state_t)

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

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 10,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        self.net.eval()
        with torch.no_grad():
            return self.net.act(obs, deterministic=deterministic)

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        advantages = []
        gae = 0.0

        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def update(self, rollout):
        states = torch.tensor(np.array(rollout["states"]), dtype=torch.float32)
        actions = torch.tensor(rollout["actions"], dtype=torch.long)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        indices = np.arange(n)

        self.net.train()

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy, values = self.net.evaluate_actions(
                    mb_states,
                    mb_actions,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location="cpu"))