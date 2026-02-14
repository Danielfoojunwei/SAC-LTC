"""
Discrete Soft Actor-Critic with LSTM Encoder — recurrent baseline.

Uses a standard multi-layer LSTM as the sequence encoder, providing a strong
recurrent baseline for comparison against SAC-LTC (proposed). The LSTM's
fixed sigmoid forget gates contrast with LTC's input-dependent time
constants, isolating the contribution of adaptive temporal dynamics.
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sac_agent import ReplayBuffer


# ======================================================================
# LSTM Encoder (drop-in replacement for LFMEncoder)
# ======================================================================

class LSTMEncoder(nn.Module):
    """
    LSTM-based sequence encoder.

    Input:  (Batch, Seq_Len, Features)
    Output: (Batch, Latent_Dim)

    Uses a multi-layer LSTM followed by a projection from the final
    hidden state to the desired latent dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj: Optional[nn.Linear] = None
        if hidden_dim != latent_dim:
            self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            z: (B, latent_dim)
        """
        # output: (B, T, hidden_dim), (h_n, c_n) each (num_layers, B, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]  # last layer's final hidden state  (B, hidden_dim)
        if self.proj is not None:
            z = self.proj(z)
        return z


# ======================================================================
# Actor / Critic with LSTM
# ======================================================================

class LSTMActor(nn.Module):
    def __init__(self, encoder: LSTMEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(state)
        return F.softmax(self.head(z), dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        probs = self.forward(state)
        if deterministic:
            return probs.argmax(dim=-1).item(), probs
        return torch.distributions.Categorical(probs).sample().item(), probs


class LSTMCritic(nn.Module):
    def __init__(self, encoder: LSTMEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


# ======================================================================
# SAC-LSTM Agent
# ======================================================================

class SACLSTMAgent:
    """
    Discrete SAC with LSTM encoder — same algorithm as SACAgent,
    different feature extractor.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        input_dim: int,
        device: torch.device,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 1000,
        target_entropy_ratio: float = 0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_actions = num_actions

        def _enc():
            return LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)

        self.actor = LSTMActor(_enc(), num_actions).to(device)
        self.critic1 = LSTMCritic(_enc(), num_actions).to(device)
        self.critic2 = LSTMCritic(_enc(), num_actions).to(device)

        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy = -target_entropy_ratio * np.log(1.0 / num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size, state_shape, device)
        self.train_step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return action

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.learning_starts:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Critic targets
        with torch.no_grad():
            next_probs = self.actor(next_states).clamp(min=1e-8)
            next_log = torch.log(next_probs)
            tq1 = self.target_critic1(next_states)
            tq2 = self.target_critic2(next_states)
            tq = torch.min(tq1, tq2)
            next_v = (next_probs * (tq - self.alpha * next_log)).sum(dim=-1)
            td_target = rewards + self.gamma * (1.0 - dones) * next_v

        # Critic losses
        q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        c1_loss = F.mse_loss(q1, td_target)
        c2_loss = F.mse_loss(q2, td_target)

        self.critic1_optimizer.zero_grad()
        c1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        c2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        probs = self.actor(states).clamp(min=1e-8)
        log_probs = torch.log(probs)
        with torch.no_grad():
            min_q = torch.min(self.critic1(states), self.critic2(states))
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        entropy = -(probs.detach() * log_probs.detach()).sum(-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        self.train_step_count += 1

        return {
            "critic1_loss": c1_loss.item(),
            "critic2_loss": c2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "train_step_count": self.train_step_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.train_step_count = ckpt["train_step_count"]
