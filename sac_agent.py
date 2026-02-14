"""
Discrete Soft Actor-Critic (SAC) Agent with LFM Encoder — attention baseline.

Serves as an attention-based baseline for comparison against SAC-LTC (proposed).
Uses the Liquid Foundation Model (LFM) encoder with adaptive gating and
multi-head self-attention for temporal mixing, providing parallel sequence
processing but lacking explicit continuous-time dynamics.

Implements SAC for discrete action spaces following the formulation
from Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings".

Key components:
  - Actor: LFM encoder → softmax policy  π(a|s)
  - Twin Critics: LFM encoders → Q-value vectors  Q(s, ·)
  - Automatic entropy tuning via learnable log(α)
  - Replay buffer for off-policy learning
"""

import copy
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from lfm_module import LFMEncoder


# ======================================================================
# Replay Buffer
# ======================================================================

class ReplayBuffer:
    """Fixed-size ring buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int, state_shape: Tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0

        # Pre-allocate numpy arrays for speed
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.from_numpy(self.states[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }

    def __len__(self):
        return self.size


# ======================================================================
# Actor Network
# ======================================================================

class Actor(nn.Module):
    """
    Policy network:  state sequence → action probabilities.
    Uses the LFM encoder followed by a linear softmax head.
    """

    def __init__(self, encoder: LFMEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, T, F)
        Returns:
            action_probs: (B, num_actions) — categorical distribution.
        """
        z = self.encoder(state)                  # (B, latent_dim)
        logits = self.head(z)                    # (B, num_actions)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample or select an action.

        Returns:
            action: int
            action_probs: (1, num_actions) tensor
        """
        action_probs = self.forward(state)  # (1, A)

        if deterministic:
            action = action_probs.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

        return action, action_probs


# ======================================================================
# Critic Network (twin)
# ======================================================================

class Critic(nn.Module):
    """
    Q-value network:  state sequence → Q-values for all actions.
    Uses the LFM encoder followed by a linear head.
    """

    def __init__(self, encoder: LFMEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, T, F)
        Returns:
            q_values: (B, num_actions)
        """
        z = self.encoder(state)
        return self.head(z)


# ======================================================================
# SAC Agent
# ======================================================================

class SACAgent:
    """
    Discrete Soft Actor-Critic with LFM-based encoder.

    Features:
      - Separate LFM encoders for actor and critics (no weight sharing)
      - Twin critics with minimum-Q trick
      - Automatic entropy coefficient α tuning
      - Polyak-averaged target critics
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        input_dim: int,
        device: torch.device,
        # LFM hyperparams
        model_dim: int = 128,
        latent_dim: int = 128,
        num_blocks: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 64,
        # SAC hyperparams
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

        # ---- Build networks ----

        def _make_encoder():
            return LFMEncoder(
                input_dim=input_dim,
                model_dim=model_dim,
                latent_dim=latent_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
            )

        # Actor
        self.actor = Actor(_make_encoder(), num_actions).to(device)

        # Twin critics
        self.critic1 = Critic(_make_encoder(), num_actions).to(device)
        self.critic2 = Critic(_make_encoder(), num_actions).to(device)

        # Target critics (Polyak-averaged copies)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # ---- Optimisers ----
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)

        # ---- Automatic entropy tuning ----
        # Target entropy: -ratio * log(1/|A|)  (a fraction of maximum entropy)
        self.target_entropy = -target_entropy_ratio * np.log(1.0 / num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        # ---- Replay buffer ----
        self.replay_buffer = ReplayBuffer(buffer_size, state_shape, device)

        # ---- Logging ----
        self.train_step_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action for a single state (numpy) during rollout."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return action

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Perform one gradient step on critics, actor, and alpha.
        Returns a dict of scalar losses for logging.
        """
        if len(self.replay_buffer) < self.learning_starts:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]           # (B, T, F)
        actions = batch["actions"]         # (B,)
        rewards = batch["rewards"]         # (B,)
        next_states = batch["next_states"] # (B, T, F)
        dones = batch["dones"]             # (B,)

        # ---- Critic update ----
        with torch.no_grad():
            next_action_probs = self.actor(next_states)  # (B, A)
            # Clamp for numerical safety before log
            next_action_probs = next_action_probs.clamp(min=1e-8)
            next_log_probs = torch.log(next_action_probs)

            # Target Q-values: min of twin targets
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = torch.min(target_q1, target_q2)  # (B, A)

            # V(s') = Σ_a π(a|s')[Q(s',a) - α log π(a|s')]
            next_v = (next_action_probs * (target_q - self.alpha * next_log_probs)).sum(dim=-1)

            # TD target
            td_target = rewards + self.gamma * (1.0 - dones) * next_v  # (B,)

        # Current Q-values for the taken actions
        q1_all = self.critic1(states)  # (B, A)
        q2_all = self.critic2(states)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic1_loss = F.mse_loss(q1, td_target)
        critic2_loss = F.mse_loss(q2, td_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---- Actor update ----
        action_probs = self.actor(states)  # (B, A)
        action_probs_clamped = action_probs.clamp(min=1e-8)
        log_probs = torch.log(action_probs_clamped)

        with torch.no_grad():
            q1_pi = self.critic1(states)
            q2_pi = self.critic2(states)
            min_q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss:  E[ Σ_a π(a|s)(α log π(a|s) - Q(s,a)) ]
        actor_loss = (action_probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Alpha (entropy coefficient) update ----
        # Loss: -α E[ Σ_a π(a|s) log π(a|s) ] - α * target_entropy
        entropy = -(action_probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Polyak-average target networks ----
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        self.train_step_count += 1

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ·θ_source + (1-τ)·θ_target."""
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
