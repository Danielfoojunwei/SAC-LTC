"""
PPO-LSTM Baseline using sb3-contrib's RecurrentPPO.

Provides a thin wrapper that matches the training / evaluation interface
used by the benchmarking harness, while delegating the actual RL logic
to Stable-Baselines3's battle-tested RecurrentPPO implementation.

Because RecurrentPPO handles temporal modelling internally via its
built-in LSTM, the environment wrapper exposes only the *latest*
timestep's features as a 1-D observation vector, letting the LSTM
accumulate history across steps.
"""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dsa_env import DSAEnv

try:
    from sb3_contrib import RecurrentPPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ======================================================================
# Environment wrapper — single-step obs for RecurrentPPO
# ======================================================================

class SingleStepDSAEnv(gym.Env):
    """
    Wraps DSAEnv to emit a 1-D observation of the *current* timestep
    only (shape: (num_channels * num_features,)), so that RecurrentPPO's
    internal LSTM handles temporal sequencing.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, **env_kwargs):
        super().__init__()
        self._inner = DSAEnv(**env_kwargs)
        self.action_space = self._inner.action_space

        obs_dim = self._inner.num_channels * self._inner.num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed, options=options)
        return obs[-1].copy(), info  # last row = most recent timestep

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        return obs[-1].copy(), reward, terminated, truncated, info

    def render(self):
        self._inner.render()


# ======================================================================
# PPO-LSTM Agent wrapper
# ======================================================================

class PPOLSTMAgent:
    """
    Wraps sb3-contrib RecurrentPPO to expose the same interface as
    SACAgent / SACLSTMAgent for the benchmark harness.
    """

    def __init__(
        self,
        env_kwargs: dict,
        device: str = "auto",
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 1,
        lr: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 0,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for PPO-LSTM. "
                "Install with: pip install sb3-contrib"
            )

        self.env_kwargs = env_kwargs
        self.env = SingleStepDSAEnv(**env_kwargs)

        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self.env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            device=device,
            policy_kwargs={
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "net_arch": dict(pi=[128], vf=[128]),
            },
        )

        # Persistent LSTM state for evaluation rollouts
        self._lstm_states = None
        self._episode_start = np.array([True])
        self.train_step_count = 0

    def learn(self, total_timesteps: int, log_interval: int = 10):
        """Train for the given number of environment steps."""
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )
        self.train_step_count += total_timesteps

    @property
    def replay_buffer(self):
        """Compatibility shim — PPO has no replay buffer."""
        return None

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action from the current policy.
        For compatibility with the benchmark harness, `state` is the full
        (seq_len, features) observation; we extract the last row.
        """
        if state.ndim == 2:
            flat_obs = state[-1].copy()
        else:
            flat_obs = state

        action, self._lstm_states = self.model.predict(
            flat_obs,
            state=self._lstm_states,
            episode_start=self._episode_start,
            deterministic=deterministic,
        )
        self._episode_start = np.array([False])
        return int(action)

    def reset_eval_state(self):
        """Reset LSTM hidden state at the start of a new eval episode."""
        self._lstm_states = None
        self._episode_start = np.array([True])

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = RecurrentPPO.load(path, env=self.env)
