"""
Dynamic Spectrum Access (DSA) Environment for Gymnasium.

Simulates a multi-channel wireless network where Primary Users (PUs)
occupy channels stochastically via a Markov On/Off process, and a
Secondary User (SU) agent must learn to select unoccupied channels.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DSAEnv(gym.Env):
    """
    Custom Gymnasium environment for Dynamic Spectrum Access.

    State: Noisy observation of channel conditions over the last T timesteps.
           Shape: (sequence_length, num_channels * num_features)
    Action: Discrete channel selection (0 .. num_channels-1).
    Reward: +1 success, -1 collision, -0.1 channel-switch overhead.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_channels: int = 10,
        sequence_length: int = 16,
        num_features: int = 3,
        pu_on_prob: float = 0.3,
        pu_off_prob: float = 0.5,
        noise_std: float = 0.1,
        max_steps: int = 200,
    ):
        """
        Args:
            num_channels: Number of wireless channels (N).
            sequence_length: History window length (T).
            num_features: Features per channel (SNR, interference, occupancy).
            pu_on_prob: Probability PU transitions from OFF -> ON.
            pu_off_prob: Probability PU transitions from ON -> OFF.
            noise_std: Gaussian noise added to observations.
            max_steps: Maximum steps per episode.
        """
        super().__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.pu_on_prob = pu_on_prob
        self.pu_off_prob = pu_off_prob
        self.noise_std = noise_std
        self.max_steps = max_steps

        # Discrete action: select one of N channels
        self.action_space = spaces.Discrete(num_channels)

        # Observation: (sequence_length, num_channels * num_features)
        obs_dim = num_channels * num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        # Internal state
        self._channel_states = None  # binary: 1 = occupied by PU
        self._history = None         # rolling observation buffer
        self._prev_action = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Markov channel dynamics
    # ------------------------------------------------------------------

    def _init_channels(self) -> np.ndarray:
        """Initialise PU occupancy randomly."""
        return (np.random.rand(self.num_channels) < 0.5).astype(np.float32)

    def _step_channels(self) -> None:
        """Advance PU Markov On/Off process by one step."""
        for ch in range(self.num_channels):
            if self._channel_states[ch] == 0:
                # OFF -> ON with probability pu_on_prob
                if np.random.rand() < self.pu_on_prob:
                    self._channel_states[ch] = 1.0
            else:
                # ON -> OFF with probability pu_off_prob
                if np.random.rand() < self.pu_off_prob:
                    self._channel_states[ch] = 0.0

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _build_features(self) -> np.ndarray:
        """
        Build a feature vector for the current time step.
        Features per channel: [SNR, Interference Power, Occupancy].

        Returns:
            Array of shape (num_channels * num_features,).
        """
        occupancy = self._channel_states.copy()

        # SNR: high when free, low when occupied (plus noise)
        snr = np.where(occupancy == 0, 1.0, 0.2) + np.random.randn(self.num_channels) * 0.05

        # Interference power: high when occupied
        interference = np.where(occupancy == 1, 0.8, 0.1) + np.random.randn(self.num_channels) * 0.05

        # Stack features: (num_channels * 3,)
        features = np.stack([snr, interference, occupancy], axis=-1).reshape(-1)
        return features.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Return noisy observation of the history buffer.
        Shape: (sequence_length, num_channels * num_features).
        """
        obs = self._history.copy()
        obs += np.random.randn(*obs.shape).astype(np.float32) * self.noise_std
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._channel_states = self._init_channels()
        self._prev_action = None
        self._step_count = 0

        # Fill history buffer with initial observations
        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros(
            (self.sequence_length, obs_dim), dtype=np.float32
        )
        for t in range(self.sequence_length):
            self._step_channels()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        info = {"channel_states": self._channel_states.copy()}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._step_count += 1

        # Advance channel dynamics
        self._step_channels()

        # --- Compute reward ---
        channel_occupied = self._channel_states[action] == 1.0
        if channel_occupied:
            reward = -1.0  # collision with PU
        else:
            reward = 1.0   # successful transmission

        # Switching cost
        if self._prev_action is not None and action != self._prev_action:
            reward -= 0.1

        self._prev_action = action

        # Update history buffer (shift left, append new features)
        new_features = self._build_features()
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        obs = self._get_observation()
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "channel_states": self._channel_states.copy(),
            "collision": channel_occupied,
            "success": not channel_occupied,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        occ = self._channel_states.astype(int)
        bar = " ".join(f"{'X' if o else '.'}" for o in occ)
        print(f"Step {self._step_count:>4d} | Channels: [{bar}]")
