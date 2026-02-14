"""
Liquid Foundation Model (LFM) Encoder.

Replaces traditional LSTM feature extractors with a stack of Liquid Blocks.
Each Liquid Block has three components:
  A. Adaptive Linear Operator  – input-dependent gated transform
  B. Token Mixing              – multi-head self-attention over time
  C. Channel Mixing            – point-wise FFN with GeLU

The adaptive operator realises the "liquid" dynamics
    y = (A·x + B) ⊙ σ(C·x)
which can be viewed as an analytic solution to a gated linear ODE
    dh/dt = A·h + B, modulated by a sigmoid gate,
evaluated in a single step (no iterative ODE solver).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# A. Adaptive Linear Operator
# ======================================================================

class AdaptiveLinearOperator(nn.Module):
    """
    The "Liquid" core of each block.

    Given input x_t of shape (..., D):
        y_t = (A · x_t + B) ⊙ σ(C · x_t)

    A, B, C are learnable parameter matrices / vectors.
    σ is a gating activation (sigmoid by default).

    Interpretation:
        The term (A·x + B) is an affine transform akin to an Euler step
        of a linear ODE  dh/dt = A·h + B.
        The gate σ(C·x) modulates each dimension, letting the network
        adaptively suppress or amplify features depending on the input,
        which is the hallmark of Liquid Neural Networks.
    """

    def __init__(self, dim: int, gate_activation: str = "sigmoid"):
        super().__init__()
        self.dim = dim

        # Learnable parameters for the affine + gate transform
        self.A = nn.Linear(dim, dim)   # linear map  A·x
        self.B = nn.Parameter(torch.zeros(dim))  # bias / constant drive
        self.C = nn.Linear(dim, dim)   # gate projection  C·x

        if gate_activation == "sigmoid":
            self.gate_fn = torch.sigmoid
        elif gate_activation == "tanh":
            self.gate_fn = torch.tanh
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")

        self._init_weights()

    def _init_weights(self):
        # Small init keeps the gate near 0.5 at the start (balanced)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.zeros_(self.A.bias)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., D)  — same shape as input.
        """
        # Affine branch: A·x + B  (analytic ODE step)
        affine = self.A(x) + self.B

        # Gate branch: σ(C·x)  (input-dependent modulation)
        gate = self.gate_fn(self.C(x))

        # Element-wise gating — the "liquid" adaptive weight
        return affine * gate


# ======================================================================
# B. Token Mixing — Multi-Head Self-Attention over the time axis
# ======================================================================

class TokenMixer(nn.Module):
    """
    Mixes information across the sequence (time) dimension using
    standard scaled-dot-product multi-head self-attention.

    This allows the model to attend to the full history window T
    in parallel, unlike recurrent models.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        residual = x
        x = self.norm(x)

        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj(out)
        return residual + out


# ======================================================================
# C. Channel Mixing — Point-wise FFN with GeLU
# ======================================================================

class ChannelMixer(nn.Module):
    """
    Standard feed-forward network applied independently at each time step.
    Mixes features (channels) using an expand-project MLP with GeLU.
    """

    def __init__(self, dim: int, expand_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expand_factor
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


# ======================================================================
# Liquid Block  =  Adaptive Operator  +  Token Mixer  +  Channel Mixer
# ======================================================================

class LiquidBlock(nn.Module):
    """
    One Liquid Block stacks the three components sequentially:
        1. Adaptive Linear Operator  (liquid gating)
        2. Token Mixer               (temporal self-attention)
        3. Channel Mixer             (point-wise FFN)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        self.adaptive_op = AdaptiveLinearOperator(dim, gate_activation)
        self.adaptive_norm = nn.LayerNorm(dim)
        self.token_mixer = TokenMixer(dim, num_heads, dropout)
        self.channel_mixer = ChannelMixer(dim, ffn_expand, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        # 1. Adaptive linear operator with residual
        x = x + self.adaptive_op(self.adaptive_norm(x))
        # 2. Token mixing (temporal attention)
        x = self.token_mixer(x)
        # 3. Channel mixing (FFN)
        x = self.channel_mixer(x)
        return x


# ======================================================================
# Full LFM Encoder
# ======================================================================

class LFMEncoder(nn.Module):
    """
    Liquid Foundation Model encoder.

    Input:  (Batch, Seq_Len, Features)
    Output: (Batch, Latent_Dim)

    Pipeline:
        1. Linear projection to model dimension.
        2. Learnable positional encoding.
        3. N stacked Liquid Blocks.
        4. Global average pooling over time → latent vector.
        5. Optional projection head to desired latent_dim.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        latent_dim: int = 128,
        num_blocks: int = 3,
        num_heads: int = 4,
        ffn_expand: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        gate_activation: str = "sigmoid",
    ):
        """
        Args:
            input_dim: Per-timestep feature dimension.
            model_dim: Internal representation dimension.
            latent_dim: Output latent vector dimension.
            num_blocks: Number of stacked Liquid Blocks.
            num_heads: Attention heads in Token Mixer.
            ffn_expand: FFN expansion factor in Channel Mixer.
            max_seq_len: Maximum sequence length for positional encoding.
            dropout: Dropout probability.
            gate_activation: Activation for the adaptive gate ("sigmoid" or "tanh").
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            LiquidBlock(model_dim, num_heads, ffn_expand, dropout, gate_activation)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(model_dim)

        # Project to desired latent dimension if different from model_dim
        self.latent_proj: Optional[nn.Linear] = None
        if latent_dim != model_dim:
            self.latent_proj = nn.Linear(model_dim, latent_dim)

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — batch of temporal feature sequences.
        Returns:
            z: (B, latent_dim) — latent representation.
        """
        B, T, F = x.shape

        # 1. Project input features to model dimension
        x = self.input_proj(x)  # (B, T, model_dim)

        # 2. Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        # 3. Pass through Liquid Blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final layer norm
        x = self.final_norm(x)

        # 5. Global average pooling over the time dimension
        z = x.mean(dim=1)  # (B, model_dim)

        # 6. Optional projection to latent_dim
        if self.latent_proj is not None:
            z = self.latent_proj(z)

        return z
