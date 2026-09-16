import math

import torch
import torch.nn as nn

from config import Olmo2Config


class TokenEmbedding(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class RoPE(nn.Module):
    """
    Rotary Positional Embedding.

    Expected input shape:
        [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self, config: Olmo2Config):
        super().__init__()

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_position_embeddings

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

        # Llama 3 uses 500,000.
        theta = config.rope_theta

        # [head_dim / 2]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        # Llama 3.2 rescales low/high frequencies to extend context length beyond pretraining.
        if config.rope_scaling is not None:
            inv_freq = self._apply_rope_scaling(inv_freq, config.rope_scaling)

        # [max_seq_len]
        positions = torch.arange(self.max_seq_len, dtype=torch.float32,)

        # [max_seq_len, head_dim / 2]
        angles = torch.outer(positions, inv_freq)

        # Meta-style RoPE uses adjacent pairs (x0, x1), (x2, x3), ... therefore
        # each angle is repeated twice: [theta0, theta0, theta1, theta1, ...]
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1,)

        self.register_buffer("cos", angles.cos(), persistent=False,)
        self.register_buffer("sin", angles.sin(), persistent=False,)

    @staticmethod
    def _apply_rope_scaling(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
        """
        Llama 3 "rope_type": "llama3" scaling.

        Low frequencies (long wavelengths) are divided by `factor`, high
        frequencies are left untouched, and the band in between is smoothly
        interpolated. This lets the model extrapolate to longer context
        lengths than it was pretrained on.
        """

        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        old_context_len = rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq

        # wavelen < high_freq_wavelen: keep as-is. wavelen > low_freq_wavelen: divide by factor.
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            smooth_factor * inv_freq_llama / factor + (1 - smooth_factor) * inv_freq_llama
        )

        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)

        return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        For adjacent pairs:

            [x0, x1, x2, x3]

        becomes:

            [-x1, x0, -x3, x2]
        """

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotated = torch.stack((-x_odd, x_even), dim=-1,)

        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            [B, H, S, D]
        """

        seq_len = x.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_sequence_length={self.max_seq_len}"
            )

        # [1, 1, S, D]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)

        return x * cos + self._rotate_half(x) * sin


class Olmo2MLP(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Olmo2(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        #TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO
        return x
