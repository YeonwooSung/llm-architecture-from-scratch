import torch
from torch import nn

from config import LLaMA3_2Config


class TokenEmbedding(nn.Module):
    def __init__(self, config: LLaMA3_2Config):
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

    def __init__(self, config: LLaMA3_2Config):
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

    
class Llama3_2(nn.Module):
    def __init__(self):
        super().__init__()
        #TODO

    def forward(self, x):
        #TODO
        pass
