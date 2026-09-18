import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Olmo2Config


class TokenEmbedding(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class RoPE(nn.Module):
    """
    Rotary Positional Embedding for OLMo 2.

    Expected input shape:
        [batch_size, num_heads, seq_len, head_dim]

    OLMo 2 follows the Llama-style RoPE convention:

        x = [x1, x2]

        rotate_half(x) = [-x2, x1]

    where x1 and x2 are the first and second halves
    of the head dimension.
    """

    def __init__(self, config: Olmo2Config):
        super().__init__()

        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        self.max_seq_len = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got {self.head_dim}"
            )

        # ---------------------------------------------------------
        # 1. Compute inverse frequencies
        #
        # inv_freq:
        #   [head_dim / 2]
        #
        # Example:
        #
        #   head_dim = 128
        #
        #   -> 64 frequencies
        #
        # θ_i = 1 / base^(2i / head_dim)
        # ---------------------------------------------------------

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

        # ---------------------------------------------------------
        # 2. Precompute cos / sin cache
        # ---------------------------------------------------------

        positions = torch.arange(
            self.max_seq_len,
            dtype=torch.float32,
        )

        # [seq_len, head_dim / 2]
        freqs = torch.outer(
            positions,
            self.inv_freq,
        )

        # Llama / OLMo style:
        #
        # [θ0, θ1, θ2, ...]
        #
        # becomes
        #
        # [θ0, θ1, θ2, ..., θ0, θ1, θ2, ...]
        #
        # Shape:
        # [seq_len, head_dim]
        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        )

        self.register_buffer(
            "cos",
            emb.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            emb.sin(),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Llama / OLMo-style half rotation.

        If:

            x = [x1, x2]

        then:

            rotate_half(x) = [-x2, x1]

        Example:

            [a, b, c, d]

        becomes:

            [-c, -d, a, b]
        """

        half_dim = x.shape[-1] // 2

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        return torch.cat(
            (-x2, x1),
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings.

        Args:
            x:
                Tensor with shape:

                    [B, H, S, D]

            position_ids:
                Optional position indices.

                Shape:
                    [B, S]

                If None, positions are assumed to be:

                    [0, 1, 2, ..., S - 1]

        Returns:
            Tensor with shape:

                [B, H, S, D]
        """

        seq_len = x.size(-2)

        if position_ids is None:
            # -----------------------------------------------------
            # Normal full-sequence forward
            # -----------------------------------------------------

            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds "
                    f"max_position_embeddings={self.max_seq_len}"
                )

            # [S, D]
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]

            # [1, 1, S, D]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

        else:
            # -----------------------------------------------------
            # Explicit positions
            #
            # Useful for KV-cache based autoregressive decoding.
            # -----------------------------------------------------

            if position_ids.max().item() >= self.max_seq_len:
                raise ValueError(
                    f"position_ids contains a position >= "
                    f"max_position_embeddings={self.max_seq_len}"
                )

            # self.cos:
            #   [max_seq_len, D]
            #
            # position_ids:
            #   [B, S]
            #
            # result:
            #   [B, S, D]

            cos = self.cos[position_ids]
            sin = self.sin[position_ids]

            # [B, 1, S, D]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Match the dtype of Q / K.
        cos = cos.to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = sin.to(
            device=x.device,
            dtype=x.dtype,
        )

        # ---------------------------------------------------------
        # Standard rotary transformation
        #
        # x_rotated =
        #
        #   x * cos(theta)
        #   +
        #   rotate_half(x) * sin(theta)
        #
        # ---------------------------------------------------------

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


class Olmo2Attention(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads,)

        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.rope = RoPE(config)

        # OLMo2 applies QK norm before splitting into heads.
        self.q_norm = nn.RMSNorm(
            config.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = nn.RMSNorm(
            config.num_key_value_heads * self.head_dim,
            eps=config.rms_norm_eps,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [batch_size, seq_len, hidden_size]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V projection
        # Q: [B, S, num_heads * head_dim]
        # K: [B, S, num_kv_heads * head_dim]
        # V: [B, S, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Split into attention heads
        # [B, S, H, D] -> [B, H, S, D]
        q = q.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim,
        ).transpose(1, 2)

        # Rotary positional embedding
        q = self.rope(q)
        k = self.rope(k)

        # Expand KV heads for GQA, when necessary
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )
            v = v.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )

        # Scaled dot-product causal attention
        attention_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scaling,
        )

        # attention_output:
        # [B, num_heads, S, head_dim]

        # Merge heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_attention_heads * self.head_dim,
        )

        return self.o_proj(attention_output)


class Olmo2Block(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        # Multi-head attention with QKNorm
        self.attention = Olmo2Attention(config)

        # Post RMSNorm 1
        self.post_rmsnorm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        # MLP
        self.mlp = Olmo2MLP(config)
        # Post RMSNorm 2
        self.post_rmsnorm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attention(x)
        x = self.post_rmsnorm1(x)
        x = residual + x
        residual = x
        x = self.mlp(x)
        x = self.post_rmsnorm2(x)
        return x + residual


class Olmo2(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # olmo2 blocks
        self.blocks = nn.ModuleList([Olmo2Block(config) for _ in range(config.num_hidden_layers)])

        # final RMSNorm before output
        self.final_rmsnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through token embedding
        x = self.token_embedding(x)

        # Forward pass through olmo2 blocks
        for block in self.blocks:
            x = block(x)

        # Forward pass through final RMSNorm
        x = self.final_rmsnorm(x)

        # Forward pass through output layer
        x = self.output_layer(x)

        return x
