import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Llama3Config


class TokenEmbedding(nn.Module):
    def __init__(self, config: Llama3Config):
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

    def __init__(self, config: Llama3Config):
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


class MaskedGroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention used by Llama 3.

    Llama 3 8B:
        num_heads = 32
        num_kv_heads = 8

    Therefore:

        4 query heads share one K/V head.
    """

    def __init__(self, config: Llama3Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.rope = RoPE(config)

        # Llama does not use bias in attention projections.
        self.w_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False,)
        self.w_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,)
        self.w_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False,)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Input:
            [B, H_kv, S, D]

        Output:
            [B, H_q, S, D]

        Example:

            8 KV heads
                ↓ repeat x4
            32 attention heads
        """

        if n_rep == 1:
            return x

        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        x = x[:, :, None, :, :]
        x = x.expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim,)

        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim,)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns:

            [1, 1, S, S]

        True means attention is allowed.
        """

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device,)
        mask = torch.tril(mask)

        return mask.unsqueeze(0).unsqueeze(0)

    def _prepare_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Combines the causal mask with an optional padding mask.

        Supported attention_mask forms:

            [B, S]
            [S, S]
            [B, 1, 1, S]
            [B, 1, S, S]

        Values:
            True / 1 = allowed
            False / 0 = masked
        """

        causal_mask = self._make_causal_mask(seq_len, device,)

        if attention_mask is None:
            return causal_mask

        attention_mask = attention_mask.to(device=device, dtype=torch.bool,)

        if attention_mask.ndim == 2:
            # Padding mask: [B, S]
            if attention_mask.shape == (batch_size, seq_len):
                attention_mask = attention_mask[:, None, None, :]

            # Explicit attention matrix: [S, S]
            elif attention_mask.shape == (seq_len, seq_len):
                attention_mask = attention_mask[None, None, :, :]

            else:
                raise ValueError(
                    f"Unsupported 2D attention mask shape: {attention_mask.shape}"
                )

        elif attention_mask.ndim != 4:
            raise ValueError(
                f"attention_mask must have 2 or 4 dimensions, got shape {attention_mask.shape}"
            )

        return causal_mask & attention_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            [B, S, hidden_size]
        """

        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K and V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Split into attention heads.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim,)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim,)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim,)

        # Convert [B, S, H, D] to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Apply RoPE to Q and K. V does not receive positional encoding.
        q = self.rope(q)
        k = self.rope(k)

        # 4. Expand KV heads for GQA (each K/V head is shared by 4 query heads).
        k = self._repeat_kv(k, self.num_kv_groups,)
        v = self._repeat_kv(v, self.num_kv_groups,)

        # q, k, v: [B, num_heads, S, head_dim]

        # 5. Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1),)
        scores = scores / math.sqrt(self.head_dim)

        # 6. Causal + padding mask
        mask = self._prepare_mask(attention_mask, batch_size, seq_len, x.device,)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min,)

        # Softmax is safer in float32.
        attn_weights = F.softmax(scores.float(), dim=-1,).to(dtype=q.dtype)

        # 7. Weighted sum of values
        output = torch.matmul(attn_weights, v,)

        # [B, H, S, D] -> [B, S, H, D]
        output = output.transpose(1, 2).contiguous()

        # Merge heads: [B, S, H * D]
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim,)

        return self.w_o(output)


class FeedForwardNetwork(nn.Module):
    """
    Llama SwiGLU feed-forward network.

    FFN(x) =
        W_down(
            SiLU(W_gate(x)) * W_up(x)
        )
    """

    def __init__(self, config: Llama3Config):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False,)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False,)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Llama3Block(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()

        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        self.attention = MaskedGroupedQueryAttention(config)

        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        self.feed_forward = FeedForwardNetwork(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention: x = x + Attention(RMSNorm(x))
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask,)
        x = residual + x

        # Pre-norm FFN: x = x + FFN(RMSNorm(x))
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x


class Llama3(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()

        self.config = config
        self.token_embedding = TokenEmbedding(config)

        # ModuleList instead of Sequential because every layer needs attention_mask.
        self.layers = nn.ModuleList(
            [Llama3Block(config) for _ in range(config.num_hidden_layers)]
        )

        # Llama has a final RMSNorm.
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)

        # Language-model head: hidden_size -> vocabulary logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False,)

        # Optional embedding/output weight tying (not required by original Llama 3).
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids:
            [B, S]

        attention_mask:
            optional [B, S]

        returns:
            logits [B, S, vocab_size]
        """
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask,)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
