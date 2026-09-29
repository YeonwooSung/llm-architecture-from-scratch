import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Phi4Config


class RoPE(nn.Module):
    """Rotary positional embeddings used by Phi-4's Phi-3 backbone."""

    def __init__(self, config: Phi4Config):
        super().__init__()
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_position_embeddings = config.max_position_embeddings

        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(
                "The rotary dimension must be a positive even number, "
                f"got {self.rotary_dim}"
            )

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        positions = torch.arange(
            self.max_position_embeddings, dtype=torch.float32
        )
        frequencies = torch.outer(positions, inv_freq)
        # Phi-3/Phi-4 uses split-half rotation, so frequencies are duplicated.
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first_half, second_half = x.chunk(2, dim=-1)
        return torch.cat((-second_half, first_half), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply RoPE to ``x`` with shape ``[batch, heads, seq, head_dim]``."""
        batch_size, _, seq_len, _ = x.shape

        if position_ids is None:
            if seq_len > self.max_position_embeddings:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_position_embeddings="
                    f"{self.max_position_embeddings}"
                )
            cos = self.cos[:seq_len].view(1, 1, seq_len, self.rotary_dim)
            sin = self.sin[:seq_len].view(1, 1, seq_len, self.rotary_dim)
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.ndim != 2 or position_ids.shape[-1] != seq_len:
                raise ValueError(
                    "position_ids must have shape [seq_len] or [batch, seq_len], "
                    f"got {tuple(position_ids.shape)}"
                )
            if position_ids.shape[0] not in (1, batch_size):
                raise ValueError(
                    "position_ids batch dimension must be 1 or match the input "
                    f"batch size ({batch_size}), got {position_ids.shape[0]}"
                )
            position_ids = position_ids.to(device=self.cos.device, dtype=torch.long)
            if torch.any(position_ids < 0) or torch.any(
                position_ids >= self.max_position_embeddings
            ):
                raise ValueError("position_ids contains an out-of-range position")
            cos = self.cos[position_ids].unsqueeze(1)
            sin = self.sin[position_ids].unsqueeze(1)

        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        x_rotary = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_rotary = x_rotary * cos + self._rotate_half(x_rotary) * sin
        return torch.cat((x_rotary, x_pass), dim=-1)


class Phi4Attention(nn.Module):
    """Causal grouped-query self-attention with fused QKV projection."""

    def __init__(self, config: Phi4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            q_size + 2 * kv_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(q_size, self.hidden_size, bias=False)
        self.rotary_embedding = RoPE(config)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
        if repeats == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(
            batch_size, num_kv_heads, repeats, seq_len, head_dim
        )
        return x.reshape(batch_size, num_kv_heads * repeats, seq_len, head_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        query_positions = torch.arange(seq_len, device=device)[:, None]
        key_positions = torch.arange(seq_len, device=device)[None, :]
        mask = key_positions <= query_positions
        if self.sliding_window is not None:
            if self.sliding_window <= 0:
                raise ValueError("sliding_window must be positive when provided")
            mask &= key_positions > query_positions - self.sliding_window
        return mask.view(1, 1, seq_len, seq_len)

    def _prepare_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = self._causal_mask(seq_len, device)
        if attention_mask is None:
            return mask

        attention_mask = attention_mask.to(device=device)
        # Accept both the usual 1/0 keep mask and an additive mask where 0 is
        # allowed and negative values represent masked positions.
        if torch.is_floating_point(attention_mask) and torch.any(attention_mask < 0):
            attention_mask = attention_mask >= 0
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)
        if attention_mask.ndim == 2:
            if attention_mask.shape == (batch_size, seq_len):
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (seq_len, seq_len):
                attention_mask = attention_mask[None, None, :, :]
            else:
                raise ValueError(
                    "A 2D attention_mask must have shape [batch, seq_len] or "
                    f"[seq_len, seq_len], got {tuple(attention_mask.shape)}"
                )
        elif attention_mask.ndim == 4:
            expected = (batch_size, 1, seq_len, seq_len)
            padding_expected = (batch_size, 1, 1, seq_len)
            if attention_mask.shape not in (expected, padding_expected):
                raise ValueError(
                    "A 4D attention_mask must have shape [batch, 1, 1, seq_len] "
                    "or [batch, 1, seq_len, seq_len], got "
                    f"{tuple(attention_mask.shape)}"
                )
        else:
            raise ValueError(
                "attention_mask must be 2D or 4D, "
                f"got {attention_mask.ndim} dimensions"
            )
        return mask & attention_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            raise ValueError("Phi4Attention does not support an empty sequence")

        qkv = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.rotary_embedding(query, position_ids)
        key = self.rotary_embedding(key, position_ids)
        key = self._repeat_kv(key, self.num_kv_groups)
        value = self._repeat_kv(value, self.num_kv_groups)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        mask = self._prepare_mask(attention_mask, batch_size, seq_len, x.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(
            weights, p=self.attention_dropout, training=self.training
        )
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.o_proj(output)


class Phi4MLP(nn.Module):
    """Phi-4's fused gate/up SwiGLU feed-forward network."""

    def __init__(self, config: Phi4Config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Only the Phi-4 SiLU activation is supported, got {config.hidden_act!r}"
            )
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Phi4Block(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.config = config
        self.input_rmsnorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention = Phi4Attention(config)
        self.post_attention_rmsnorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Phi4MLP(config)
        self.residual_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.residual_dropout(
            self.self_attention(self.input_rmsnorm(x), attention_mask, position_ids)
        )
        x = x + self.residual_dropout(self.mlp(self.post_attention_rmsnorm(x)))
        return x


class Phi4Model(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embedding_dropout = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList(
            [Phi4Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_rmsnorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.output_layer = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.output_layer.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-token logits for input token IDs shaped ``[batch, seq]``."""
        if x.ndim != 2:
            raise ValueError(
                f"input token IDs must have shape [batch, seq_len], got {tuple(x.shape)}"
            )
        hidden_states = self.embedding_dropout(self.token_embedding(x))
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, position_ids)
        hidden_states = self.final_rmsnorm(hidden_states)
        return self.output_layer(hidden_states)
