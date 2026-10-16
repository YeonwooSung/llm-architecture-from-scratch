"""SmolLM3 decoder with grouped-query attention and interleaved RoPE/NoPE."""

import torch
from torch import nn
from torch.nn import functional as F

from config import SmolLM3Config


class SmolLM3RMSNorm(nn.Module):
    """Normalize the last dimension using float32 variance accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RoPE(nn.Module):
    """Apply split-half rotary embeddings to [batch, heads, sequence, head_dim]."""

    def __init__(self, config: SmolLM3Config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if config.rope_scaling is not None:
            raise NotImplementedError("Only default RoPE (rope_scaling=None) is supported")
        if config.rope_theta <= 0 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta and max_position_embeddings must be positive")
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings
        # Store frequencies only; a 65,536-position table per layer is unnecessary.
        self.register_buffer(
            "inv_freq",
            self.rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("Sequence exceeds max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if (position_ids < 0).any() or (position_ids >= self.max_seq_len).any():
            raise ValueError("position_ids must be within max_position_embeddings")

        # Recover float32 frequencies after model.to(bfloat16/float16).
        inv_freq = self.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = self.rope_theta ** (
                -torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim
            )
        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        return x * angles.cos().to(x.dtype) + self._rotate_half(x) * angles.sin().to(x.dtype)


class SmolLM3MLP(nn.Module):
    """Bias-free SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: SmolLM3Config):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only hidden_act='silu' is supported")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SmolLM3Attention(nn.Module):
    """Causal GQA; every fourth layer omits rotary position embeddings."""

    def __init__(self, config: SmolLM3Config, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if config.hidden_size <= 0 or config.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_attention_heads")
        if config.no_rope_layer_interval <= 0:
            raise ValueError("no_rope_layer_interval must be positive")
        if not 0 <= config.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")

        self.head_dim = config.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout
        self.layer_idx = layer_idx

        # One-based layers 4, 8, 12, ... use NoPE; causality still applies.
        self.use_rope = (layer_idx + 1) % config.no_rope_layer_interval != 0
        self.rope = RoPE(config) if self.use_rope else None
        self.window = None
        if config.use_sliding_window:
            if config.sliding_window is None or config.sliding_window <= 0:
                raise ValueError("use_sliding_window requires a positive sliding_window")
            # The optional HF sliding-window configuration targets NoPE layers.
            if not self.use_rope:
                self.window = config.sliding_window

        # SmolLM3 does not apply the Q/K normalization used by Qwen3 or OLMo2.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, sequence, hidden_size] using an optional [batch, sequence] key mask."""
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (Rotary Positional Embedding) if enabled for this layer.
        # For every 4 layers, NoPE is applied instead of RoPE.
        if self.use_rope:
            q = self.rope(q, position_ids)
            k = self.rope(k, position_ids)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        allowed = None
        if attention_mask is not None or self.window is not None:
            indices = torch.arange(length, device=x.device)
            distance = indices[:, None] - indices[None, :]
            allowed = (distance >= 0)[None, None]
            if self.window is not None:
                allowed = allowed & (distance < self.window)
            if attention_mask is not None:
                if attention_mask.shape != (batch, length):
                    raise ValueError("attention_mask must have shape [batch, sequence]")
                allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]

        # SDPA also handles fully masked queries from left-padded sequences.
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(output)


class SmolLM3Block(nn.Module):
    """Pre-RMSNorm attention and SwiGLU, each followed by a residual addition."""

    def __init__(self, config: SmolLM3Config, layer_idx: int = 0):
        super().__init__()

        self.input_layernorm = SmolLM3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SmolLM3Attention(config, layer_idx)
        self.post_attention_layernorm = SmolLM3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SmolLM3MLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


class SmolLM3Model(nn.Module):
    """SmolLM3 language model returning logits, following the repository's model API.

    This implementation processes full sequences; KV caching is not implemented.
    """

    def __init__(self, config: SmolLM3Config):
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(SmolLM3Block(config, i) for i in range(config.num_hidden_layers))
        self.norm = SmolLM3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            # The input embedding and vocabulary projection share one parameter.
            self.lm_head.weight = self.token_embedding.weight


    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, sequence, vocab_size] logits for integer token IDs.

        attention_mask is [batch, sequence], with 1/True for valid keys.
        position_ids optionally gives [1 or batch, sequence] rotary positions;
        otherwise positions start at zero, including padding tokens.
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        if x.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        return self.lm_head(self.norm(x))
