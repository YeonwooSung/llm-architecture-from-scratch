import torch
from torch import nn
from torch.nn import functional as F

from config import OuroThinkingConfig


class OuroThinkingRMSNorm(nn.Module):
    """RMSNorm with float32 variance and an ordinary learned scale (initially 1)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RoPE(nn.Module):
    """Split-half RoPE; compute one set of cos/sin values for every forward call."""

    def __init__(self, config: OuroThinkingConfig):
        super().__init__()
        self.head_dim = config.head_dim
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if config.rope_theta <= 0 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta and max_position_embeddings must be positive")
        if config.rope_scaling is not None:
            raise NotImplementedError("Only default RoPE (rope_scaling=None) is supported")
        self.theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings
        self.register_buffer(
            "inv_freq", self.theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        batch, length, _ = x.shape
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

        # Rebuild frequencies if model.to(dtype) rounded the nonpersistent buffer.
        inv_freq = self.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = self.theta ** (-torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim)
        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin


class OuroThinkingMLP(nn.Module):
    """Bias-free SwiGLU, with intermediate width 5,632 in the 2.6B model."""

    def __init__(self, config: OuroThinkingConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only hidden_act='silu' is supported")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OuroThinkingAttention(nn.Module):
    """Causal MHA (16 Q and KV heads by default), without Q/K normalization."""

    def __init__(self, config: OuroThinkingConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if not 0 <= config.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        layer_type = config.layer_types[layer_idx]
        if layer_type not in ("full_attention", "sliding_attention"):
            raise ValueError(f"Unsupported layer type: {layer_type}")
        self.window = config.sliding_window if layer_type == "sliding_attention" else None
        if layer_type == "sliding_attention" and (self.window is None or self.window <= 0):
            raise ValueError("sliding_attention requires a positive sliding_window")

        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)


    def forward(self, x, position_embeddings, attention_mask=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = RoPE.apply_rotary(q, cos, sin), RoPE.apply_rotary(k, cos, sin)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)

        allowed = None
        if attention_mask is not None or self.window is not None:
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            allowed = (distance >= 0)[None, None]
            if self.window is not None:
                allowed = allowed & (distance < self.window)
            if attention_mask is not None:
                if attention_mask.shape != (batch, length):
                    raise ValueError("attention_mask must have shape [batch, sequence]")
                allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(result.transpose(1, 2).reshape(batch, length, -1))


class OuroThinkingBlock(nn.Module):
    """Sandwich normalization: pre/post RMSNorm around each residual branch."""

    def __init__(self, config: OuroThinkingConfig, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = OuroThinkingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = OuroThinkingAttention(config, layer_idx)
        self.input_layernorm_2 = OuroThinkingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = OuroThinkingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = OuroThinkingMLP(config)
        self.post_attention_layernorm_2 = OuroThinkingRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, position_embeddings, attention_mask=None):
        x = x + self.input_layernorm_2(self.attention(self.input_layernorm(x), position_embeddings, attention_mask))
        return x + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x)))



class OuroThinkingModel(nn.Module):
    def __init__(self, config: OuroThinkingConfig):
        super().__init__()
        if config.num_hidden_layers <= 0 or config.total_ut_steps <= 0:
            raise ValueError("num_hidden_layers and total_ut_steps must be positive")
        if len(config.layer_types) < config.num_hidden_layers:
            raise ValueError("layer_types must describe every decoder layer")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RoPE(config)
        self.layers = nn.ModuleList(OuroThinkingBlock(config, i) for i in range(config.num_hidden_layers))
        self.final_rmsnorm = OuroThinkingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1, bias=True)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.output_layer.weight = self.token_embedding.weight


    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None, *, exit_at_step: int | None = None,
        exit_threshold: float | None = None, use_weighted_exit: bool = False,
    ) -> torch.Tensor:
        """
        Process token IDs with optional padding mask and explicit RoPE positions.

        exit_at_step is zero-based and takes precedence over the threshold.
        Otherwise exit_threshold defaults to config.early_exit_threshold (1.0:
        final pass). As in HF, weighted exit is used only if the threshold is None;
        set config.early_exit_threshold=None to enable it. No selection options
        means the last pass. Masks are [B, S], positions are [1 or B, S].
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        steps = self.config.total_ut_steps
        threshold = self.config.early_exit_threshold if exit_threshold is None else exit_threshold
        if exit_at_step is not None and not 0 <= exit_at_step < steps:
            raise ValueError("exit_at_step must be within [0, total_ut_steps)")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("exit_threshold must be between 0 and 1")

        x = self.token_embedding(x)
        position_embeddings = self.rope(x, position_ids)
        states, probabilities = [], []
        remaining = torch.ones(x.shape[:2], device=x.device, dtype=torch.float32)
        for step in range(steps):
            for layer in self.layers:
                x = layer(x, position_embeddings, attention_mask)
            x = self.final_rmsnorm(x)
            states.append(x)
            halt = self.early_exit_gate(x).squeeze(-1).float().sigmoid()
            probabilities.append(remaining * halt if step < steps - 1 else remaining)
            remaining = remaining * (1 - halt)

        if exit_at_step is not None:
            return self.output_layer(states[exit_at_step])
        if threshold is None and use_weighted_exit:
            # A bias-free linear head commutes with the weighted hidden-state sum.
            weighted = sum(state * prob.unsqueeze(-1).to(state.dtype) for state, prob in zip(states, probabilities))
            return self.output_layer(weighted)
        if threshold is None or threshold == 1.0:
            return self.output_layer(states[-1])

        cumulative = torch.stack(probabilities, dim=-1).cumsum(-1)
        crossed = cumulative >= threshold
        indices = crossed.int().argmax(-1)
        indices = torch.where(crossed.any(-1), indices, steps - 1)
        stacked = torch.stack(states, dim=2)
        selected = stacked.gather(2, indices[..., None, None].expand(-1, -1, 1, x.size(-1))).squeeze(2)
        return self.output_layer(selected)
