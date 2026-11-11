import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Olmo3Config


class Olmo3RMSNorm(nn.Module):
    """Accumulate normalization and the learned scale in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normalized).to(x.dtype)


class RoPE(nn.Module):
    """Split-half RoPE; YaRN is used only by global attention layers.

    Compute requested positions on demand instead of storing a full context
    cache in every layer. Frequencies and rotations remain float32 even when
    model parameters are converted to bfloat16.
    """

    def __init__(self, config: Olmo3Config, use_yarn: bool = True):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if self.rope_theta <= 1 or self.max_seq_len <= 0:
            raise ValueError("rope_theta must exceed 1 and max_position_embeddings must be positive")

        self.scaling = dict(config.rope_scaling or {}) if use_yarn else {}
        rope_type = self.scaling.get("rope_type", self.scaling.get("type", "default"))
        if rope_type not in ("default", "yarn"):
            raise NotImplementedError("Only default RoPE and YaRN are supported")
        self.use_yarn = rope_type == "yarn"
        self.attention_factor = 1.0
        if self.use_yarn:
            factor = self.scaling["factor"]
            if factor < 1:
                raise ValueError("YaRN factor must be >= 1")
            self.attention_factor = self.scaling.get("attention_factor")
            if self.attention_factor is None:
                self.attention_factor = 1.0 + 0.1 * math.log(factor)
        self.register_buffer("inv_freq", self._frequencies(), persistent=False)

    def _frequencies(self, device=None) -> torch.Tensor:
        indices = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        frequencies = 1.0 / self.rope_theta ** (indices / self.head_dim)
        if not self.use_yarn:
            return frequencies

        # High frequencies extrapolate; low frequencies interpolate by factor.
        original_length = self.scaling.get("original_max_position_embeddings", self.max_seq_len)
        beta_fast = self.scaling.get("beta_fast", 32)
        beta_slow = self.scaling.get("beta_slow", 1)
        if original_length <= 0 or not 0 < beta_slow <= beta_fast:
            raise ValueError("YaRN requires positive original length and 0 < beta_slow <= beta_fast")

        def boundary(rotations):
            return self.head_dim * math.log(original_length / (2 * math.pi * rotations)) / (
                2 * math.log(self.rope_theta)
            )

        low, high = boundary(beta_fast), boundary(beta_slow)
        if self.scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, self.head_dim - 1)
        ramp = ((indices / 2 - low) / (high - low if high != low else 0.001)).clamp(0, 1)
        return frequencies * (1 - ramp) + (frequencies / self.scaling["factor"]) * ramp

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

        inv_freq = self.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = self._frequencies(x.device)
        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos = angles.cos() * self.attention_factor
        sin = angles.sin() * self.attention_factor
        return (x.float() * cos + self._rotate_half(x).float() * sin).to(x.dtype)


class Olmo3MLP(nn.Module):
    """Bias-free SwiGLU feedforward network."""

    def __init__(self, config: Olmo3Config):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only hidden_act='silu' is supported")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Olmo3Attention(nn.Module):
    def __init__(self, config: Olmo3Config, layer_idx: int = 0):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("Attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if config.hidden_size <= 0 or config.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be a positive multiple of num_attention_heads")
        if not 0 <= config.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError("layer_idx must be within num_hidden_layers")
        if config.layer_types is None:
            self.attention_type = "full_attention" if layer_idx % 4 == 3 else "sliding_attention"
        else:
            if len(config.layer_types) != config.num_hidden_layers:
                raise ValueError("layer_types must have num_hidden_layers entries")
            self.attention_type = config.layer_types[layer_idx]

        if self.attention_type not in ("sliding_attention", "full_attention"):
            raise ValueError("Unknown attention type in layer_types")

        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        if self.attention_type == "sliding_attention" and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("Sliding attention requires a positive sliding_window")

        self.head_dim = config.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5

        q_width = self.num_attention_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)

        # Normalize across the entire projection, before splitting into heads.
        self.q_norm = Olmo3RMSNorm(q_width, config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(kv_width, config.rms_norm_eps)
        self.rope = RoPE(config, use_yarn=self.attention_type == "full_attention")


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x)).view(batch, length, self.num_attention_heads, self.head_dim)
        k = self.k_norm(self.k_proj(x)).view(batch, length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim)
        q = self.rope(q.transpose(1, 2), position_ids)
        k = self.rope(k.transpose(1, 2), position_ids)
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.num_key_value_groups, dim=1)

        allowed = None
        if self.sliding_window is not None or attention_mask is not None:
            indices = torch.arange(length, device=x.device)
            distance = indices[:, None] - indices[None, :]
            allowed = (distance >= 0)[None, None]
            if self.sliding_window is not None:
                # Window includes the current token: 0 <= q - k < window.
                allowed = allowed & (distance < self.sliding_window)
            if attention_mask is not None:
                if attention_mask.shape != (batch, length):
                    raise ValueError("attention_mask must have shape [batch, sequence]")
                allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.attention_dropout if self.training else 0.0, scale=self.scaling,
        )
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class Olmo3Block(nn.Module):
    def __init__(self, config: Olmo3Config, layer_idx: int = 0):
        super().__init__()
        self.attention = Olmo3Attention(config, layer_idx)
        self.post_rmsnorm1 = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Olmo3MLP(config)
        self.post_rmsnorm2 = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.post_rmsnorm1(self.attention(x, attention_mask, position_ids))
        return x + self.post_rmsnorm2(self.mlp(x))


class Olmo3(nn.Module):
    def __init__(self, config: Olmo3Config):
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Olmo3Block(config, i) for i in range(config.num_hidden_layers))
        self.final_rmsnorm = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        if x.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, attention_mask, position_ids)
        return self.output_layer(self.final_rmsnorm(x))
