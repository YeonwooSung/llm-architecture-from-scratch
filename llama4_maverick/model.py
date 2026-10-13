import math

import torch
from config import Llama4Config, Llama4TextConfig, Llama4VisionConfig
from torch import nn
from torch.nn import functional as F


def _initialize(module: nn.Module, std: float) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class Llama4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.eps) * self.weight.to(x.dtype)


def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs in [batch, heads, sequence, head_dim]."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    real, imag = pairs.unbind(-1)
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
    return rotated.flatten(-2).to(x.dtype)


class Llama4Attention(nn.Module):
    """iRoPE: chunked causal RoPE layers and full causal NoPE layers."""

    def __init__(self, config: Llama4TextConfig, layer_index: int = 0):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if config.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive")
        if config.rope_scaling is not None:
            raise ValueError("Maverick uses unscaled RoPE; rope_scaling is not supported")

        # These defaults are supplied by Transformers but absent from config.json.
        interval = getattr(config, "no_rope_layer_interval", 4)
        if interval <= 0:
            raise ValueError("no_rope_layer_interval must be positive")
        rope_layers = getattr(config, "no_rope_layers", None)
        self.use_rope = bool(rope_layers[layer_index]) if rope_layers is not None else (layer_index + 1) % interval != 0
        self.floor_scale = getattr(config, "floor_scale", 8192)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.temperature_tuning = bool(getattr(config, "attn_temperature_tuning", True))
        if self.floor_scale <= 0:
            raise ValueError("floor_scale must be positive")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.register_buffer(
            "inv_freq", config.rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        positions = position_ids.to(device=x.device, dtype=torch.float32)
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            angles = positions[..., None] * self.inv_freq.float()
            q, k = _apply_rope(q, angles), _apply_rope(k, angles)
            # Maverick disables Q/K normalization; keep the configurable path.
            if self.config.use_qk_norm:
                q, k = _rms_norm(q, self.config.rms_norm_eps), _rms_norm(k, self.config.rms_norm_eps)
        elif self.temperature_tuning:
            scale = 1 + self.attn_scale * torch.log1p(torch.floor((positions + 1) / self.floor_scale))
            q = (q * scale[:, None, :, None]).to(q.dtype)

        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        indices = torch.arange(length, device=x.device)
        allowed = (indices[:, None] >= indices[None, :])[None, None]
        if self.use_rope:
            # A fixed chunk boundary, not a sliding window. NoPE layers bridge chunks.
            chunks = torch.div(position_ids.to(x.device), self.config.attention_chunk_size, rounding_mode="floor")
            allowed = allowed & (chunks[:, :, None] == chunks[:, None, :])[:, None]
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim))


class Llama4MLP(nn.Module):
    """SwiGLU for dense layers and both types of experts."""

    def __init__(self, config: Llama4TextConfig, intermediate_size: int | None = None):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Llama 4 text feedforward networks require hidden_act='silu'")
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Llama4MoE(nn.Module):
    """Always-on shared expert plus top-k routed experts (top-1 for Maverick)."""

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        if not 1 <= config.num_experts_per_tok <= config.num_local_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4MLP(config)
        self.experts = nn.ModuleList(Llama4MLP(config) for _ in range(config.num_local_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, x.size(-1))
        router_logits = self.router(flat)
        selected_logits, selected = router_logits.topk(self.top_k, dim=-1)
        scores = selected_logits.float().sigmoid().to(x.dtype)
        output = self.shared_expert(flat)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(selected == expert_index)
            if token_indices.numel() == 0:
                continue
            # Llama 4 scales the INPUT, rather than the expert's output.
            routed_input = flat[token_indices] * scores[token_indices, slots, None]
            output = output.index_add(0, token_indices, expert(routed_input))
        return output.view_as(x), router_logits


class Llama4Block(nn.Module):
    def __init__(self, config: Llama4TextConfig, layer_index: int = 0):
        super().__init__()
        if config.interleave_moe_layer_step <= 0:
            raise ValueError("interleave_moe_layer_step must be positive")
        self.input_layernorm = Llama4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Llama4Attention(config, layer_index)
        self.post_attention_layernorm = Llama4RMSNorm(config.hidden_size, config.rms_norm_eps)
        moe_layers = getattr(config, "moe_layers", None)
        self.is_moe_layer = layer_index in moe_layers if moe_layers is not None else (layer_index + 1) % config.interleave_moe_layer_step == 0
        self.feed_forward = Llama4MoE(config) if self.is_moe_layer else Llama4MLP(config, config.intermediate_size_mlp)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        result = self.feed_forward(self.post_attention_layernorm(x))
        result, router_logits = result if self.is_moe_layer else (result, None)
        return x + result, router_logits


class Llama4VisionAttention(nn.Module):
    """Bidirectional attention with axial 2D RoPE; the final token is CLS."""

    def __init__(self, config: Llama4VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        if self.num_heads <= 0 or config.hidden_size % self.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_attention_heads")
        self.head_dim = config.hidden_size // self.num_heads
        if self.head_dim % 4:
            raise ValueError("vision head_dim must be divisible by 4 for axial RoPE")
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = [projection(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                   for projection in (self.q_proj, self.k_proj, self.v_proj)]
        q, k = _apply_rope(q, angles), _apply_rope(k, angles)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, width))


class Llama4VisionBlock(nn.Module):
    def __init__(self, config: Llama4VisionConfig):
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError("Llama 4 vision requires hidden_act='gelu'")
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = Llama4VisionAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), angles)
        return x + self.fc2(F.gelu(self.fc1(self.post_attention_layernorm(x))))


class Llama4VisionAdapter(nn.Module):
    """Combine neighboring patches, then project using a two-layer GELU MLP."""

    def __init__(self, config: Llama4VisionConfig):
        super().__init__()
        ratio = config.pixel_shuffle_ratio
        if not 0 < ratio <= 1 or not math.isclose(1 / ratio, round(1 / ratio)):
            raise ValueError("pixel_shuffle_ratio must be the reciprocal of a positive integer")
        self.factor = round(1 / ratio)
        self.fc1 = nn.Linear(config.hidden_size * self.factor ** 2, config.projector_input_dim, bias=False)
        self.fc2 = nn.Linear(config.projector_input_dim, config.projector_output_dim, bias=False)
        self.dropout = config.projector_dropout
        if config.projector_output_dim != config.vision_output_dim:
            raise ValueError("projector_output_dim must equal vision_output_dim")

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, channels = x.shape
        side = math.isqrt(patches)
        factor = self.factor
        if side * side != patches or side % factor:
            raise ValueError("Pixel shuffle requires a square patch grid divisible by the shuffle factor")
        # Preserve Llama 4's row-offset, column-offset, channel ordering.
        x = x.reshape(batch, side, side // factor, channels * factor).transpose(1, 2).contiguous()
        x = x.reshape(batch, side // factor, side // factor, channels * factor ** 2)
        return x.transpose(1, 2).reshape(batch, patches // factor ** 2, channels * factor ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(self.pixel_shuffle(x)))
        return F.gelu(self.fc2(F.dropout(x, self.dropout, training=self.training)))


class Llama4VisionEncoder(nn.Module):
    """Encode independent fixed-size image tiles to compressed image tokens."""

    def __init__(self, config: Llama4VisionConfig):
        super().__init__()
        self.config = config
        if config.patch_size <= 0 or config.image_size <= 0 or config.image_size % config.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if config.vision_feature_select_strategy != "default":
            raise ValueError("Llama 4 pixel shuffle requires vision_feature_select_strategy='default'")
        side = config.image_size // config.patch_size
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size ** 2, config.hidden_size, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.positional_embedding = nn.Parameter(torch.empty(side * side + 1, config.hidden_size))
        self.layernorm_pre = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = nn.ModuleList(Llama4VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.layernorm_post = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.vision_adapter = Llama4VisionAdapter(config)
        if side % self.vision_adapter.factor:
            raise ValueError("Patch grid must be divisible by the pixel shuffle factor")
        self.tokens_per_image = (side // self.vision_adapter.factor) ** 2
        self.feature_layer = config.vision_feature_layer
        if not -(len(self.layers) + 1) <= self.feature_layer <= len(self.layers):
            raise ValueError("vision_feature_layer is outside the vision hidden states")

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = config.rope_theta ** (-torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
        coordinates = torch.arange(side * side)
        angles = torch.cat(((coordinates % side + 1)[:, None] * inv_freq,
                            (coordinates // side + 1)[:, None] * inv_freq), -1)
        # CLS has identity rotation (zero angles).
        self.register_buffer("rope_angles", F.pad(angles, (0, 0, 0, 1)).unsqueeze(0), persistent=False)
        self.apply(lambda module: _initialize(module, config.initializer_range))
        nn.init.normal_(self.class_embedding, std=config.hidden_size ** -0.5)
        nn.init.normal_(self.positional_embedding, std=config.hidden_size ** -0.5)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (cfg.num_channels, cfg.image_size, cfg.image_size):
            raise ValueError(f"pixel_values must have shape [images, {cfg.num_channels}, {cfg.image_size}, {cfg.image_size}]")
        patches = F.unfold(pixel_values, kernel_size=cfg.patch_size, stride=cfg.patch_size).transpose(1, 2)
        x = self.patch_embedding(patches)
        x = torch.cat((x, self.class_embedding.to(x.dtype).expand(x.size(0), 1, -1)), 1)
        x = self.layernorm_pre(x + self.positional_embedding.to(x.dtype))
        feature_index = self.feature_layer % (len(self.layers) + 1)
        selected = x if feature_index == 0 else None
        for index, layer in enumerate(self.layers, 1):
            x = layer(x, self.rope_angles.float())
            if index == feature_index:
                selected = x
        return self.vision_adapter(self.layernorm_post(selected)[:, :-1])


class Llama4MultiModalProjector(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.vision_output_dim, config.text_config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Llama4MaverickModel(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        self.config = config
        cfg = config.text_config
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Llama4Block(cfg, index) for index in range(cfg.num_hidden_layers))
        self.norm = Llama4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.apply(lambda module: _initialize(module, cfg.initializer_range))
        self.vision_encoder = Llama4VisionEncoder(config.vision_config)
        if getattr(cfg, "tie_word_embeddings", False):
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return [batch, sequence, vocabulary] logits, optionally with router logits.

        Image tiles and their placeholder runs must have matching row-major order
        across the batch. Each default 336px tile consumes 144 image placeholders.
        attention_mask is [batch, sequence], with 1/True marking valid keys.
        position_ids defaults to 0..sequence-1; pass explicit positions for padding.
        """
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        if input_ids.size(1) > self.config.text_config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        image_mask = input_ids == self.config.image_token_index
        x = self.token_embedding(input_ids)
        if pixel_values is not None:
            if pixel_values.ndim != 4:
                raise ValueError("pixel_values must have shape [images, channels, height, width]")
            available = pixel_values.size(0) * self.vision_encoder.tokens_per_image
            if int(image_mask.sum().item()) != available:
                raise ValueError(f"Expected {available} image placeholders for the supplied tiles")
            features = self.multi_modal_projector(self.vision_encoder(pixel_values))
            x = x.clone()
            x[image_mask] = features.reshape(-1, x.size(-1)).to(x.dtype)
        elif image_mask.any():
            raise ValueError("Image placeholder tokens require pixel_values")
        collect_router = self.config.text_config.output_router_logits if output_router_logits is None else output_router_logits
        router_outputs = []
        for layer in self.layers:
            x, router_logits = layer(x, attention_mask, position_ids)
            if collect_router and router_logits is not None:
                router_outputs.append(router_logits)
        logits = self.lm_head(self.norm(x))
        return (logits, tuple(router_outputs)) if collect_router else logits
