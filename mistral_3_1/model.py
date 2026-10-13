"""Mistral Small 3.1 text decoder and Pixtral vision encoder, in plain PyTorch.

Like the other models in this repository, forward processes a full sequence and
returns logits. It does not implement a KV cache, tokenization, or image resizing.
Architecture reference: Hugging Face Transformers v4.51.3 Mistral3/Pixtral.
"""

import torch
from torch import nn
from torch.nn import functional as F

from config import Mistral3Config


class Mistral3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # angles: [batch or 1, sequence, head_dim / 2]
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return x * angles.cos().to(x.dtype) + _rotate_half(x) * angles.sin().to(x.dtype)


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported activation: {name}")


class Mistral3MLP(nn.Module):
    """Bias-free gated feedforward network used by both text and vision blocks."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.hidden_act not in ("silu", "gelu"):
            raise ValueError("hidden_act must be 'silu' or 'gelu'")
        self.hidden_act = cfg.hidden_act
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_activation(self.hidden_act, self.gate_proj(x)) * self.up_proj(x))


class Mistral3Attention(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        cfg = config.text_config

        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if cfg.sliding_window is not None and cfg.sliding_window <= 0:
            raise ValueError("sliding_window must be positive or None")

        self.window = cfg.sliding_window
        self.dropout = cfg.attention_dropout

        # The attention width (32 * 128) differs from hidden_size (5120).
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.register_buffer(
            "inv_freq", cfg.rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        elif position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")

        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * self.inv_freq.float()
        q, k = _apply_rope(q, angles), _apply_rope(k, angles)
        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)

        row = torch.arange(length, device=x.device)
        distance = row[:, None] - row[None, :]
        allowed = (distance >= 0)[None, None]
        if self.window is not None:
            allowed = allowed & (distance < self.window)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        scores = (q @ k.transpose(-2, -1)).float() * self.head_dim ** -0.5
        weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        # A fully padded query has no permitted keys; its attention contribution is zero.
        weights = weights.masked_fill(~allowed.any(-1, keepdim=True), 0).to(v.dtype)
        weights = F.dropout(weights, self.dropout, training=self.training)
        result = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(result)


class Mistral3Block(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        cfg = config.text_config
        self.input_layernorm = Mistral3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Mistral3Attention(config)
        self.post_attention_layernorm = Mistral3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Mistral3MLP(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


class PixtralAttention(nn.Module):
    """Bidirectional vision attention; packed images are isolated by a block mask."""

    def __init__(self, config: Mistral3Config):
        super().__init__()
        cfg = config.vision_config
        self.num_heads, self.head_dim = cfg.num_attention_heads, cfg.head_dim

        if self.num_heads <= 0 or cfg.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("Vision hidden_size must equal num_attention_heads * head_dim")
        if self.head_dim <= 0 or self.head_dim % 4:
            raise ValueError("Vision head_dim must be a positive multiple of four for 2D RoPE")

        self.dropout = cfg.attention_dropout
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)


    def forward(self, x, angles, allowed):
        batch, length, _ = x.shape
        q, k, v = [
            proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]

        q, k = _apply_rope(q, angles), _apply_rope(k, angles)
        scores = (q @ k.transpose(-2, -1)).float() * self.head_dim ** -0.5

        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1).to(v.dtype)
        weights = F.dropout(weights, self.dropout, training=self.training)
        return self.o_proj((weights @ v).transpose(1, 2).reshape(batch, length, -1))


class PixtralBlock(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        dim = config.vision_config.hidden_size
        self.attention_norm = Mistral3RMSNorm(dim, 1e-5)
        self.attention = PixtralAttention(config)
        self.ffn_norm = Mistral3RMSNorm(dim, 1e-5)
        self.feed_forward = Mistral3MLP(config.vision_config)

    def forward(self, x, angles, allowed):
        x = x + self.attention(self.attention_norm(x), angles, allowed)
        return x + self.feed_forward(self.ffn_norm(x))


class Mistral3VisionEncoder(nn.Module):
    """Pixtral patch encoder for resized, normalized, optionally padded images."""

    def __init__(self, config: Mistral3Config):
        super().__init__()
        cfg = config.vision_config
        self.config = cfg
        self.feature_layer = config.vision_feature_layer
        if not -cfg.num_hidden_layers - 1 <= self.feature_layer <= cfg.num_hidden_layers:
            raise ValueError("vision_feature_layer is outside the vision hidden-state range")
        self.patch_conv = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, cfg.patch_size, cfg.patch_size, bias=False,
        )
        self.ln_pre = Mistral3RMSNorm(cfg.hidden_size, 1e-5)
        self.layers = nn.ModuleList(PixtralBlock(config) for _ in range(cfg.num_hidden_layers))
        self.register_buffer(
            "inv_freq", cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor | None = None) -> torch.Tensor:
        """Return packed image features [total patches, vision hidden size]."""
        cfg = self.config
        if pixel_values.ndim != 4 or pixel_values.size(1) != cfg.num_channels or pixel_values.size(0) == 0:
            raise ValueError("pixel_values must have shape [images, num_channels, height, width]")
        if image_sizes is None:
            image_sizes = torch.tensor([pixel_values.shape[-2:]] * pixel_values.size(0))
        if image_sizes.shape != (pixel_values.size(0), 2):
            raise ValueError("image_sizes must have shape [images, 2] in (height, width) order")
        sizes = image_sizes.tolist()
        for height, width in sizes:
            if (int(height) != height or int(width) != width or min(height, width) < cfg.patch_size
                    or max(height, width) > cfg.image_size
                    or height > pixel_values.size(-2) or width > pixel_values.size(-1)
                    or height % cfg.patch_size or width % cfg.patch_size):
                raise ValueError("Image sizes must be patch multiples within image_size and the padded pixel grid")

        patches = self.patch_conv(pixel_values)
        features, angles, image_ids = [], [], []
        for index, (height, width) in enumerate(sizes):
            h, w = int(height) // cfg.patch_size, int(width) // cfg.patch_size
            features.append(patches[index, :, :h, :w].flatten(1).T)
            rows, cols = torch.meshgrid(
                torch.arange(h, device=patches.device), torch.arange(w, device=patches.device), indexing="ij",
            )
            # Pixtral assigns alternating frequencies to the row and column axes.
            angles.append(
                torch.cat(
                    (rows.flatten()[:, None] * self.inv_freq.float()[::2], cols.flatten()[:, None] * self.inv_freq.float()[1::2]),
                    dim=-1
                )
            )
            image_ids.append(torch.full((h * w,), index, device=patches.device))
        x = self.ln_pre(torch.cat(features).unsqueeze(0))
        angles = torch.cat(angles).unsqueeze(0)
        image_ids = torch.cat(image_ids)
        allowed = (image_ids[:, None] == image_ids[None, :])[None, None]
        selected_index = self.feature_layer % (len(self.layers) + 1)
        selected = x if selected_index == 0 else None
        for index, layer in enumerate(self.layers, start=1):
            x = layer(x, angles, allowed)
            if index == selected_index:
                selected = x
        return selected.squeeze(0)


class Mistral3PatchMerger(nn.Module):
    """Learned spatial compression, with channel-first unfold ordering."""

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.patch_size = config.vision_config.patch_size
        self.merge_size = config.spatial_merge_size
        if self.merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        dim = config.vision_config.hidden_size
        self.merging_layer = nn.Linear(dim * self.merge_size ** 2, dim, bias=False)


    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        grids = [(int(h) // self.patch_size, int(w) // self.patch_size) for h, w in image_sizes.tolist()]
        counts = [h * w for h, w in grids]
        if image_features.ndim != 2 or image_features.size(0) != sum(counts):
            raise ValueError("Vision features do not match the image patch counts")
        if any(min(h, w) < self.merge_size for h, w in grids):
            raise ValueError("Each image must contain at least one complete merge block")
        merged = []
        for features, (h, w) in zip(image_features.split(counts), grids):
            grid = features.T.reshape(1, -1, h, w)
            # Like the reference implementation, incomplete edge blocks are dropped.
            merged.append(F.unfold(grid, self.merge_size, stride=self.merge_size).squeeze(0).T)
        return self.merging_layer(torch.cat(merged))


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        if config.projector_hidden_act not in ("silu", "gelu"):
            raise ValueError("projector_hidden_act must be 'silu' or 'gelu'")
        self.hidden_act = config.projector_hidden_act
        self.norm = Mistral3RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias
        )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias
        )

    def forward(self, image_features, image_sizes):
        x = self.patch_merger(self.norm(image_features), image_sizes)
        return self.linear_2(_activation(self.hidden_act, self.linear_1(x)))


class Mistral3OutputLayer(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        cfg = config.text_config
        self.rms_norm = Mistral3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.linear = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, x):
        return self.linear(self.rms_norm(x))


class Mistral3Model(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config
        cfg = config.text_config

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Mistral3Block(config) for _ in range(cfg.num_hidden_layers))

        # Mistral uses an independent LM head and no embedding scale.
        self.output_layer = Mistral3OutputLayer(config)
        self.vision_encoder = Mistral3VisionEncoder(config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.apply(lambda module: self._init_weights(module, cfg.initializer_range))
        self.vision_encoder.apply(
            lambda module: self._init_weights(module, config.vision_config.initializer_range)
        )


    @staticmethod
    def _init_weights(module, std):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return [batch, sequence, vocab] logits.

        Images are ordered by their placeholders in the flattened text batch.
        Each image requires (patch_rows // merge_size) * (patch_cols // merge_size)
        placeholders. image_sizes describes the unpadded, preprocessed pixel grid.
        """
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        if input_ids.size(1) > self.config.text_config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        image_mask = input_ids == self.config.image_token_index
        x = self.token_embedding(input_ids)
        if pixel_values is not None:
            features = self.vision_encoder(pixel_values, image_sizes)
            if image_sizes is None:
                image_sizes = torch.tensor([pixel_values.shape[-2:]] * pixel_values.size(0))
            features = self.multi_modal_projector(features, image_sizes)
            if int(image_mask.sum()) != features.size(0):
                raise ValueError(f"Expected {features.size(0)} image placeholders, found {int(image_mask.sum())}")
            x = x.clone()
            x[image_mask] = features.to(device=x.device, dtype=x.dtype)
        elif image_mask.any():
            raise ValueError("Image placeholder tokens require pixel_values")
        elif image_sizes is not None:
            raise ValueError("image_sizes requires pixel_values")

        # Pass the input through each transformer layer.
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)

        return self.output_layer(x)
