import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Gemma3Config


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + self.weight.float())).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config, local: bool):
        super().__init__()

        cfg = config.text_config
        self.local = local
        self.window = cfg.sliding_window
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError("Query heads must divide by KV heads and head_dim must be even")

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        eps = getattr(cfg, "rms_norm_eps", 1e-6)
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps)
        self.scale = cfg.query_pre_attn_scalar ** -0.5
        theta = 10_000.0 if local else 1_000_000.0

        # register the inverse frequency buffer for rotary positional embeddings
        self.register_buffer("inv_freq", theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim), persistent=False)
        self.rope_factor = 1.0 if local else cfg.rope_scaling.get("factor", 1.0)


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(length, device=x.device, dtype=torch.float32) / self.rope_factor
        angles = torch.outer(positions, self.inv_freq.to(x.device))
        cos = torch.cat((angles, angles), -1).cos().to(q.dtype)[None, None]
        sin = torch.cat((angles, angles), -1).sin().to(q.dtype)[None, None]

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Expand each KV head to the query heads in its group.
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        row = torch.arange(length, device=x.device)
        distance = row[:, None] - row[None, :]
        allowed = distance >= 0
        if image_mask is not None:
            if image_mask.shape != (batch, length):
                raise ValueError("image_mask must have shape [batch, sequence]")
            # Tokens within the same contiguous image block can see each other.
            starts = image_mask & ~F.pad(image_mask[:, :-1], (1, 0), value=False)
            blocks = starts.long().cumsum(1)
            same_image = (blocks[:, :, None] == blocks[:, None, :]) & image_mask[:, :, None] & image_mask[:, None, :]
            allowed = allowed[None] | same_image

        if self.local:
            allowed = allowed & (distance.abs() < self.window if image_mask is not None else distance < self.window)

        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask[:, None, :].bool()

        scores = scores.float().masked_fill(~allowed[:, None] if allowed.ndim == 3 else ~allowed, torch.finfo(torch.float32).min)
        weights = F.softmax(scores, dim=-1).to(v.dtype)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, -1)

        return self.o_proj(result)


class Gemma3Block(nn.Module):
    def __init__(self, config: Gemma3Config, layer_index: int = 0):
        super().__init__()
        cfg = config.text_config
        eps = getattr(cfg, "rms_norm_eps", 1e-6)
        self.local = (layer_index + 1) % 6 != 0
        self.input_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps)
        self.self_attn = Gemma3Attention(config, self.local)
        self.post_attention_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.post_feedforward_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps)


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), attention_mask, image_mask))
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        return x + self.post_feedforward_layernorm(h)


class Gemma3OutputLayer(nn.Module):
    def __init__(self, config: Gemma3Config, embedding: nn.Embedding | None = None):
        super().__init__()
        cfg = config.text_config
        self.rms_norm = Gemma3RMSNorm(cfg.hidden_size, getattr(cfg, "rms_norm_eps", 1e-6))
        self.linear = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if embedding is not None:
            self.linear.weight = embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.rms_norm(x))


class Gemma3VisionEncoder(nn.Module):
    """SigLIP-style patch encoder; expects already resized and normalized pixels."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        cfg = config.vision_config
        patches = cfg.image_size // cfg.patch_size
        self.image_size = cfg.image_size

        # patch embedding
        self.patch_embedding = nn.Conv2d(3, cfg.hidden_size, cfg.patch_size, cfg.patch_size)
        # position embedding for each patch
        self.position_embedding = nn.Embedding(patches * patches, cfg.hidden_size)

        # transformer encoder layers for processing the sequence of patch embeddings
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(cfg.num_hidden_layers)
        ])

        # final layer normalization after the transformer encoder layers
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=getattr(cfg, "layer_norm_eps", 1e-6))


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"pixel_values must have shape [images, 3, {self.image_size}, {self.image_size}]")

        # apply patch embedding and flatten the spatial dimensions
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)

        # add position embeddings to the patch embeddings
        x = x + self.position_embedding(torch.arange(x.size(1), device=x.device))

        # pass through the transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # apply final layer normalization and return the output
        return self.post_layernorm(x)


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        cfg = config.vision_config

        self.patches_per_side = cfg.image_size // cfg.patch_size
        self.tokens_per_side = math.isqrt(config.mm_tokens_per_image)
        if self.tokens_per_side ** 2 != config.mm_tokens_per_image or self.patches_per_side % self.tokens_per_side:
            raise ValueError("Image token grid must evenly divide the patch grid")

        kernel = self.patches_per_side // self.tokens_per_side
        self.pool = nn.AvgPool2d(kernel, kernel)
        self.norm = Gemma3RMSNorm(cfg.hidden_size, getattr(cfg, "layer_norm_eps", 1e-6))
        self.projection = nn.Linear(cfg.hidden_size, config.text_config.hidden_size, bias=False)


    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        images, patches, channels = vision_outputs.shape
        if patches != self.patches_per_side ** 2:
            raise ValueError("Vision output has an unexpected patch count")

        # reshape and pool the vision outputs to match the multi-modal token grid
        x = vision_outputs.transpose(1, 2).reshape(images, channels, self.patches_per_side, self.patches_per_side)

        # apply average pooling to reduce the spatial resolution to match the token grid
        x = self.pool(x).flatten(2).transpose(1, 2)

        # apply RMS normalization and project to the text hidden size
        return self.projection(self.norm(x))


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        cfg = config.text_config

        # token embedding
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # gemma 3 blocks
        self.layers = nn.ModuleList(Gemma3Block(config, i) for i in range(cfg.num_hidden_layers))

        # output layer
        self.output_layer = Gemma3OutputLayer(config, self.token_embedding)

        # vision encoder and multi-modal projector
        self.vision_encoder = Gemma3VisionEncoder(config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        # token embedding scale buffer
        self.token_embedding_scale = nn.Buffer(torch.tensor(math.sqrt(cfg.hidden_size)))


    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        # create image mask and embed tokens
        image_mask = input_ids == self.config.image_token_index
        x = self.token_embedding(input_ids) * self.token_embedding_scale.to(input_ids.device)

        # process images if provided
        if pixel_values is not None:
            if pixel_values.ndim != 4:
                raise ValueError("pixel_values must have shape [images, 3, height, width]")

            expected = int(image_mask.sum().item())
            available = pixel_values.size(0) * self.config.mm_tokens_per_image
            if expected != available:
                raise ValueError(f"Expected {available} image placeholders, found {expected}")

            image_features = self.multi_modal_projector(self.vision_encoder(pixel_values))
            # replace image placeholder tokens with image features
            x = x.clone()
            x[image_mask] = image_features.reshape(-1, x.size(-1)).to(x.dtype)
        elif image_mask.any():
            # raise an error if image placeholders exist but no pixel values are provided
            raise ValueError("Image placeholder tokens require pixel_values")

        # pass the embeddings through the Gemma 3 blocks
        for layer in self.layers:
            x = layer(x, attention_mask, image_mask if pixel_values is not None else None)

        # project the final hidden states to the output vocabulary
        return self.output_layer(x)
