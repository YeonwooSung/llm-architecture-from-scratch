import torch
from torch import nn
from torch.nn import functional as F

from config import GLM4_5Config


class GLM4_5RMSNorm(nn.Module):
    """Accumulate normalization statistics in float32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RoPE(nn.Module):
    """Split-half rotation of only the leading partial_rotary_factor features."""

    def __init__(self, config: GLM4_5Config):
        super().__init__()
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        if not 0 < config.partial_rotary_factor <= 1 or self.dim <= 0 or self.dim % 2:
            raise ValueError("partial_rotary_factor must produce a positive even rotary dimension <= head_dim")
        if config.rope_scaling is not None:
            raise NotImplementedError("Only default RoPE (rope_scaling=None) is supported")
        if config.rope_theta <= 0 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta and max_position_embeddings must be positive")
        self.theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings

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

        # Generate only requested positions, preserving float32 frequencies even
        # after model.bfloat16(), rather than caching a large table per layer.
        with torch.autocast(device_type=x.device.type, enabled=False):
            frequencies = self.theta ** (-torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
            angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * frequencies
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        rotary, unchanged = x[..., :self.dim], x[..., self.dim:]
        left, right = rotary.chunk(2, dim=-1)
        rotated = rotary * cos + torch.cat((-right, left), dim=-1) * sin
        return torch.cat((rotated, unchanged), dim=-1)


class GLM4_5Attention(nn.Module):
    def __init__(self, config: GLM4_5Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if self.head_dim <= 0 or not 0 <= config.attention_dropout <= 1:
            raise ValueError("head_dim must be positive and attention_dropout must be between 0 and 1")
        self.dropout = config.attention_dropout
        self.rope = RoPE(config)
        # The query width (96 * 128) differs from hidden_size (5120).
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = GLM4_5RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = GLM4_5RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(self.q_norm(q), position_ids)
        k = self.rope(self.k_norm(k), position_ids)
        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)

        allowed = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None] & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(result)


class GLM4_5MLP(nn.Module):
    """Bias-free SwiGLU used by dense layers and all experts."""

    def __init__(self, config: GLM4_5Config, intermediate_size: int | None = None):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only hidden_act='silu' is supported")
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        if width <= 0:
            raise ValueError("intermediate_size must be positive")
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GLM4_5Router(nn.Module):
    """Select with corrected sigmoid scores; weight with uncorrected scores."""

    def __init__(self, config: GLM4_5Config):
        super().__init__()
        if config.n_routed_experts <= 0 or config.n_group <= 0 or config.n_routed_experts % config.n_group:
            raise ValueError("n_group must divide the positive n_routed_experts")
        if not 1 <= config.topk_group <= config.n_group:
            raise ValueError("topk_group must be between 1 and n_group")
        self.experts_per_group = config.n_routed_experts // config.n_group
        if config.n_group > 1 and self.experts_per_group < 2:
            raise ValueError("Grouped routing requires at least two experts per group")
        if not 1 <= config.num_experts_per_tok <= config.topk_group * self.experts_per_group:
            raise ValueError("num_experts_per_tok exceeds the available routed experts")
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=config.initializer_range)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.size(-1))
        # Routing remains float32 under both explicit dtype conversion and AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(tokens.float(), self.weight.float()).sigmoid()
            with torch.no_grad():
                choices = scores + self.e_score_correction_bias.float()
                if self.config.n_group > 1:
                    grouped = choices.view(-1, self.config.n_group, self.experts_per_group)
                    group_scores = grouped.topk(2, dim=-1).values.sum(-1)
                    groups = group_scores.topk(self.config.topk_group, dim=-1).indices
                    allowed = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
                    choices = grouped.masked_fill(~allowed[..., None], -torch.inf).flatten(1)
                indices = choices.topk(self.config.num_experts_per_tok, dim=-1).indices
            weights = scores.gather(1, indices)
            if self.config.norm_topk_prob:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * self.config.routed_scaling_factor
        return weights, indices


class GLM4_5MoE(nn.Module):
    def __init__(self, config: GLM4_5Config):
        super().__init__()
        if config.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be nonnegative")
        self.gate = GLM4_5Router(config)
        self.experts = nn.ModuleList(
            GLM4_5MLP(config, config.moe_intermediate_size) for _ in range(config.n_routed_experts)
        )
        self.shared_experts = (
            GLM4_5MLP(config, config.n_shared_experts * config.moe_intermediate_size)
            if config.n_shared_experts else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, x.size(-1))
        weights, indices = self.gate(tokens)
        output = torch.zeros_like(tokens, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]).float() * weights[token_ids, slots, None]
            output = output.index_add(0, token_ids, contribution)
        output = output.to(x.dtype)
        if self.shared_experts is not None:
            output = output + self.shared_experts(tokens)
        return output.reshape_as(x)


class GLM4_5Block(nn.Module):
    def __init__(self, config: GLM4_5Config, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = GLM4_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GLM4_5Attention(config)
        self.post_attention_layernorm = GLM4_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GLM4_5MLP(config) if layer_idx < config.first_k_dense_replace else GLM4_5MoE(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


class GLM4_5Model(nn.Module):
    def __init__(self, config: GLM4_5Config):
        super().__init__()
        if config.num_hidden_layers <= 0 or config.first_k_dense_replace < 0:
            raise ValueError("num_hidden_layers must be positive and first_k_dense_replace nonnegative")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(GLM4_5Block(config, i) for i in range(config.num_hidden_layers))
        self.norm = GLM4_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, sequence, vocab] logits for [batch, sequence] token IDs.

        attention_mask is [batch, sequence], with 1/True for valid keys.
        position_ids is [1 or batch, sequence]; defaults to positions from zero.
        Padding affects attention only. No inference cache or MTP is used.
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        if x.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        return self.lm_head(self.norm(x))
