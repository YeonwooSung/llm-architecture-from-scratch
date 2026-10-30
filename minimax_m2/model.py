import torch
from torch import nn
from torch.nn import functional as F

from config import MiniMaxM2Config


class MiniMaxM2RMSNorm(nn.Module):
    """Accumulate normalization statistics in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RoPE(nn.Module):
    """Split-half rotation of only the leading rotary_dim features."""

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.dim = config.rotary_dim
        if not 0 < self.dim <= config.head_dim or self.dim % 2:
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
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


class MiniMaxM2Attention(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
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

        # M2 query width is 48 * 128 = 6144, versus hidden_size 3072.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.qk_norm_type = config.qk_norm_type
        if config.use_qk_norm and self.qk_norm_type not in ("per_layer", "per_head"):
            raise NotImplementedError("qk_norm_type must be 'per_layer' or 'per_head'")

        q_width = self.num_heads * self.head_dim if self.qk_norm_type == "per_layer" else self.head_dim
        k_width = self.num_kv_heads * self.head_dim if self.qk_norm_type == "per_layer" else self.head_dim

        self.q_norm = MiniMaxM2RMSNorm(q_width, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = MiniMaxM2RMSNorm(k_width, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

        if config.sliding_window is not None:
            raise NotImplementedError("Only full causal attention (sliding_window=None) is supported")


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k = self.q_proj(x), self.k_proj(x)
        # M2 normalizes across all projected heads in per_layer mode.
        if self.qk_norm_type == "per_layer":
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm_type != "per_layer":
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, position_ids), self.rope(k, position_ids)
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


class MiniMaxM2MLP(nn.Module):
    """Bias-free SwiGLU expert: 3072 -> 1536 -> 3072 by default."""

    def __init__(self, config: MiniMaxM2Config, intermediate_size: int | None = None):
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


class MiniMaxM2SparseMoeBlock(nn.Module):
    """Sigmoid top-k routing without shared experts.

    Correction bias affects selection only. Uncorrected selected scores are
    normalized to sum to one, then used to weight expert outputs.
    """

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        if not 1 <= config.num_experts_per_tok <= config.num_local_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")
        if config.scoring_func != "sigmoid":
            raise NotImplementedError("Only sigmoid routing is supported")
        if config.shared_intermediate_size != 0:
            raise NotImplementedError("M2 uses no shared experts (shared_intermediate_size=0)")
        if not 0 <= config.router_jitter_noise < 1:
            raise ValueError("router_jitter_noise must be in [0, 1)")

        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.use_routing_bias = config.use_routing_bias
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

        # register buffer for expert score correction bias
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

        self.experts = nn.ModuleList(
            MiniMaxM2MLP(config) for _ in range(config.num_local_experts)
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.size(-1))
        if self.training and self.jitter_noise:
            tokens = tokens * torch.empty_like(tokens).uniform_(
                1 - self.jitter_noise, 1 + self.jitter_noise,
            )
        # Preserve routing precision under explicit dtype conversion and AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(tokens.float(), self.gate.weight.float())
            scores = router_logits.sigmoid()
            with torch.no_grad():
                choices = scores
                if self.use_routing_bias:
                    choices = choices + self.e_score_correction_bias.float()
                selected = choices.topk(self.top_k, dim=-1).indices
            # Normalize in log space to also handle very negative logits.
            weights = F.logsigmoid(router_logits).gather(1, selected).softmax(dim=-1)

        output = torch.zeros_like(tokens, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]).float() * weights[token_ids, slots, None]
            output = output.index_add(0, token_ids, contribution)
        return output.to(x.dtype).reshape_as(x), router_logits


class MiniMaxM2Block(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.input_layernorm = MiniMaxM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = MiniMaxM2Attention(config)
        self.post_attention_layernorm = MiniMaxM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MiniMaxM2SparseMoeBlock(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        result, router_logits = self.mlp(self.post_attention_layernorm(x))
        return x + result, router_logits



class MiniMaxM2Model(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if len(config.attn_type_list) < config.num_hidden_layers or any(
            kind != 1 for kind in config.attn_type_list[:config.num_hidden_layers]
        ):
            raise NotImplementedError("Every decoder layer must use full attention (attn_type=1)")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MiniMaxM2Block(config) for _ in range(config.num_hidden_layers))
        self.norm = MiniMaxM2RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        output_router_logits: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return [batch, sequence, vocab] logits for [batch, sequence] IDs.

        attention_mask: [batch, sequence], 1/True for valid keys.
        position_ids: [1 or batch, sequence], defaulting to positions from zero.
        With output_router_logits, return (logits, tuple of per-layer router
        logits), each shaped [batch * sequence, num_local_experts]. Padding
        masks affect attention only; router logits include padded tokens.
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        if x.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        collect_router = self.config.output_router_logits if output_router_logits is None else output_router_logits
        x = self.token_embedding(x)
        router_outputs = []
        for layer in self.layers:
            x, router_logits = layer(x, attention_mask, position_ids)
            if collect_router:
                router_outputs.append(router_logits)
        logits = self.lm_head(self.norm(x))
        return (logits, tuple(router_outputs)) if collect_router else logits


