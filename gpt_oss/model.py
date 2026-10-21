import math

import torch
from torch import nn
from torch.nn import functional as F

from config import GptOssConfig


class GptOssRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RoPE(nn.Module):
    """Split-half rotary embeddings with YaRN frequency and attention scaling."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        if config.head_dim <= 0 or config.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if config.rope_theta <= 1 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta must exceed 1 and max_position_embeddings must be positive")

        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings
        scaling = config.rope_scaling or {}
        rope_type = scaling.get("rope_type", "default")
        if rope_type not in ("default", "yarn"):
            raise NotImplementedError("Only default and yarn RoPE are supported")

        self.factor = scaling.get("factor", 1.0) if rope_type == "yarn" else 1.0
        self.original_length = scaling.get("original_max_position_embeddings", config.initial_context_length)
        self.beta_fast = scaling.get("beta_fast", 32.0)
        self.beta_slow = scaling.get("beta_slow", 1.0)
        self.truncate = scaling.get("truncate", False)
        if self.factor < 1 or self.original_length <= 0 or not 0 < self.beta_slow < self.beta_fast:
            raise ValueError("Invalid YaRN scaling factor, context length, or beta values")
        self.attention_scaling = scaling.get("attention_factor", 1 + 0.1 * math.log(self.factor))


    def _inverse_frequencies(self, device: torch.device) -> torch.Tensor:
        # Derive frequencies in float32 even after model.to(bfloat16/float16).
        indices = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        frequencies = self.theta ** (indices / self.head_dim)
        extrapolated = 1.0 / frequencies
        if self.factor == 1:
            return extrapolated
        low = self.head_dim * math.log(self.original_length / (2 * math.pi * self.beta_fast)) / (2 * math.log(self.theta))
        high = self.head_dim * math.log(self.original_length / (2 * math.pi * self.beta_slow)) / (2 * math.log(self.theta))
        if self.truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, self.head_dim - 1)
        ramp = ((indices / 2 - low) / max(high - low, 0.001)).clamp(0, 1)
        return (1.0 / (self.factor * frequencies)) * ramp + extrapolated * (1 - ramp)


    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("Sequence exceeds max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device)[None]
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if (position_ids < 0).any() or (position_ids >= self.max_seq_len).any():
            raise ValueError("position_ids must be within max_position_embeddings")
        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * self._inverse_frequencies(x.device)
        cos = (angles.cos() * self.attention_scaling).to(x.dtype)[:, None]
        sin = (angles.sin() * self.attention_scaling).to(x.dtype)[:, None]
        left, right = x.chunk(2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if not 0 <= config.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        if not 0 <= layer_idx < len(config.layer_types):
            raise ValueError("layer_types must specify each decoder layer")

        # Uses sliding attention and full attention
        # (1st layer = sliding, 2nd layer onwards = full, 3rd layer = slding, 4th layer = full, ...)
        layer_type = config.layer_types[layer_idx]
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError("Unsupported attention layer type")
        self.window = config.sliding_window if layer_type == "sliding_attention" else None
        if self.window is not None and self.window <= 0:
            raise ValueError("sliding_window must be positive")

        self.dropout = config.attention_dropout
        self.rope = RoPE(config)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # A learned logit per head that absorbs probability mass but contributes no value.
        # Basically, attention allocates probability mass to the actual keys (sum of probs should be 1).
        # But, it is clear that the query might not always have a perfect match among the keys,
        # so the sink absorbs the leftover probability mass.
        # The main purpose of the sink is to reduce the impact of queries that do not have a good match among the keys.
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, position_ids), self.rope(k, position_ids)
        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)

        indices = torch.arange(length, device=x.device)
        distance = indices[:, None] - indices[None, :]
        allowed = (distance >= 0)[None, None]
        if self.window is not None:
            allowed = allowed & (distance < self.window)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]

        scores = (q.float() @ k.float().transpose(-2, -1)) * self.head_dim ** -0.5
        scores = scores.masked_fill(~allowed, -torch.inf)

        # A learned logit per head absorbs probability mass but contributes no value.
        # Keeping it unmasked also makes fully padded rows well-defined.
        sink = self.sinks.float()[None, :, None, None].expand(batch, -1, length, 1)
        weights = F.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        weights = F.dropout(weights, p=self.dropout, training=self.training).to(v.dtype)
        result = (weights @ v).transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)

        return self.o_proj(result)


class GptOssExpert(nn.Module):
    """Biased expert with interleaved gate/up projection and clamped SwiGLU."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only GPT OSS's modified SwiGLU is supported")
        if config.intermediate_size <= 0 or config.swiglu_limit <= 0:
            raise ValueError("intermediate_size and swiglu_limit must be positive")

        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.limit = config.swiglu_limit


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate = gate_up[..., ::2].clamp(max=self.limit)
        up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * torch.sigmoid(1.702 * gate) * (up + 1))


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        # Use the HF-style field, as in the other MoE models in this repository.
        self.top_k = config.num_experts_per_tok
        if not 1 <= self.top_k <= config.num_local_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = nn.ModuleList(GptOssExpert(config) for _ in range(config.num_local_experts))


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.size(-1))

        # Compute the router logits and select the top-k experts for each token.
        # GPT OSS allocates each token to the top-4 experts (based on the config)
        router_logits = self.router(tokens)
        selected_logits, selected = router_logits.topk(self.top_k, dim=-1)

        weights = F.softmax(selected_logits, dim=-1, dtype=torch.float32).to(tokens.dtype)
        output = torch.zeros_like(tokens)

        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]) * weights[token_ids, slots, None]
            output = output.index_add(0, token_ids, contribution.to(output.dtype))
        return output.reshape_as(x), router_logits


class GptOssBlock(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GptOssMLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        result, router_logits = self.mlp(self.post_attention_layernorm(x))
        return x + result, router_logits


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        if config.num_hidden_layers <= 0 or len(config.layer_types) < config.num_hidden_layers:
            raise ValueError("layer_types must specify each of the positive num_hidden_layers")
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(GptOssBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight


    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Return [batch, sequence, vocab] logits for integer token IDs.

        attention_mask: [batch, sequence], True/1 for valid keys.
        position_ids: [1 or batch, sequence]; defaults to positions starting at 0.
        If output_router_logits is enabled, also return one flattened
        [batch * sequence, num_local_experts] tensor per layer. Padding masks
        affect attention only. No auxiliary loss is computed here.
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
