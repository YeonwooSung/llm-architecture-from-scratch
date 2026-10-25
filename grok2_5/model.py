import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Grok2_5Config


class Grok2_5RMSNorm(nn.Module):
    """Accumulate RMS statistics in float32, preserving the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def _softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(x / cap) if cap > 0 else x


class RoPE(nn.Module):
    def __init__(self, config: Grok2_5Config):
        super().__init__()

        if config.head_dim <= 0 or config.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if config.rope_type not in ("original", "yarn"):
            raise NotImplementedError("Only Grok original/YaRN rotary embeddings are supported")
        if config.rope_theta <= 1 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta must exceed 1 and max_position_embeddings must be positive")
        if config.scaling_factor < 1 or config.original_max_position_embeddings <= 0:
            raise ValueError("Invalid rotary scaling factor or original context length")
        if not 0 < config.beta_slow < config.beta_fast:
            raise ValueError("Expected 0 < beta_slow < beta_fast")
        if not 0 <= config.extrapolation_factor <= 1 or config.attn_factor <= 0:
            raise ValueError("Invalid rotary extrapolation or attention factor")

        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings
        self.factor = config.scaling_factor
        self.original_length = config.original_max_position_embeddings
        self.beta_fast = config.beta_fast
        self.beta_slow = config.beta_slow
        self.extrapolation = config.extrapolation_factor
        self.attn_factor = config.attn_factor

    def _inverse_frequencies(self, device: torch.device) -> torch.Tensor:
        # Recompute in float32 even when the model was cast to bfloat16.
        indices = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        frequencies = self.theta ** (-indices / self.head_dim)
        if self.factor == 1:
            return frequencies

        def correction_dim(rotations: float) -> float:
            return self.head_dim * math.log(self.original_length / (2 * math.pi * rotations)) / (2 * math.log(self.theta))

        low = max(math.floor(correction_dim(self.beta_fast)), 0)
        high = min(math.ceil(correction_dim(self.beta_slow)), self.head_dim - 1)
        ramp = ((indices / 2 - low) / max(high - low, 0.001)).clamp(0, 1)
        blend = (1 - ramp) * self.extrapolation
        return frequencies * blend + frequencies / self.factor * (1 - blend)

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
        magnitude = self.attn_factor * (1 + 0.1 * math.log(self.factor))
        cos = (angles.cos() * magnitude).to(x.dtype)[:, None]
        sin = (angles.sin() * magnitude).to(x.dtype)[:, None]
        left, right = x.chunk(2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Grok2_5Attention(nn.Module):
    """Causal grouped-query attention with softcapped attention logits."""

    def __init__(self, config: Grok2_5Config, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads")
        if config.global_attn_every_n <= 0:
            raise ValueError("global_attn_every_n must be positive")
        if config.sliding_window_size == 0 or config.sliding_window_size < -1:
            raise ValueError("sliding_window_size must be -1 or a positive integer")
        if config.attn_temperature_len != -1 and config.attn_temperature_len <= 1:
            raise ValueError("attn_temperature_len must be -1 or greater than one")
        self.window = None
        if config.sliding_window_size > 0 and (layer_idx + 1) % config.global_attn_every_n:
            self.window = config.sliding_window_size
        self.softcap = config.attn_logit_softcapping
        self.temperature_length = config.attn_temperature_len
        self.rope = RoPE(config)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

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
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.head_dim ** -0.5
        if self.temperature_length > 1:
            positions = torch.arange(length, device=x.device)[None] if position_ids is None else position_ids.to(x.device)
            temperature = ((positions.float() + 1).log() / math.log(self.temperature_length)).clamp_min(1)
            scores = scores * temperature[:, None, :, None]
        scores = _softcap(scores, self.softcap)

        # Apply masks AFTER the softcap so future keys retain zero probability.
        indices = torch.arange(length, device=x.device)
        distance = indices[:, None] - indices[None, :]
        allowed = (distance >= 0)[None, None]
        if self.window is not None:
            allowed = allowed & (distance < self.window)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        # Fully masked queries (e.g. left padding) contribute zero, without NaNs
        # in either the forward or backward pass.
        has_keys = allowed.any(dim=-1, keepdim=True)
        probabilities = F.softmax(scores.masked_fill(~has_keys, 0), dim=-1)
        probabilities = probabilities.masked_fill(~allowed, 0).to(v.dtype)
        output = torch.matmul(probabilities, v).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class Grok2_5MLP(nn.Module):
    """Bias-free SwiGLU, shared by the dense branch and routed experts."""

    def __init__(self, config: Grok2_5Config, intermediate_size: int | None = None):
        super().__init__()
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        if width <= 0:
            raise ValueError("intermediate_size must be positive")
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Grok2_5MoE(nn.Module):
    """Top-k token routing; only selected experts are evaluated."""

    def __init__(self, config: Grok2_5Config):
        super().__init__()
        if not 1 <= config.num_experts_per_tok <= config.num_local_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_local_experts")
        self.top_k = config.num_experts_per_tok
        self.softcap = config.router_logit_softcapping
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            Grok2_5MLP(config, config.moe_intermediate_size) for _ in range(config.num_local_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.size(-1))
        router_logits = _softcap(self.gate(tokens).float(), self.softcap)
        # Normalize over the selected experts so their weights sum to one.
        selected_logits, selected = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(selected_logits, dim=-1).to(tokens.dtype)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]) * weights[token_ids, slots, None]
            output = output.index_add(0, token_ids, contribution.to(output.dtype))
        return output.reshape_as(x), router_logits


class Grok2_5Block(nn.Module):
    """Follow the supplied diagram: pre-norm GQA, then parallel dense + MoE."""

    def __init__(self, config: Grok2_5Config, layer_idx: int = 0):
        super().__init__()
        self.rms_norm1 = Grok2_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Grok2_5Attention(config, layer_idx)
        self.rms_norm2 = Grok2_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = Grok2_5MoE(config)
        self.mlp = Grok2_5MLP(config) if config.residual_moe else None

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.rms_norm1(x), attention_mask, position_ids)
        normalized = self.rms_norm2(x)
        output, router_logits = self.moe(normalized)
        if self.mlp is not None:
            output = output + self.mlp(normalized)
        return x + output, router_logits


class Grok2_5Model(nn.Module):
    def __init__(self, config: Grok2_5Config):
        super().__init__()
        if config.vocab_size <= 0 or config.hidden_size <= 0 or config.num_hidden_layers <= 0:
            raise ValueError("Vocabulary size, hidden size, and layer count must be positive")
        if config.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if min(config.attn_logit_softcapping, config.router_logit_softcapping, config.final_logit_softcapping) < 0:
            raise ValueError("Softcapping values must be nonnegative (zero disables softcapping)")
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Grok2_5Block(config, i) for i in range(config.num_hidden_layers))
        self.final_rmsnorm = Grok2_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None, output_router_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return [batch, sequence, vocab] logits for [batch, sequence] token IDs.

        attention_mask is [batch, sequence], with 1/True for valid keys.
        position_ids is [1 or batch, sequence], defaulting to zero-based indices.
        Optional router logits contain [batch * sequence, num_local_experts]
        per block, including padded tokens; auxiliary losses are left to callers.
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        if x.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        x = self.token_embedding(x) * self.config.embedding_multiplier_scale
        router_outputs = []
        for block in self.blocks:
            x, router_logits = block(x, attention_mask, position_ids)
            if output_router_logits:
                router_outputs.append(router_logits)

        logits = self.output_layer(self.final_rmsnorm(x)) * self.config.output_multiplier_scale
        logits = _softcap(logits, self.config.final_logit_softcapping)

        return (logits, tuple(router_outputs)) if output_router_logits else logits
