import math

import torch
from torch import nn
from torch.nn import functional as F

from config import DeepSeekV3_2Config


class DeepSeekV3_2RMSNorm(nn.Module):
    """Accumulate the normalization statistics in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normalized).to(x.dtype)


class RoPE(nn.Module):
    """Adjacent-pair rotary embeddings on the positional part of MLA Q/K."""

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.theta = config.rope_theta
        if self.dim <= 0 or self.dim % 2:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if self.theta <= 1 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta must exceed 1 and max_position_embeddings must be positive")

        self.factor = 1.0
        self.low, self.high = 0, 1
        self.mscale = 1.0
        self.attention_scale = 1.0
        scaling = config.rope_scaling
        if scaling is not None:
            if scaling.get("rope_type", scaling.get("type")) != "yarn":
                raise NotImplementedError("Only default RoPE and YaRN scaling are supported")
            self.factor = scaling["factor"]
            original_length = scaling["original_max_position_embeddings"]
            beta_fast = scaling.get("beta_fast", 32)
            beta_slow = scaling.get("beta_slow", 1)
            if self.factor <= 0 or original_length <= 0 or not 0 < beta_slow <= beta_fast:
                raise ValueError("YaRN requires positive factor/length and beta_fast >= beta_slow > 0")
            if config.max_position_embeddings <= original_length:
                self.factor = 1.0
                return

            def correction_dim(rotations: float) -> float:
                return self.dim * math.log(original_length / (rotations * 2 * math.pi)) / (2 * math.log(self.theta))

            self.low = max(math.floor(correction_dim(beta_fast)), 0)
            self.high = min(math.ceil(correction_dim(beta_slow)), self.dim - 1)
            if self.low == self.high:
                self.high += 0.001

            def magnitude(value: float) -> float:
                return 1.0 if self.factor <= 1 else 1.0 + 0.1 * value * math.log(self.factor)

            all_dim_scale = magnitude(scaling.get("mscale_all_dim", 0))
            self.mscale = magnitude(scaling.get("mscale", 1)) / all_dim_scale
            self.attention_scale = all_dim_scale ** 2

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, interleaved: bool = True,
    ) -> torch.Tensor:
        """Rotate [batch, sequence, heads, rope_dim] using [1 or batch, sequence] positions."""
        # Recompute small frequency vectors in float32 even after model.bfloat16().
        # No position table is replicated across the 61 decoder layers.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dims = torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            inv_freq = self.theta ** (-dims / self.dim)
            if self.factor != 1:
                ramp = ((dims / 2 - self.low) / (self.high - self.low)).clamp(0, 1)
                inv_freq = inv_freq * (1 - ramp) + inv_freq / self.factor * ramp
            angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
            cos = angles.cos().unsqueeze(2) * self.mscale
            sin = angles.sin().unsqueeze(2) * self.mscale
            if not interleaved:
                # Indexer checkpoints pair the first and second feature halves.
                x = torch.stack(x.chunk(2, dim=-1), dim=-1).flatten(-2)
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            rotated = torch.stack((
                pairs[..., 0] * cos - pairs[..., 1] * sin,
                pairs[..., 0] * sin + pairs[..., 1] * cos,
            ), dim=-1)
        if not interleaved:
            return torch.cat((rotated[..., 0], rotated[..., 1]), dim=-1).to(x.dtype)
        return rotated.flatten(-2).to(x.dtype)


class LightningIndexer(nn.Module):
    """Rank causal keys using a weighted sum of rectified per-head dot products.

    Selection is discrete, so the indexer needs its own distillation objective
    to train. Scores are computed head by head to avoid a [B, H, S, S] tensor;
    the reference indexer still performs quadratic work in sequence length.
    """

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        if config.index_n_heads <= 0 or config.index_topk <= 0:
            raise ValueError("index_n_heads and index_topk must be positive")
        if config.index_head_dim < config.qk_rope_head_dim:
            raise ValueError("index_head_dim must be at least qk_rope_head_dim")
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank or config.hidden_size,
            self.num_heads * self.head_dim, bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rope = RoPE(config)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.wq_b(qr).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).unsqueeze(2)
        # Indexer RoPE uses split-half pairs; MLA uses adjacent pairs.
        q_pe = self.rope(q[..., :self.rope_dim], position_ids, interleaved=False)
        k_pe = self.rope(k[..., :self.rope_dim], position_ids, interleaved=False)
        # The indexer uses unit-magnitude rotations, independent of MLA's YaRN
        # attention magnitude adjustment.
        q = torch.cat((q_pe / self.rope.mscale, q[..., self.rope_dim:]), -1)
        k = torch.cat((k_pe / self.rope.mscale, k[..., self.rope_dim:]), -1).squeeze(2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = F.linear(x.float(), self.weights_proj.weight.float())
            weights = weights * (self.num_heads * self.head_dim) ** -0.5
            scores = torch.zeros(batch, length, length, device=x.device, dtype=torch.float32)
            for head in range(self.num_heads):
                dots = q[:, :, head].float() @ k.float().transpose(-1, -2)
                scores.add_(dots.relu() * weights[:, :, head, None])
        allowed = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allowed = allowed.unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, :]
        scores.masked_fill_(~allowed, -torch.inf)
        # Early queries have fewer than topk valid keys. The attention layer
        # rechecks validity so arbitrary -inf entries never leak future tokens.
        return scores.topk(min(self.topk, length), dim=-1).indices


class MultiHeadLatentAttention(nn.Module):
    """Compress Q/KV, expand per-head content, and share the positional key."""

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        if min(self.num_heads, config.kv_lora_rank, config.qk_nope_head_dim, config.v_head_dim) <= 0:
            raise ValueError("MLA head counts, head dimensions, and kv_lora_rank must be positive")
        if config.q_lora_rank is not None and config.q_lora_rank < 0:
            raise ValueError("q_lora_rank must be nonnegative or None")
        if not 0 <= config.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")

        self.use_q_lora = bool(config.q_lora_rank)
        if self.use_q_lora:
            self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_norm = DeepSeekV3_2RMSNorm(config.q_lora_rank, config.rms_norm_eps)
            self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.wq = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=config.attention_bias)

        self.wkv_a = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_norm = DeepSeekV3_2RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.wkv_b = nn.Linear(
            config.kv_lora_rank, self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.rope = RoPE(config)
        self.softmax_scale = self.qk_head_dim ** -0.5 * self.rope.attention_scale
        self.indexer = LightningIndexer(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        if length > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if (position_ids < 0).any() or (position_ids >= self.config.max_position_embeddings).any():
            raise ValueError("position_ids must be within max_position_embeddings")

        qr = self.q_norm(self.wq_a(x)) if self.use_q_lora else x
        q = self.wq_b(qr) if self.use_q_lora else self.wq(x)
        q = q.reshape(batch, length, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split((self.config.qk_nope_head_dim, self.config.qk_rope_head_dim), dim=-1)
        kv, k_pe = self.wkv_a(x).split((self.config.kv_lora_rank, self.config.qk_rope_head_dim), dim=-1)
        kv = self.wkv_b(self.kv_norm(kv)).reshape(
            batch, length, self.num_heads, self.config.qk_nope_head_dim + self.config.v_head_dim,
        )
        k_nope, v = kv.split((self.config.qk_nope_head_dim, self.config.v_head_dim), dim=-1)
        q_pe = self.rope(q_pe, position_ids)
        k_pe = self.rope(k_pe.unsqueeze(2), position_ids)
        q = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2)
        k = torch.cat((k_nope, k_pe.expand(-1, -1, self.num_heads, -1)), dim=-1).transpose(1, 2)

        # One set of selected tokens is shared by all MLA heads. Gather only
        # those keys/values instead of constructing a dense MLA score matrix.
        indices = self.indexer(x, qr, position_ids, attention_mask)
        batch_ids = torch.arange(batch, device=x.device)[:, None, None]
        selected_k = k.transpose(1, 2)[batch_ids, indices]
        selected_v = v[batch_ids, indices]
        scores = torch.einsum("bhsd,bskhd->bshk", q.float(), selected_k.float())
        scores = scores * self.softmax_scale
        allowed = indices <= torch.arange(length, device=x.device)[None, :, None]
        if attention_mask is not None:
            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)
            allowed = allowed & valid_keys[batch_ids, indices]
        scores = scores.masked_fill(~allowed.unsqueeze(2), -torch.inf)
        # A fully padded prefix has no valid keys. Make its output zero without
        # generating NaNs in softmax or in the backward pass.
        has_keys = allowed.any(-1, keepdim=True).unsqueeze(2)
        scores = torch.where(has_keys, scores, 0.0)
        attention = F.softmax(scores, dim=-1).masked_fill(~allowed.unsqueeze(2), 0)
        attention = F.dropout(
            attention.to(v.dtype), p=self.config.attention_dropout, training=self.training,
        )
        output = torch.einsum("bshk,bskhd->bshd", attention, selected_v)
        return self.wo(output.reshape(batch, length, self.num_heads * self.config.v_head_dim))


class Expert(nn.Module):
    """Bias-free SwiGLU used by dense FFNs, routed experts, and shared experts."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEGate(nn.Module):
    """Use corrected scores for selection and original scores for mixture weights."""

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        self.config = config
        if config.topk_method != "noaux_tc" or config.scoring_func != "sigmoid":
            raise NotImplementedError("DeepSeek V3.2 routing requires noaux_tc and sigmoid scoring")
        if config.n_routed_experts <= 0 or config.n_group <= 0 or config.n_routed_experts % config.n_group:
            raise ValueError("n_group must be positive and divide n_routed_experts")
        if not 1 <= config.topk_group <= config.n_group:
            raise ValueError("topk_group must be between 1 and n_group")
        per_group = config.n_routed_experts // config.n_group
        if config.n_group > 1 and per_group < 2:
            raise ValueError("Grouped routing requires at least two experts per group")
        if not 1 <= config.num_experts_per_tok <= config.topk_group * per_group:
            raise ValueError("num_experts_per_tok exceeds the available experts")
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=config.initializer_range)
        # Updated externally for load balancing, rather than by backpropagation.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Casting only the logits would still compute the router in low precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(x.float(), self.weight.float()).sigmoid()
        choice_scores = scores + self.e_score_correction_bias.float()
        if self.config.n_group > 1:
            grouped = choice_scores.reshape(x.size(0), self.config.n_group, -1)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected = group_scores.topk(self.config.topk_group, dim=-1).indices
            allowed = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, selected, True)
            choice_scores = grouped.masked_fill(~allowed.unsqueeze(-1), -torch.inf).flatten(1)
        indices = choice_scores.topk(self.config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.config.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (weights * self.config.routed_scaling_factor).to(x.dtype), indices


class MoE(nn.Module):
    """Sum top-k routed experts and an always-active shared expert FFN."""

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        if config.ep_size != 1:
            raise NotImplementedError("Only ep_size=1 is supported")
        self.config = config
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList(
            Expert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_experts = Expert(
            config.hidden_size, config.n_shared_experts * config.moe_intermediate_size,
        ) if config.n_shared_experts else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, self.config.hidden_size)
        weights, indices = self.gate(tokens)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]) * weights[token_ids, slots, None]
            output.index_add_(0, token_ids, contribution)
        if self.shared_experts is not None:
            output = output + self.shared_experts(tokens)
        return output.reshape(x.shape)


class DeepSeekV3_2Block(nn.Module):
    """Pre-norm residual block; the first three layers use dense FFNs by default."""

    def __init__(self, config: DeepSeekV3_2Config, layer_idx: int = 0):
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError("Only hidden_act='silu' is supported")
        if config.moe_layer_freq <= 0 or config.first_k_dense_replace < 0:
            raise ValueError("moe_layer_freq must be positive and first_k_dense_replace nonnegative")
        self.rms_norm1 = DeepSeekV3_2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.multi_head_latent_attention = MultiHeadLatentAttention(config)
        self.rms_norm2 = DeepSeekV3_2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = (
            MoE(config)
            if layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0
            else Expert(config.hidden_size, config.intermediate_size)
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.multi_head_latent_attention(self.rms_norm1(x), attention_mask, position_ids)
        return x + self.ffn(self.rms_norm2(x))


class DeepSeekV3_2Model(nn.Module):
    """Return [batch, sequence, vocab_size] logits without an inference cache."""

    def __init__(self, config: DeepSeekV3_2Config):
        super().__init__()
        if min(config.num_hidden_layers, config.hidden_size, config.vocab_size,
               config.intermediate_size, config.moe_intermediate_size) <= 0:
            raise ValueError("Layer counts and model dimensions must be positive")
        if config.n_shared_experts is not None and config.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be nonnegative")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(DeepSeekV3_2Block(config, i) for i in range(config.num_hidden_layers))
        self.final_rmsnorm = DeepSeekV3_2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.output_layer.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process integer token IDs and an optional [batch, sequence] valid-key mask.

        position_ids has shape [1 or batch, sequence]. By default positions start
        at zero, including padding tokens; supply explicit positions for padding.
        """
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, attention_mask, position_ids)
        return self.output_layer(self.final_rmsnorm(x))
