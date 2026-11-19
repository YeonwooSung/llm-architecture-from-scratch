import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Mistral3LargeConfig


class RMSNorm(nn.Module):
    """Accumulate normalization in float32, retaining the activation dtype."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    """Compressed Q/KV with a shared, separately rotated positional key."""

    def __init__(self, config: Mistral3LargeConfig):
        super().__init__()
        self.config = config
        if min(config.dim, config.n_heads, config.kv_lora_rank,
               config.qk_nope_head_dim, config.v_head_dim) <= 0:
            raise ValueError("MLA dimensions and n_heads must be positive")
        if config.q_lora_rank < 0:
            raise ValueError("q_lora_rank must be nonnegative (zero uses a direct query projection)")
        if config.qk_rope_head_dim <= 0 or config.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")
        if config.head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim:
            raise ValueError("head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
        if config.n_kv_heads != config.n_heads:
            raise ValueError("MLA requires n_kv_heads == n_heads")
        if config.rope_theta <= 1:
            raise ValueError("rope_theta must be greater than one")

        self.softmax_scale = config.head_dim ** -0.5
        if config.q_lora_rank:
            self.wq_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
            self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
        else:
            self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wkv_a_with_mqa = nn.Linear(
            config.dim, config.kv_lora_rank + config.qk_rope_head_dim, bias=False,
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, config.norm_eps)
        self.wkv_b = nn.Linear(
            config.kv_lora_rank, config.n_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False,
        )
        self.wo = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

        dim = config.qk_rope_head_dim
        inv_freq = config.rope_theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        if config.yarn:
            yarn = config.yarn
            factor = yarn["factor"]
            original = yarn["original_max_position_embeddings"]
            if factor < 1 or original <= 0 or not 0 < yarn["alpha"] <= yarn["beta"]:
                raise ValueError("YaRN requires factor >= 1, positive original length and 0 < alpha <= beta")

            def correction(rotations):
                return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(config.rope_theta))

            low = max(math.floor(correction(yarn["beta"])), 0)
            high = min(math.ceil(correction(yarn["alpha"])), dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                    / (high - low if high != low else 0.001)).clamp(0, 1)
            inv_freq = inv_freq * (1 - ramp) + inv_freq / factor * ramp
            # Native apply_scale=False disables YaRN magnitude correction.
            if yarn.get("apply_scale", True):
                self.softmax_scale *= (1 + 0.1 * math.log(factor)) ** 2
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if config.llama_4_scaling and (
            config.llama_4_scaling["original_max_position_embeddings"] <= 0
            or config.llama_4_scaling["beta"] < 0
        ):
            raise ValueError("Llama 4 scaling requires a positive original length and nonnegative beta")

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Adjacent feature pairs, as in the native Mistral checkpoints.
        angles = positions.float()[..., None] * self.inv_freq.float()
        cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
        pairs = x.float().unflatten(-1, (-1, 2))
        rotated = torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                               pairs[..., 0] * sin + pairs[..., 1] * cos), dim=-1)
        return rotated.flatten(-2).to(x.dtype)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        batch, length, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device)[None]
        elif position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        position_ids = position_ids.to(x.device)
        if attention_mask is not None and attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, sequence]")

        q = self.wq_b(self.q_a_norm(self.wq_a(x))) if cfg.q_lora_rank else self.wq(x)
        q = q.reshape(batch, length, cfg.n_heads, cfg.head_dim)
        q_nope, q_pe = q.split((cfg.qk_nope_head_dim, cfg.qk_rope_head_dim), dim=-1)
        latent, k_pe = self.wkv_a_with_mqa(x).split((cfg.kv_lora_rank, cfg.qk_rope_head_dim), dim=-1)
        kv = self.wkv_b(self.kv_a_norm(latent)).reshape(
            batch, length, cfg.n_heads, cfg.qk_nope_head_dim + cfg.v_head_dim,
        )
        k_nope, v = kv.split((cfg.qk_nope_head_dim, cfg.v_head_dim), dim=-1)
        q = torch.cat((q_nope, self._apply_rope(q_pe, position_ids)), dim=-1)
        k_pe = self._apply_rope(k_pe.unsqueeze(2), position_ids)
        k = torch.cat((k_nope, k_pe.expand(-1, -1, cfg.n_heads, -1)), dim=-1)
        if cfg.llama_4_scaling:
            scaling = cfg.llama_4_scaling
            scale = 1 + scaling["beta"] * torch.log1p(
                torch.floor(position_ids.float() / scaling["original_max_position_embeddings"])
            )
            q = q * scale[..., None, None].to(q.dtype)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))

        # SDPA also returns zeros for fully masked rows (e.g. left padding).
        allowed = None
        if attention_mask is not None:
            allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()[None, None]
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None, scale=self.softmax_scale,
        )
        return self.wo(output.transpose(1, 2).reshape(batch, length, cfg.n_heads * cfg.v_head_dim))


class Expert(nn.Module):
    """Bias-free SwiGLU, also used for the first dense feed-forward layers."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEGate(nn.Module):
    """Softmax routing with normalized top-k weights; no correction bias."""

    def __init__(self, config: Mistral3LargeConfig):
        super().__init__()
        moe = config.moe
        self.top_k = moe["num_experts_per_tok"]
        self.routed_scale = moe["routed_scale"]
        self.weight = nn.Parameter(torch.empty(moe["num_experts"], config.dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        # Softmax over selected logits equals renormalizing global top-k probabilities.
        logits, indices = scores.topk(self.top_k, dim=-1)
        weights = logits.softmax(-1) * self.routed_scale
        return weights.to(x.dtype), indices


class MoE(nn.Module):
    def __init__(self, config: Mistral3LargeConfig):
        super().__init__()
        moe = config.moe
        if moe["expert_parallel"] != 1 or moe["expert_model_parallel"] != 1:
            raise ValueError("Only single-device experts are supported")
        if moe["num_expert_groups"] != 1 or moe["num_expert_groups_per_tok"] != 1:
            raise ValueError("Only ungrouped Mistral routing is supported")
        if not 1 <= moe["num_experts_per_tok"] <= moe["num_experts"]:
            raise ValueError("num_experts_per_tok must be between one and num_experts")
        if moe["expert_hidden_dim"] <= 0 or moe["num_shared_experts"] < 0:
            raise ValueError("Invalid expert dimensions or shared expert count")
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList(
            Expert(config.dim, moe["expert_hidden_dim"]) for _ in range(moe["num_experts"])
        )
        self.shared_experts = (
            Expert(config.dim, moe["expert_hidden_dim"] * moe["num_shared_experts"])
            if moe["num_shared_experts"] else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, x.size(-1))
        weights, indices = self.gate(tokens)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel():
                contribution = expert(tokens[token_ids]) * weights[token_ids, slots, None]
                output.index_add_(0, token_ids, contribution)
        if self.shared_experts is not None:
            output = output + self.shared_experts(tokens)
        return output.reshape_as(x)


class Mistral3LargeBlock(nn.Module):
    def __init__(self, config: Mistral3LargeConfig, layer_idx: int):
        super().__init__()
        moe = config.moe
        if moe["route_every_n"] <= 0 or moe["first_k_dense_replace"] < 0:
            raise ValueError("route_every_n must be positive and first_k_dense_replace nonnegative")
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = (
            MoE(config) if layer_idx >= moe["first_k_dense_replace"] and layer_idx % moe["route_every_n"] == 0
            else Expert(config.dim, config.hidden_dim)
        )

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.attention(self.attention_norm(x), attention_mask, position_ids)
        return x + self.feed_forward(self.ffn_norm(x))


class Mistral3LargeModel(nn.Module):
    """Return [batch, sequence, vocab] logits for a full sequence of token IDs.

    Default dimensions describe a hundreds-of-billions parameter model. Use a
    reduced config for local experiments, or torch.device('meta') to inspect it.
    Vision and quantization fields in the config are checkpoint metadata here.
    """

    def __init__(self, config: Mistral3LargeConfig):
        super().__init__()
        if min(config.dim, config.hidden_dim, config.n_layers, config.vocab_size,
               config.max_position_embeddings) <= 0 or config.norm_eps <= 0:
            raise ValueError("Model dimensions, context length and norm_eps must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Mistral3LargeBlock(config, i) for i in range(config.n_layers))
        self.final_rmsnorm = RMSNorm(config.dim, config.norm_eps)
        self.output_layer = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tied_embeddings:
            self.output_layer.weight = self.token_embedding.weight


    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must have shape [nonempty batch, nonempty sequence]")
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")
        if position_ids is not None:
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("position_ids must contain integer positions")
            if torch.any(position_ids < 0) or torch.any(position_ids >= self.config.max_position_embeddings):
                raise ValueError("position_ids are outside max_position_embeddings")
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        return self.output_layer(self.final_rmsnorm(x))
