import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DeepSeekV3Config


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek V3 causal attention with compressed Q/KV and decoupled RoPE.

    The KV projection first produces a shared latent vector and a positional
    key. Each attention head's non-positional key and value are then expanded
    from that latent vector. This implementation processes a full sequence;
    it does not keep an inference KV cache.
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        if config.qk_rope_head_dim <= 0 or config.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if config.num_attention_heads <= 0 or config.kv_lora_rank <= 0:
            raise ValueError("num_attention_heads and kv_lora_rank must be positive")

        self.num_heads = config.num_attention_heads
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Queries optionally use a low-rank projection before expanding to heads.
        if config.q_lora_rank > 0:
            self.wq_a = nn.Linear(
                config.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.wq_b = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.wq = nn.Linear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=config.attention_bias
            )

        # The positional key bypasses KV compression and is shared by all heads.
        self.wkv_a = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias
        )
        self.kv_norm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(
            self.num_heads * config.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )

        # Store only frequencies; positions are generated for each forward call.
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, config.qk_rope_head_dim, 2, dtype=torch.float32)
            / config.qk_rope_head_dim
        ))
        scaling = config.rope_scaling
        if scaling is not None:
            if scaling.get("type") != "yarn":
                raise ValueError("MLA currently supports YaRN RoPE scaling only")
            factor = scaling["factor"]
            original_length = scaling["original_max_position_embeddings"]
            if factor <= 0 or original_length <= 0:
                raise ValueError("YaRN factor and original length must be positive")
            if config.max_position_embeddings > original_length:
                dim = config.qk_rope_head_dim

                def correction_dim(rotations: float) -> float:
                    return dim * math.log(original_length / (rotations * 2 * math.pi)) / (2 * math.log(config.rope_theta))

                low = max(math.floor(correction_dim(scaling["beta_fast"])), 0)
                high = min(math.ceil(correction_dim(scaling["beta_slow"])), dim - 1)
                if low == high:
                    high += 0.001
                ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                        / (high - low)).clamp(0, 1)
                smooth = 1 - ramp
                # YaRN blends original and stretched frequencies by dimension.
                inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

            mscale_all_dim = scaling.get("mscale_all_dim", 0)
            if factor > 1 and mscale_all_dim:
                mscale = 1 + 0.1 * mscale_all_dim * math.log(factor)
                self.softmax_scale *= mscale * mscale
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Rotate adjacent feature pairs in the positional Q/K subspace.

        Args:
            x: Tensor shaped (batch, sequence, heads, rope_dim).
            positions: Position indices for the sequence dimension.
        """
        angles = torch.outer(positions.float(), self.inv_freq)
        cos = angles.cos().view(1, x.size(1), 1, -1)
        sin = angles.sin().view(1, x.size(1), 1, -1)
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((
            pairs[..., 0] * cos - pairs[..., 1] * sin,
            pairs[..., 0] * sin + pairs[..., 1] * cos,
        ), dim=-1)
        return rotated.flatten(-2).to(x.dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return causal MLA output with the same (batch, sequence, hidden) shape.

        All positions attend only to themselves and earlier positions. RoPE
        starts at position zero on each call because this path has no KV cache.
        """
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            return x.clone()

        if self.config.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split(
            (self.config.qk_nope_head_dim, self.config.qk_rope_head_dim), dim=-1
        )

        # Expand the compressed KV latent into per-head content keys and values.
        kv, k_pe = self.wkv_a(x).split(
            (self.config.kv_lora_rank, self.config.qk_rope_head_dim), dim=-1
        )
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim
        )
        k_nope, v = kv.split(
            (self.config.qk_nope_head_dim, self.config.v_head_dim), dim=-1
        )

        positions = torch.arange(seq_len, device=x.device)
        q_pe = self._apply_rope(q_pe, positions)
        k_pe = self._apply_rope(k_pe.unsqueeze(2), positions)
        # Only the positional slices rotate; every head shares the positional key.
        q = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2)
        k = torch.cat(
            (k_nope, k_pe.expand(-1, -1, self.num_heads, -1)), dim=-1
        ).transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        # Mask future tokens before softmax so this works for causal language modeling.
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device,
            dtype=torch.bool
        ).triu(1)
        scores = scores.masked_fill(causal_mask, -torch.inf)
        attention = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attention = F.dropout(
            attention,
            p=self.config.attention_dropout,
            training=self.training
        )
        output = torch.matmul(attention, v).transpose(1, 2)
        return self.wo(output.reshape(batch_size, seq_len, -1))


class Expert(nn.Module):
    """SwiGLU feed-forward network used by routed and shared experts."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEGate(nn.Module):
    """Select experts using corrected scores, then weight them with raw scores."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=config.initializer_range)

        # This bias controls routing only; it does not change expert weights.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x, self.weight)
        if self.config.scoring_func == "sigmoid":
            scores = scores.float().sigmoid()
        else:
            scores = scores.float().softmax(dim=-1)

        choice_scores = scores + self.e_score_correction_bias
        if self.config.n_group > 1:
            grouped = choice_scores.reshape(x.size(0), self.config.n_group, -1)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected_groups = group_scores.topk(self.config.topk_group, dim=-1).indices
            allowed = torch.zeros_like(group_scores, dtype=torch.bool)
            allowed.scatter_(1, selected_groups, True)
            choice_scores = grouped.masked_fill(~allowed.unsqueeze(-1), -torch.inf)
            choice_scores = choice_scores.flatten(1)

        indices = choice_scores.topk(self.config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.config.routed_scaling_factor
        return weights.to(x.dtype), indices


class MoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        if config.ep_size != 1:
            raise ValueError("MoE currently supports ep_size=1 only")
        if config.topk_method != "noaux_tc":
            raise ValueError("MoE currently supports topk_method='noaux_tc' only")
        if config.scoring_func not in ("sigmoid", "softmax"):
            raise ValueError("scoring_func must be 'sigmoid' or 'softmax'")
        if config.n_group < 1 or config.n_routed_experts % config.n_group:
            raise ValueError("n_group must divide n_routed_experts")
        if not 1 <= config.topk_group <= config.n_group:
            raise ValueError("topk_group must be between 1 and n_group")
        experts_per_group = config.n_routed_experts // config.n_group
        if config.n_group > 1 and experts_per_group < 2:
            raise ValueError("noaux_tc requires at least two experts per group")
        if not 1 <= config.num_experts_per_tok <= config.topk_group * experts_per_group:
            raise ValueError("num_experts_per_tok exceeds the available routed experts")

        self.gate = MoEGate(config)
        self.experts = nn.ModuleList(
            Expert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_experts = Expert(
            config.hidden_size, config.n_shared_experts * config.moe_intermediate_size
        ) if config.n_shared_experts else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        tokens = x.reshape(-1, self.config.hidden_size)
        if tokens.size(0) == 0:
            return x.clone()

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
        return output.reshape(shape)


class DeepSeekV3Block(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        # RMS Norm 1
        self.rms_norm1 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Multi-head Latent Attention
        self.multi_head_latent_attention = MultiHeadLatentAttention(config)

        #\ RMS Norm 2
        self.rms_norm2 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE
        self.moe = MoE(config)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.rms_norm1(x)
        x = self.multi_head_latent_attention(x)
        x = x + residual

        residual = x
        x = self.rms_norm2(x)
        x = self.moe(x)
        x = x + residual

        return x


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        # token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # DeepSeekV3Block layers
        self.blocks = nn.ModuleList([
            DeepSeekV3Block(config) for _ in range(config.num_hidden_layers)
        ])

        # final RMS layer
        self.final_rmsnorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # output layer
        self.output_layer = nn.Linear(
            config.hidden_size, config.vocab_size
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_rmsnorm(x)
        x = self.output_layer(x)
        return x
