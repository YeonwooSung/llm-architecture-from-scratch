import torch
from torch import nn
from torch.nn import functional as F

from config import Qwen3NextConfig


class Qwen3NextRMSNorm(nn.Module):
    """Qwen3-Next uses a zero-initialized offset from unit scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(x.dtype)


class Qwen3NextRMSNormGated(nn.Module):
    """DeltaNet uses ordinary RMSNorm followed by a SiLU output gate."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        normalized = self.weight * normalized.to(x.dtype)
        return (normalized * F.silu(gate.float())).to(x.dtype)


class RoPE(nn.Module):
    """Split-half rotary embeddings on only the leading part of each head."""

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()

        self.dim = int(config.head_dim * config.partial_rotary_factor)
        if not 0 < self.dim <= config.head_dim or self.dim % 2:
            raise ValueError("Rotary dimension must be positive, even, and <= head_dim")

        if config.rope_scaling is not None:
            raise NotImplementedError("Only default RoPE (rope_scaling=None) is supported")
        if config.rope_theta <= 0 or config.max_position_embeddings <= 0:
            raise ValueError("rope_theta and max_position_embeddings must be positive")

        self.theta = config.rope_theta


    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # Generate frequencies in float32 even after model.bfloat16().
        frequencies = self.theta ** (-torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        angles = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * frequencies
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        rotary, unchanged = x[..., :self.dim], x[..., self.dim:]
        left, right = rotary.chunk(2, dim=-1)
        rotated = rotary * angles.cos().to(x.dtype) + torch.cat((-right, left), -1) * angles.sin().to(x.dtype)
        return torch.cat((rotated, unchanged), dim=-1)


class Qwen3NextAttention(nn.Module):
    """Causal GQA with per-head QK normalization and a sigmoid output gate."""

    def __init__(self, config: Qwen3NextConfig):
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

        self.q_proj = nn.Linear(config.hidden_size, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, config.rms_norm_eps)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        # Queries and gates are packed per head, rather than in two global halves.
        q, gate = self.q_proj(x).view(batch, length, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(self.q_norm(q).transpose(1, 2), position_ids)
        k = self.rope(self.k_norm(k).transpose(1, 2), position_ids)
        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)

        allowed = None
        if attention_mask is not None:
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None] & attention_mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, is_causal=allowed is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2) * gate.sigmoid()
        return self.o_proj(result.reshape(batch, length, -1))


class Qwen3NextGatedDeltaNet(nn.Module):
    """
    Causal depthwise convolution followed by the recurrent gated delta rule.

    The explicit token loop is an educational reference implementation. Its
    recurrent state is [batch, value_heads, key_head_dim, value_head_dim];
    it never builds a sequence-by-sequence attention matrix.
    """

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        if min(self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim) <= 0:
            raise ValueError("DeltaNet head counts and dimensions must be positive")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if config.linear_conv_kernel_dim <= 0:
            raise ValueError("linear_conv_kernel_dim must be positive")
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkvz = nn.Linear(config.hidden_size, 2 * self.key_dim + 2 * self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(config.hidden_size, 2 * self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, config.linear_conv_kernel_dim,
            groups=conv_dim, padding=config.linear_conv_kernel_dim - 1, bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).clamp_min_(1e-6).log())
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        repeats = self.num_v_heads // self.num_k_heads
        # Checkpoint layout groups Q, K, V and output gates by KEY head.
        packed = self.in_proj_qkvz(x).view(batch, length, self.num_k_heads, -1)
        q, k, v, z = packed.split(
            [self.head_k_dim, self.head_k_dim, repeats * self.head_v_dim, repeats * self.head_v_dim], -1,
        )
        b, a = self.in_proj_ba(x).view(batch, length, self.num_k_heads, 2 * repeats).chunk(2, -1)
        b, a = b.reshape(batch, length, self.num_v_heads), a.reshape(batch, length, self.num_v_heads)
        z = z.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        qkv = torch.cat((q.reshape(batch, length, -1), k.reshape(batch, length, -1),
                         v.reshape(batch, length, -1)), dim=-1)
        qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[..., :length]).transpose(1, 2)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], -1)
        q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim).repeat_interleave(repeats, 2)
        k = k.reshape(batch, length, self.num_k_heads, self.head_k_dim).repeat_interleave(repeats, 2)
        v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)

        # Keep decay, L2 statistics and state updates in float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k = q.float(), k.float()
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q = q * self.head_k_dim ** -0.5
            beta = b.float().sigmoid()
            log_decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
            state = torch.zeros(
                batch, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                device=x.device, dtype=torch.float32,
            )
            outputs = []
            for t in range(length):
                state = state * log_decay[:, t, :, None, None].exp()
                prediction = (state * k[:, t, :, :, None]).sum(-2)
                delta = beta[:, t, :, None] * (v[:, t].float() - prediction)
                state = state + k[:, t, :, :, None] * delta[..., None, :]
                outputs.append((state * q[:, t, :, :, None]).sum(-2))
            result = torch.stack(outputs, dim=1).to(v.dtype)
        return self.out_proj(self.norm(result, z).reshape(batch, length, self.value_dim))


class Qwen3NextMLP(nn.Module):
    """Bias-free SwiGLU, used by routed experts, the shared expert and dense layers."""

    def __init__(self, config: Qwen3NextConfig, intermediate_size: int | None = None):
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


class Qwen3NextSparseMoeBlock(nn.Module):
    """Softmax top-k routed experts plus a separately sigmoid-gated shared expert."""

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        if not 1 <= config.num_experts_per_tok <= config.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = nn.ModuleList(
            Qwen3NextMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)
        )

        self.shared_expert = Qwen3NextMLP(config, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.size(-1))
        router_logits = self.gate(tokens)
        probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, selected = probabilities.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(tokens.dtype)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            if token_ids.numel() == 0:
                continue
            contribution = expert(tokens[token_ids]) * weights[token_ids, slots, None]
            output = output.index_add(0, token_ids, contribution.to(output.dtype))
        output = output + self.shared_expert(tokens) * self.shared_expert_gate(tokens).sigmoid()
        return output.reshape_as(x), router_logits


class Qwen3NextBlock(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int = 0):
        super().__init__()

        if config.full_attention_interval <= 0 or config.decoder_sparse_step <= 0:
            raise ValueError("full_attention_interval and decoder_sparse_step must be positive")
        if config.num_experts < 0:
            raise ValueError("num_experts must be nonnegative")

        self.layer_type = "full_attention" if (layer_idx + 1) % config.full_attention_interval == 0 else "linear_attention"
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config)
        else:
            self.linear_attn = Qwen3NextGatedDeltaNet(config)

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.is_moe_layer = (
            config.num_experts > 0 and layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.mlp = Qwen3NextSparseMoeBlock(config) if self.is_moe_layer else Qwen3NextMLP(config)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            mixed = self.self_attn(normalized, attention_mask, position_ids)
        else:
            mixed = self.linear_attn(normalized, attention_mask)
        x = x + mixed
        result = self.mlp(self.post_attention_layernorm(x))
        result, router_logits = result if self.is_moe_layer else (result, None)
        return x + result, router_logits


class Qwen3NextModel(nn.Module):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()

        if config.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if config.use_sliding_window:
            raise NotImplementedError("Only full causal attention is supported")
        if config.rope_scaling is not None:
            raise NotImplementedError("Only default RoPE (rope_scaling=None) is supported")
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen3NextBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = Qwen3NextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight


    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Return [batch, sequence, vocab] logits for [batch, sequence] IDs.

        attention_mask: [batch, sequence], 1/True for valid tokens. Full
        attention masks keys; DeltaNet zeros padded inputs before convolution.
        position_ids: [1 or batch, sequence], defaults to positions from zero;
        only full-attention layers use positions. Router logits, when requested,
        include padding and have shape [batch * sequence, num_experts] per MoE
        layer. Returns (logits, tuple of router logits) in that case.
        """

        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError("x must have shape [batch, nonempty sequence]")
        batch, length = x.shape

        if length > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings")

        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape [1 or batch, sequence]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if (position_ids < 0).any() or (position_ids >= self.config.max_position_embeddings).any():
            raise ValueError("position_ids must be within max_position_embeddings")

        collect_router = self.config.output_router_logits if output_router_logits is None else output_router_logits
        x = self.token_embedding(x)
        router_outputs = []
        for layer in self.layers:
            x, router_logits = layer(x, attention_mask, position_ids)
            if collect_router and router_logits is not None:
                router_outputs.append(router_logits)
        logits = self.lm_head(self.norm(x))

        return (logits, tuple(router_outputs)) if collect_router else logits
