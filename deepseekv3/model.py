import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DeepSeekV3Config


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        #TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO Implement the forward pass for the Multi-head Latent Attention module
        pass


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
