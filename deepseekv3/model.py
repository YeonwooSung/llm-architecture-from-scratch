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


class MoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        #TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO Implement the forward pass for the MoE module
        pass


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
