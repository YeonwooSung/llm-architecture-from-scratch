import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Phi4Config


class Phi4Block(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.config = config

        #TODO RMSNorm 1
        #TODO GQA + RoPE
        #TODO RMSNorm 2
        #TODO MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Phi4Model(nn.Module):
    def __init__(self, config: Phi4Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        #TODO phi4 blocks

        # final RMSNorm before output
        self.final_rmsnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
