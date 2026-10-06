import torch
import torch.nn as nn

from config import Gemma3Config


class Gemma3Block(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config

        #TODO Pre RMSNorm 1
        #TODO sliding window + GQA
        #TODO Pre RMSNorm 2
        #TODO MLP

    def forward(self, x):
        #TODO implement the forward pass for the Gemma3 block
        pass


class Gemma3OutputLayer(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        #TODO RMSNorm
        self.rms_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps,)
        #TODO linear projection
        self.linear = nn.Linear(config.text_config.hidden_size, config.image_token_index)

    def forward(self, x):
        #TODO implement the forward pass for the output layer
        pass


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config

        # token embedding layer
        self.token_embedding = nn.Embedding(config.image_token_index, config.text_config.hidden_size)

        #TODO gemma blocks
        #TODO output layer

    def forward(self, x):
        #TODO
        pass
