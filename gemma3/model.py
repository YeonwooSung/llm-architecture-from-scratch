import torch
import torch.nn as nn

from config import Gemma3Config


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        #TODO

    def forward(self, x):
        #TODO
        pass
