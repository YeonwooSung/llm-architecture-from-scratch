import torch
import torch.nn as nn

from config import XLSTMConfig


class XLSTMBlock(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super(XLSTMBlock, self).__init__()
        self.config = config

    def forward(self, x):
        #TODO
        pass


class XLSTMOutputLayer(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super(XLSTMOutputLayer, self).__init__()
        self.config = config
        # rmsnorm
        self.rmsnorm = nn.RMSNorm(config.embedding_dim, eps=config.eps)
        # linear layer
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x):
        # apply rmsnorm
        x = self.rmsnorm(x)
        # apply linear layer
        x = self.linear(x)
        return x


class XLSTM(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super(XLSTM, self).__init__()
        self.config = config

        # token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # xLSTM blocks
        self.xlstm_blocks = nn.ModuleList([XLSTMBlock(config) for _ in range(config.num_blocks)])
        # output layer
        self.output_layer = XLSTMOutputLayer(config)

    def forward(self, x):
        # apply token embedding
        x = self.token_embedding(x)
        # apply xLSTM blocks
        for block in self.xlstm_blocks:
            x = block(x)
        # apply output layer
        x = self.output_layer(x)
        return x
