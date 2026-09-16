import torch
import torch.nn as nn

from config import GPT2XLConfig


class LayerNorm(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.eps = config.layer_norm_epsilon
        self.alpha = nn.Parameter(torch.ones(self.n_embd))
        self.bias = nn.Parameter(torch.zeros(self.n_embd))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * x + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.config = config
        # Initialize multi-head attention layers here based on the configuration
        self.num_heads = config.num_heads
        self.d_model = config.n_embd

        self.d_k = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0, "make sure d_model % num_heads == 0"

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.mask = torch.triu(torch.ones(config.seq_len, config.seq_len), diagonal=1)
        self.register_buffer('mask', self.mask)

    def forward(self, x):
        #TODO
        pass


class GPT2XLBlock(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.config = config

        # Initialize block layers here based on the configuration
        self.ln_1 = LayerNorm(config)
        self.masked_self_attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x):
        #TODO Implement the forward pass for the GPT2XL block
        pass


class GPT2XL(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.config = config
        # Initialize model layers here based on the configuration
        #TODO token embedding
        #TODO positional embedding
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GPT2XLBlock(config) for _ in range(config.num_layers)])
        self.ln_f = LayerNorm(config)
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)


    def forward(self, x):
        #TODO Implement the forward pass for the GPT2XL model
        pass
