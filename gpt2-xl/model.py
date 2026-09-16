import torch
import torch.nn as nn
import math

from config import GPT2XLConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.scale = math.sqrt(self.token_embedding.embedding_dim)

    def forward(self, x):
        return self.token_embedding(x) * self.scale


class PositionalEmbedding(nn.Module):
    def __init__(self, config: GPT2XLConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        pe = torch.zeros(config.seq_len, config.n_embd)
        pos = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)

        i = torch.arange(0, config.n_embd//2)
        div_term = torch.exp(-math.log(10000)  * (2*i//config.n_embd))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        # The positional encoding values are used for model forward pass, but it is not a learnable parameter
        # Thus, we register it as a buffer so it is saved and moved with the model, but not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor and apply dropout
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


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
        self.token_embedding = TokenEmbedding(config)
        self.positional_embedding = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        # self.blocks = nn.ModuleList([GPT2XLBlock(config) for _ in range(config.num_layers)])
        self.gpt2xl_blocks = nn.Sequential(*[GPT2XLBlock(config) for _ in range(config.num_layers)])
        self.ln_f = LayerNorm(config)
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)


    def forward(self, x):
        # Implement the forward pass for the GPT2XL model
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        x = self.gpt2xl_blocks(x)
        x = self.ln_f(x)
        return self.output_layer(x)
