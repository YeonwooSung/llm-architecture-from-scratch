import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import XLSTMConfig


class XLSTMRMSNorm(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.force_float32 = config.norm_reduction_force_float32
        self.weight = nn.Parameter(torch.ones(config.embedding_dim))
        self.bias = nn.Parameter(torch.zeros(config.embedding_dim)) if config.use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float() if self.force_float32 and x.dtype != torch.float64 else x
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        output = normalized.to(x.dtype) * self.weight
        return output if self.bias is None else output + self.bias


class MLSTMLayer(nn.Module):
    """
    xLSTM-7B matrix memory, using a differentiable PyTorch recurrence.

    Input/output: [batch, sequence, embedding_dim]. Each call starts with
    empty memory; backend/kernel and cache settings are not used here.
    """

    def __init__(self, config: XLSTMConfig):
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)
        self.v_dim = int(config.embedding_dim * config.v_dim_factor)

        if self.num_heads <= 0 or any(
            dim <= 0 or dim % self.num_heads for dim in (self.qk_dim, self.v_dim)
        ):
            raise ValueError("Q/K and V dimensions must be positive multiples of num_heads")

        self.qk_head_dim = self.qk_dim // self.num_heads
        self.v_head_dim = self.v_dim // self.num_heads

        self.q_proj = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)
        self.output_gate = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)

        # Input and forget gates have one scalar per head and always use biases.
        self.input_gate = nn.Linear(config.embedding_dim, self.num_heads, bias=True)
        self.forget_gate = nn.Linear(config.embedding_dim, self.num_heads, bias=True)
        self.head_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.v_head_dim))
        self.head_norm_bias = (
            nn.Parameter(torch.zeros(self.num_heads, self.v_head_dim)) if config.use_bias else None
        )
        self.out_proj = nn.Linear(self.v_dim, config.embedding_dim, bias=config.use_bias)


    def _soft_cap(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Apply a soft cap to the gate values to prevent them from becoming too large.

        Args:
            gate (torch.Tensor): The input gate tensor.

        Returns:
            torch.Tensor: The gated tensor after applying the soft cap.
        """
        cap = self.config.gate_soft_cap
        return gate if cap is None else cap * torch.tanh(gate / cap)


    def _recurrent_memory(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        input_gate: torch.Tensor, forget_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the recurrent memory for the xLSTM layer.

        Args:
            q (torch.Tensor): The query tensor of shape [B, S, H, Dq].
            k (torch.Tensor): The key tensor of shape [B, S, H, Dq].
            v (torch.Tensor): The value tensor of shape [B, S, H, Dv].
            input_gate (torch.Tensor): The input gate tensor of shape [B, S, H].
            forget_gate (torch.Tensor): The forget gate tensor of shape [B, S, H].

        Returns:
            torch.Tensor: The output tensor after applying the recurrent memory, of shape [B, S, H, Dv].
        """

        #
        # Reads tokens sequentially and updates the memo,
        # and extract the required information for current token.
        #
        # `key`: clue for finding relevant memory entries.
        # `value`: contains the information to be stored in the memory.
        # `query`: clue for retrieving relevant information from the memory for the current token.
        #

        # Q/K: [B, S, H, Dq], V: [B, S, H, Dv]. Accumulate low precision
        # inputs in float32 to keep the matrix memory and gates stable.
        batch_size, seq_len = q.shape[:2]
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32

        q, k, v = (tensor.to(dtype) for tensor in (q, k, v))
        input_gate = input_gate.to(dtype)

        log_forget = F.logsigmoid(forget_gate.to(dtype))
        memory = q.new_zeros(batch_size, self.num_heads, self.qk_head_dim, self.v_head_dim)
        normalizer = q.new_zeros(batch_size, self.num_heads, self.qk_head_dim)
        log_scale = q.new_zeros(batch_size, self.num_heads, 1)

        outputs = []
        for t in range(seq_len):
            log_i = input_gate[:, t].unsqueeze(-1)
            log_f = log_forget[:, t].unsqueeze(-1) + log_scale
            next_scale = torch.maximum(log_f, log_i)

            # Rescale the exponential input gate rather than exponentiating
            # large logits directly: C_t = f_t C_(t-1) + i_t k_t v_t^T.

            i = torch.exp(log_i - next_scale)
            f = torch.exp(log_f - next_scale)
            key, value = k[:, t], v[:, t]

            memory = f.unsqueeze(-1) * memory + i.unsqueeze(-1) * (
                key.unsqueeze(-1) * value.unsqueeze(-2)
            )
            normalizer = f * normalizer + i * key

            log_scale = next_scale
            query = q[:, t] * self.qk_head_dim ** -0.5

            numerator = (query.unsqueeze(-1) * memory).sum(dim=-2)
            denominator = torch.maximum(
                (query * normalizer).sum(dim=-1, keepdim=True).abs(),
                torch.exp(-log_scale),
            )
            outputs.append(numerator / (denominator + self.config.eps))
        return torch.stack(outputs, dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            return x.new_empty(batch_size, 0, self.config.embedding_dim)

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.v_head_dim)

        h = self._recurrent_memory(
            q, k, v, self._soft_cap(self.input_gate(x)), self._soft_cap(self.forget_gate(x)),
        ).to(v.dtype)

        # The 7B variant uses LayerNorm separately within each value head.
        work = h.float() if self.config.norm_reduction_force_float32 and h.dtype != torch.float64 else h
        h = F.layer_norm(work, (self.v_head_dim,), eps=self.config.norm_eps).to(h.dtype)
        h = h * self.head_norm_weight
        if self.head_norm_bias is not None:
            h = h + self.head_norm_bias
        h = h.reshape(batch_size, seq_len, self.v_dim)
        return self.out_proj(torch.sigmoid(self.output_gate(x)) * h)


class XLSTMFeedForward(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super().__init__()
        multiple = config.ffn_round_up_to_multiple_of
        if multiple <= 0 or config.ffn_proj_factor <= 0:
            raise ValueError("FFN projection factor and rounding multiple must be positive")
        hidden_dim = math.ceil(config.embedding_dim * config.ffn_proj_factor / multiple) * multiple
        self.gate_proj = nn.Linear(config.embedding_dim, hidden_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(config.embedding_dim, hidden_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(hidden_dim, config.embedding_dim, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class XLSTMBlock(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super(XLSTMBlock, self).__init__()
        self.config = config

        # RMSNorm 1
        self.rmsnorm1 = XLSTMRMSNorm(config)
        # mLSTM layer
        self.mlstm = MLSTMLayer(config)
        # RMSNorm 2
        self.rmsnorm2 = XLSTMRMSNorm(config)
        # Feed-Forward Network (SwiGLU)
        self.ffn = XLSTMFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mLSTM and SwiGLU residual branches to [B, S, D]."""
        if x.ndim != 3 or x.shape[-1] != self.config.embedding_dim:
            raise ValueError("Input must have shape [batch, sequence, embedding_dim]")
        x = x + self.mlstm(self.rmsnorm1(x))
        return x + self.ffn(self.rmsnorm2(x))


class XLSTMOutputLayer(nn.Module):
    def __init__(self, config: XLSTMConfig):
        super(XLSTMOutputLayer, self).__init__()
        self.config = config
        # rmsnorm
        self.rmsnorm = XLSTMRMSNorm(config)
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
