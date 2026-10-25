"""
config: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3NextConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen3NextForCausalLM"])
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    decoder_sparse_step: int = 1
    eos_token_id: int = 151645
    full_attention_interval: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 5120
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    max_position_embeddings: int = 262144
    mlp_only_layers: list[int] = field(default_factory=list)
    model_type: str = "qwen3_next"
    moe_intermediate_size: int = 512
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_experts: int = 512
    num_experts_per_tok: int = 10
    num_hidden_layers: int = 48
    num_key_value_heads: int = 2
    output_router_logits: bool = False
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000000
    router_aux_loss_coef: float = 0.001
    shared_expert_intermediate_size: int = 512
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.57.0.dev0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
