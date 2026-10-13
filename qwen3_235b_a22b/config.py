"""
config: https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3_235B_A22BConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen3MoeForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    decoder_sparse_step: int = 1
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 12288
    max_position_embeddings: int = 40960
    max_window_layers: int = 94
    mlp_only_layers: list[int] = field(default_factory=list)
    model_type: str = "qwen3_moe"
    moe_intermediate_size: int = 1536
    norm_topk_prob: bool = True
    num_attention_heads: int = 64
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 94
    num_key_value_heads: int = 4
    output_router_logits: bool = False
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 1000000.0
    router_aux_loss_coef: float = 0.001
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
