"""
config: https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GLM4_5Config:
    architectures: list[str] = field(default_factory=lambda: ["Glm4MoeForCausalLM"])
    attention_bias: bool = True
    attention_dropout: float = 0.0
    pad_token_id: int = 151329
    eos_token_id: list[int] = field(default_factory=lambda: [151329, 151336, 151338])
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 5120
    partial_rotary_factor: float = 0.5
    initializer_range: float = 0.02
    intermediate_size: int = 12288
    max_position_embeddings: int = 131072
    model_type: str = "glm4_moe"
    moe_intermediate_size: int = 1536
    norm_topk_prob: bool = True
    num_attention_heads: int = 96
    n_group: int = 1
    topk_group: int = 1
    n_routed_experts: int = 160
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.5
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    num_hidden_layers: int = 92
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[dict] = None
    rope_theta: float = 1000000
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.54.0"
    use_cache: bool = True
    use_qk_norm: bool = True
    vocab_size: int = 151552
