"""
config: https://huggingface.co/xai-org/grok-2.5/blob/main/config.json
"""
from dataclasses import dataclass, field


@dataclass
class Grok2_5Config:
    architectures: list[str] = field(default_factory=lambda: ["Grok1ForCausalLM"])
    embedding_multiplier_scale: float = 90.50966799187809
    output_multiplier_scale: float = 0.5
    vocab_size: int = 131072
    hidden_size: int = 8192
    intermediate_size: int = 32768
    moe_intermediate_size: int = 16384
    max_position_embeddings: int = 131072
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    residual_moe: bool = True
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_hidden_layers: int = 64
    head_dim: int = 128
    rms_norm_eps: float = 1e-05
    final_logit_softcapping: float = 50
    attn_logit_softcapping: float = 30.0
    router_logit_softcapping: float = 30.0
    rope_theta: float = 208533496
    attn_temperature_len: int = 1024
    sliding_window_size: int = -1
    global_attn_every_n: int = 1
    model_type: str = "git"
    torch_dtype: str = "bfloat16"
    rope_type: str = "original"
    original_max_position_embeddings: int = 8192
    scaling_factor: float = 16.0
    extrapolation_factor: float = 1.0
    attn_factor: float = 1.0
    beta_fast: int = 8
    beta_slow: int = 1
