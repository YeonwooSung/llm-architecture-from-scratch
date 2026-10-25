"""
config: https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MiniMaxM2Config:
    architectures: list[str] = field(default_factory=lambda: ["MiniMaxM2ForCausalLM"])
    attention_dropout: float = 0.0
    attn_type_list: list[int] = field(default_factory=lambda: [1] * 62)
    auto_map: dict[str, str] = field(
        default_factory=lambda: {
            "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
            "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM",
        }
    )
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 3072
    initializer_range: float = 0.02
    intermediate_size: int = 1536
    layernorm_full_attention_beta: float = 1.0
    layernorm_linear_attention_beta: float = 1.0
    layernorm_mlp_beta: float = 1.0
    max_position_embeddings: int = 196608
    mlp_intermediate_size: int = 8192
    model_type: str = "minimax_m2"
    mtp_transformer_layers: int = 1
    num_attention_heads: int = 48
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 62
    num_key_value_heads: int = 8
    num_local_experts: int = 256
    num_mtp_modules: int = 3
    output_router_logits: bool = False
    qk_norm_type: str = "per_layer"
    quantization_config: dict[str, Any] = field(
        default_factory=lambda: {
            "activation_scheme": "dynamic",
            "fmt": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "gate",
                "e_score_correction_bias",
                "lm_head",
            ],
        }
    )
    rms_norm_eps: float = 1e-06
    rope_theta: float = 5000000
    rotary_dim: int = 64
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    scoring_func: str = "sigmoid"
    shared_intermediate_size: int = 0
    shared_moe_mode: str = "sigmoid"
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    transformers_version: str = "4.57.1"
    use_cache: bool = True
    use_mtp: bool = True
    use_qk_norm: bool = True
    use_routing_bias: bool = True
    vocab_size: int = 200064
