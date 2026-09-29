"""
config: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeepSeekV3Config:
    architectures: list[str] = field(default_factory=lambda: ["DeepseekV3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    auto_map: dict[str, str] = field(
        default_factory=lambda: {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModel": "modeling_deepseek.DeepseekV3Model",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
        }
    )
    bos_token_id: int = 0
    eos_token_id: int = 1
    ep_size: int = 1
    first_k_dense_replace: int = 3
    hidden_act: str = "silu"
    hidden_size: int = 7168
    initializer_range: float = 0.02
    intermediate_size: int = 18432
    kv_lora_rank: int = 512
    max_position_embeddings: int = 163840
    model_type: str = "deepseek_v3"
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    n_group: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    norm_topk_prob: bool = True
    num_attention_heads: int = 128
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 61
    num_key_value_heads: int = 128
    num_nextn_predict_layers: int = 1
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    quantization_config: dict[str, Any] = field(
        default_factory=lambda: {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    )
    rms_norm_eps: float = 1e-06
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    rope_theta: float = 10000
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    tie_word_embeddings: bool = False
    topk_group: int = 4
    topk_method: str = "noaux_tc"
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.33.1"
    use_cache: bool = True
    v_head_dim: int = 128
    vocab_size: int = 129280
