"""
config: https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GptOssConfig:
    architectures: list[str] = field(default_factory=lambda: ["GptOssForCausalLM"])
    attention_bias: bool = True
    attention_dropout: float = 0.0
    eos_token_id: int = 200002
    experts_per_token: int = 4
    head_dim: int = 64
    hidden_act: str = "silu"
    hidden_size: int = 2880
    initial_context_length: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 2880
    layer_types: list[str] = field(
        default_factory=lambda: [
            "sliding_attention",
            "full_attention",
        ]
        * 18
    )
    max_position_embeddings: int = 131072
    model_type: str = "gpt_oss"
    num_attention_heads: int = 64
    num_experts_per_tok: int = 4
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    num_local_experts: int = 128
    output_router_logits: bool = False
    pad_token_id: int = 199999
    quantization_config: dict[str, Any] = field(
        default_factory=lambda: {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
            "quant_method": "mxfp4",
        }
    )
    rms_norm_eps: float = 1e-05
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
        }
    )
    rope_theta: float = 150000
    router_aux_loss_coef: float = 0.9
    sliding_window: int = 128
    swiglu_limit: float = 7.0
    tie_word_embeddings: bool = False
    transformers_version: str = "4.55.0.dev0"
    use_cache: bool = True
    vocab_size: int = 201088
