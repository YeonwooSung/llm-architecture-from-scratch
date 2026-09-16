"""
config: https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLaMA3_2_3BConfig:
    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 3072
    initializer_range: float = 0.02
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    model_type: str = "llama"
    num_attention_heads: int = 24
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
    )
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.45.0.dev0"
    use_cache: bool = True
    vocab_size: int = 128256
