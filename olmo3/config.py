"""
config: https://huggingface.co/allenai/Olmo3-7B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Olmo3Config:
    architectures: list[str] = field(default_factory=lambda: ["Olmo3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    dtype: str = "bfloat16"
    eos_token_id: int = 100257
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 27648
    layer_types: list[str] = field(
        default_factory=lambda: ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 16
    )
    max_position_embeddings: int = 65536
    model_type: str = "olmo3"
    num_attention_heads: int = 40
    num_hidden_layers: int = 64
    num_key_value_heads: int = 8
    pad_token_id: int = 100277
    rms_norm_eps: float = 1e-06
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
        }
    )
    rope_theta: float = 500000
    sliding_window: int = 4096
    tie_word_embeddings: bool = False
    transformers_version: str = "4.57.1"
    use_cache: bool = False
    vocab_size: int = 100278
