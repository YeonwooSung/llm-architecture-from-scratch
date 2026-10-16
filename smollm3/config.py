"""
config: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SmolLM3Config:
    architectures: list[str] = field(default_factory=lambda: ["SmolLM3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: int = 128001
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 65536
    model_type: str = "smollm3"
    no_rope_layer_interval: int = 4
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 4
    pad_token_id: int = 128004
    pretraining_tp: int = 2
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 5000000.0
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.53.0.dev0"
    transformers_js_config: dict[str, Any] = field(
        default_factory=lambda: {
            "use_external_data_format": True,
            "dtype": "q4",
            "kv_cache_dtype": {
                "q4f16": "float16",
                "fp16": "float16",
            },
        }
    )
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 128256
