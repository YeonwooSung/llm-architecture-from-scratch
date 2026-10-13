"""
config: https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3_32BConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 25600
    max_position_embeddings: int = 40960
    max_window_layers: int = 64
    model_type: str = "qwen3"
    num_attention_heads: int = 64
    num_hidden_layers: int = 64
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 1000000
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
