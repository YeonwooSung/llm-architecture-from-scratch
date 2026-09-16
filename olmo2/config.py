"""
config: https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Olmo2Config:
    architectures: list[str] = field(default_factory=lambda: ["Olmo2ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    eos_token_id: int = 100257
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    model_type: str = "olmo2"
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    pad_token_id: int = 100277
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.47.0.dev0"
    use_cache: bool = False
    vocab_size: int = 100352
