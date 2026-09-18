"""
config: https://huggingface.co/microsoft/phi-4/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Phi4Config:
    name_or_path: str = "microsoft/phi-4"
    architectures: list[str] = field(default_factory=lambda: ["Phi3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 100257
    embd_pdrop: float = 0.0
    eos_token_id: int = 100265
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 17920
    max_position_embeddings: int = 16384
    model_type: str = "phi3"
    num_attention_heads: int = 40
    num_hidden_layers: int = 40
    num_key_value_heads: int = 10
    original_max_position_embeddings: int = 16384
    pad_token_id: int = 100349
    resid_pdrop: float = 0.0
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[dict] = None
    rope_theta: float = 250000
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.47.0"
    use_cache: bool = True
    vocab_size: int = 100352
