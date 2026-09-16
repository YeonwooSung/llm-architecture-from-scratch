"""
config: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Llama3Config:
    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    max_position_embeddings: int = 8192
    model_type: str = "llama"
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[dict] = None
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.40.0.dev0"
    use_cache: bool = True
    vocab_size: int = 128256
