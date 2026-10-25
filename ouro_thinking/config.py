"""
config: https://huggingface.co/ByteDance-Seed/Ouro-1.4B-Thinking/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OuroThinkingConfig:
    architectures: list[str] = field(default_factory=lambda: ["OuroForCausalLM"])
    attention_dropout: float = 0.0
    auto_map: dict[str, str] = field(
        default_factory=lambda: {
            "AutoConfig": "configuration_ouro.OuroConfig",
            "AutoModel": "modeling_ouro.OuroModel",
            "AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM",
        }
    )
    bos_token_id: int = 1
    eos_token_id: int = 2
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 5632
    layer_types: list[str] = field(default_factory=lambda: ["full_attention"] * 48)
    max_position_embeddings: int = 65536
    max_window_layers: int = 48
    model_type: str = "ouro"
    num_attention_heads: int = 16
    num_hidden_layers: int = 48
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: float = 1000000.0
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    total_ut_steps: int = 4
    early_exit_threshold: float = 1.0
    transformers_version: str = "4.55.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 49152
