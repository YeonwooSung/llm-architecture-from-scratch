"""
config: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mistral3TextConfig:
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 32768
    max_position_embeddings: int = 131072
    model_type: str = "mistral"
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_theta: float = 1000000000.0
    sliding_window: Optional[int] = None
    use_cache: bool = True
    vocab_size: int = 131072


@dataclass
class Mistral3VisionConfig:
    attention_dropout: float = 0.0
    head_dim: int = 64
    hidden_act: str = "silu"
    hidden_size: int = 1024
    image_size: int = 1540
    initializer_range: float = 0.02
    intermediate_size: int = 4096
    model_type: str = "pixtral"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 24
    patch_size: int = 14
    rope_theta: float = 10000.0


@dataclass
class Mistral3Config:
    architectures: list[str] = field(default_factory=lambda: ["Mistral3ForConditionalGeneration"])
    image_token_index: int = 10
    model_type: str = "mistral3"
    multimodal_projector_bias: bool = False
    projector_hidden_act: str = "gelu"
    spatial_merge_size: int = 2
    text_config: Mistral3TextConfig = field(default_factory=Mistral3TextConfig)
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.50.0.dev0"
    vision_config: Mistral3VisionConfig = field(default_factory=Mistral3VisionConfig)
    vision_feature_layer: int = -1
