"""
config: https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Gemma3TextConfig:
    head_dim: int = 128
    hidden_size: int = 5376
    intermediate_size: int = 21504
    model_type: str = "gemma3_text"
    num_attention_heads: int = 32
    num_hidden_layers: int = 62
    num_key_value_heads: int = 16
    query_pre_attn_scalar: int = 168
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "factor": 8.0,
            "rope_type": "linear",
        }
    )
    sliding_window: int = 1024


@dataclass
class Gemma3VisionConfig:
    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    vision_use_head: bool = False


@dataclass
class Gemma3Config:
    architectures: list[str] = field(default_factory=lambda: ["Gemma3ForConditionalGeneration"])
    boi_token_index: int = 255999
    eoi_token_index: int = 256000
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    image_token_index: int = 262144
    initializer_range: float = 0.02
    mm_tokens_per_image: int = 256
    model_type: str = "gemma3"
    text_config: Gemma3TextConfig = field(default_factory=Gemma3TextConfig)
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.50.0.dev0"
    vision_config: Gemma3VisionConfig = field(default_factory=Gemma3VisionConfig)
