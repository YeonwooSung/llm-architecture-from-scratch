"""
config: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Llama4TextConfig:
    attn_implementation_autoset: bool = True
    attention_bias: bool = False
    attention_chunk_size: int = 8192
    attention_dropout: float = 0.0
    bos_token_id: int = 200000
    eos_token_id: list[int] = field(default_factory=lambda: [200001, 200007, 200008])
    for_llm_compressor: bool = False
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 5120
    initializer_range: float = 0.02
    interleave_moe_layer_step: int = 2
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    max_position_embeddings: int = 1048576
    model_type: str = "llama4_text"
    num_attention_heads: int = 40
    num_experts_per_tok: int = 1
    num_hidden_layers: int = 48
    num_key_value_heads: int = 8
    num_local_experts: int = 128
    output_router_logits: bool = False
    pad_token_id: int = 200018
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[dict] = None
    rope_theta: float = 500000.0
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_qk_norm: bool = False
    vocab_size: int = 202048


@dataclass
class Llama4VisionConfig:
    attn_implementation_autoset: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "gelu"
    hidden_size: int = 1408
    image_size: int = 336
    initializer_range: float = 0.02
    intermediate_size: int = 5632
    model_type: str = "llama4_vision_model"
    multi_modal_projector_bias: bool = False
    norm_eps: float = 1e-05
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 34
    patch_size: int = 14
    pixel_shuffle_ratio: float = 0.5
    projector_dropout: float = 0.0
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    rope_theta: float = 10000
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "default"
    vision_output_dim: int = 4096


@dataclass
class Llama4Config:
    architectures: list[str] = field(default_factory=lambda: ["Llama4ForConditionalGeneration"])
    boi_token_index: int = 200080
    eoi_token_index: int = 200081
    image_token_index: int = 200092
    model_type: str = "llama4"
    text_config: Llama4TextConfig = field(default_factory=Llama4TextConfig)
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0.dev0"
    vision_config: Llama4VisionConfig = field(default_factory=Llama4VisionConfig)
