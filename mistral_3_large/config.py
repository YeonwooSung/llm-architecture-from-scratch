"""
config: https://huggingface.co/mistralai/Mistral-Large-3-2510/blob/main/params.json
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Mistral3LargeVisionEncoderConfig:
    adapter_bias: bool = False
    add_pre_mm_projector_layer_norm: bool = True
    hidden_size: int = 1664
    image_break_token_id: int = 12
    image_end_token_id: int = 13
    image_size: int = 1540
    image_token_id: int = 10
    intermediate_size: int = 8192
    max_image_size: int = 1540
    mm_projector_id: str = "patch_merge"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 48
    patch_size: int = 14
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2


@dataclass
class Mistral3LargeConfig:
    dim: int = 7168
    head_dim: int = 192
    hidden_dim: int = 16384
    kv_lora_rank: int = 512
    llama_4_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "beta": 0.1,
            "original_max_position_embeddings": 8192,
        }
    )
    max_position_embeddings: int = 294912
    moe: dict[str, Any] = field(
        default_factory=lambda: {
            "expert_hidden_dim": 4096,
            "expert_model_parallel": 1,
            "expert_parallel": 1,
            "first_k_dense_replace": 3,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "route_every_n": 1,
            "routed_scale": 1.0,
        }
    )
    n_heads: int = 128
    n_kv_heads: int = 128
    n_layers: int = 61
    norm_eps: float = 1e-06
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    quantization_config: dict[str, Any] = field(
        default_factory=lambda: {
            "config_groups": {
                "FP8_BLOCK": {
                    "format": "float-quantized",
                    "input_activations": {
                        "actorder": None,
                        "block_structure": None,
                        "dynamic": True,
                        "group_size": 128,
                        "num_bits": 8,
                        "observer": None,
                        "observer_kwargs": {},
                        "strategy": "group",
                        "symmetric": True,
                        "type": "float",
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": None,
                        "block_structure": [128, 128],
                        "dynamic": False,
                        "group_size": None,
                        "num_bits": 8,
                        "observer": "static_minmax",
                        "observer_kwargs": {},
                        "strategy": "block",
                        "symmetric": True,
                        "type": "float",
                    },
                }
            },
            "format": "float-quantized",
            "global_compression_ratio": None,
            "ignore": [
                "model.embed_tokens",
                "re:patch_merger.*",
                "re:vision_encoder.*",
                "re:vision_language_adapter.*",
                "re:.*kv_a_proj_with_mqa$",
                "re:.*q_a_proj$",
                "re:.*gate$",
                "lm_head",
            ],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.12.3.dev29+g73c2cf9.d20251119",
        }
    )
    rope_theta: float = 10000.0
    tied_embeddings: bool = False
    v_head_dim: int = 128
    vision_encoder: Mistral3LargeVisionEncoderConfig = field(default_factory=Mistral3LargeVisionEncoderConfig)
    vocab_size: int = 131072
    yarn: dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 1,
            "apply_scale": False,
            "beta": 32,
            "factor": 36,
            "original_max_position_embeddings": 8192,
        }
    )
