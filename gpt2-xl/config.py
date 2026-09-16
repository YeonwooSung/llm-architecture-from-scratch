"""
config: https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GPT2XLConfig:
    activation_function: str = "gelu_new"
    architectures: list[str] = field(default_factory=lambda: ["GPT2LMHeadModel"])
    attn_pdrop: float = 0.1
    bos_token_id: int = 50256
    embd_pdrop: float = 0.1
    eos_token_id: int = 50256
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    model_type: str = "gpt2"
    n_ctx: int = 1024
    n_embd: int = 1600
    n_head: int = 25
    n_layer: int = 48
    n_positions: int = 1024
    output_past: bool = True
    resid_pdrop: float = 0.1
    summary_activation: Optional[str] = None
    summary_first_dropout: float = 0.1
    summary_proj_to_labels: bool = True
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    task_specific_params: dict[str, Any] = field(
        default_factory=lambda: {
            "text-generation": {
                "do_sample": True,
                "max_length": 50,
            }
        }
    )
    vocab_size: int = 50257
