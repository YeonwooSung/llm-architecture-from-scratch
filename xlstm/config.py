"""
config: https://huggingface.co/NX-AI/xLSTM-7b/blob/main/config.json
"""
from dataclasses import dataclass, field


@dataclass
class XLSTMConfig:
    name_or_path: str = "NX-AI/xLSTM-7b"
    add_embedding_dropout: bool = False
    add_forward_backend_padding: bool = False
    add_out_norm: bool = True
    add_post_blocks_norm: bool = True
    add_post_norm: bool = False
    add_qk_norm: bool = False
    architectures: list[str] = field(default_factory=lambda: ["xLSTMForCausalLM"])
    autocast_kernel_dtype: str = "bfloat16"
    bos_token_id: int = 0
    cell_norm_eps: float = 1e-06
    chunk_size: int = 64
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    embedding_dim: int = 4096
    eos_token_id: int = 2
    eps: float = 1e-06
    ffn_proj_factor: float = 2.667
    ffn_round_up_to_multiple_of: int = 64
    force_bos_token_insert: bool = True
    gate_soft_cap: float = 15.0
    head_dim: int = 512
    igate_bias_init_range: float = -10.0
    inference_state_dtype: str = "float32"
    mlstm_round_up_to_multiple_of: int = 64
    mode: str = "inference"
    model_type: str = "xlstm"
    norm_eps: float = 1e-06
    norm_reduction_force_float32: bool = True
    num_blocks: int = 32
    num_heads: int = 8
    output_logit_soft_cap: float = 30.0
    pad_token_id: int = 1
    qk_dim_factor: float = 0.5
    return_last_states: bool = True
    sequence_kernel: str = "native_sequence__triton"
    step_kernel: str = "triton"
    tie_word_embeddings: bool = False
    torch_dtype: str = "float32"
    transformers_version: str = "4.48.0.dev0"
    use_bias: bool = False
    use_cache: bool = True
    v_dim_factor: float = 1.0
    vocab_size: int = 50304
    weight_mode: str = "single"
