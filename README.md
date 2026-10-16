# LLM architecture from scratch

Implementing LLM architectures from scratch for deep understanding of modern LLM architectures.

## Implemented

| Model | Model Size | Modality | Normalization | Positional Encoding | Attention | Mixture of Experts |
| --- | --- | --- | --- | --- | --- | --- |
| [GPT-2 XL](./gpt2-xl) | 1.5B | Text | LayerNorm | Learned Absolute | Multi-Head Self-Attention (causal) | ❌ |
| [Llama3-8B](./llama3-8b) | 8B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ❌ |
| [Llama 3.2-3B](./llama-3-2-3b/) | 3B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ❌ |
| [OlMo 2 7B](./olmo2/) | 7B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ❌ |
| [Phi-4](./phi4/) | 14B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ❌ |
| [DeepSeek v3](./deepseekv3/) | 671B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ✅ |
| [Gemma 3](./gemma3/) | 27B | Text + Image | RMSNorm | RoPE | Sliding window + GQA | ❌ |
| [xLSTM](./xlstm/) | 7B | Text | RMSNorm | RoPE | mLSTM | ❌ |
| [Mistral 3.1 Small](./mistral_3_1/) | 24B | Text + Image | RMSNorm | RoPE | Sliding window + GQA | ❌ |
| [Llama 4 Maverick](./llama4_maverick/) | 400B | Text | RMSNorm | RoPE | GQA | ✅ |
| [Qwen3](./qwen3_32b/) | 32B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) |  ❌ |
| [Qwen3 (235B-A22B)](./qwen3_235b_a22b/) | 235B A22B | Text | RMSNorm | RoPE | Grouped-Query Attention (causal) | ✅ |
| [SmolLM3](./smollm3/) | 3B | Text | RMSNorm | RoPE + NoPE | Grouped-Query Attention (causal) |  ❌ |

## In progress

- Kimi K2 (1T)
- GLM 4.5 (355B)
- GPT OSS 120B
- Grok 2.5 (270B)
- Qwen3 Next (80B-A3B)
