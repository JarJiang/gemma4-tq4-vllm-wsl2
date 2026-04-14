# TurboQuant KV Q4 Notes

This document describes the experimental route that has already been pushed to real generation, not an upstream one-click feature.

## Validated Parameters

```bash
VLLM_PLUGINS=tq4_backend
TQ4_K_BITS=4
TQ4_V_BITS=4
TQ4_CG_DECOMPRESS_TOKENS_CAP=256
TQ4_CG_PREFILL_TOKENS_CAP=256
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_BATCHED_TOKENS=1024
MAX_MODEL_LEN=8192
```

Unchanged pieces:

- model: `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- parallelism: `TP=2`
- attention backend: `TRITON_ATTN`

## What Was Required

This was not just a matter of enabling a plugin flag. The work included:

- a Turing-specific attention workaround in `vLLM`
- adapting `turboquant-vllm` from a FlashAttention-oriented path to `TRITON_ATTN`
- enabling packed KV for both full-attention and sliding-window layers
- fixing generation-time KV dtype and cache layout issues
- reducing runtime scratch pressure to leave headroom on `2080 Ti`

## Current Outcome

This route now supports:

- `GET /v1/models`
- `POST /v1/chat/completions`
- real generation output

KV cache capacity also increased materially to roughly:

- `37k+ tokens`

which is significantly above the earlier default-KV path.

## Current Limitations

This remains an engineering PoC:

- parameters are sensitive
- it is not an upstream officially supported Gemma 4 TurboQuant route
- `2080 Ti` still benefits from conservative memory settings
