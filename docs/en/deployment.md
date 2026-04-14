# Deployment Overview

This document describes a deployment path that has already been validated:

- Model: `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- Framework: `vLLM 0.19.0`
- Environment: `WSL2 + Ubuntu 24.04`
- GPU: `2 x RTX 2080 Ti`
- Parallelism: `TP=2`
- Backend: `TRITON_ATTN`

## Why This Is Tricky

The main challenge is not just model size. The real constraints are:

- `Gemma 4` is forced onto `TRITON_ATTN` in vLLM
- `RTX 2080 Ti` is `Turing / cc 7.5`
- default Triton settings on older GPUs can hit shared-memory limits

That is why this setup usually needs:

- `TP=2`
- AWQ 4-bit weights
- controlled `max-model-len` and `max-num-batched-tokens`
- conservative handling for `TRITON_ATTN` on `Turing`

## Validated Baseline Parameters

```bash
MODEL=ebircak/gemma-4-31B-it-4bit-W4A16-AWQ
PORT=8001
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
```

## Baseline Validation

The baseline has already passed:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Practical Advice

If you are reproducing this from scratch:

1. Bring up the baseline first.
2. Verify `/health` and `/v1/models`.
3. Verify a real `chat/completions` request.
4. Only then move on to `TurboQuant KV Q4`.
