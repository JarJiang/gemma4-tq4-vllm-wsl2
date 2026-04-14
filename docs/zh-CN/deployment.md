# 部署总览

本文档对应一条已经实测跑通的部署路线：

- 模型：`ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- 框架：`vLLM 0.19.0`
- 系统：`WSL2 + Ubuntu 24.04`
- GPU：`2 x RTX 2080 Ti`
- 并行：`TP=2`
- 后端：`TRITON_ATTN`

## 为什么这条路线不简单

这里真正的难点不是“Gemma 4 太大”，而是：

- `Gemma 4` 在 vLLM 中会被强制走 `TRITON_ATTN`
- `RTX 2080 Ti` 是 `Turing / cc 7.5`
- 默认 Triton 配置在这类老卡上可能触发 shared memory 限制

因此，想把它在双 `2080 Ti` 上跑稳，通常需要：

- 保持 `TP=2`
- 使用 AWQ 4bit 权重
- 控制 `max-model-len` 和 `max-num-batched-tokens`
- 给 `TRITON_ATTN` 做面向 `Turing` 的保守化处理

## 已验证的基线参数

```bash
MODEL=ebircak/gemma-4-31B-it-4bit-W4A16-AWQ
PORT=8001
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
```

对应启动思路：

```bash
vllm serve ebircak/gemma-4-31B-it-4bit-W4A16-AWQ \
  --quantization compressed-tensors \
  --dtype auto \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --disable-custom-all-reduce \
  --port 8001
```

## 基线能力

这套配置已经验证过：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## 实战建议

如果你是第一次复现，建议顺序如下：

1. 先把基线服务跑通，不要一上来就改 KV 压缩。
2. 先确认 `/health` 和 `/v1/models` 都正常。
3. 先用一个最小 `chat/completions` 请求验证真实生成。
4. 基线稳定后，再切到 `TurboQuant KV Q4` 实验参数。

## 网络说明

在一些 Windows + WSL2 环境中：

- WSL 内部访问服务正常
- Windows 到 WSL 的访问可能会被 `Hyper-V / WSL 防火墙` 影响

所以“服务在 WSL 内跑通”和“Windows 侧能直接访问”是两件事，排障时不要混在一起。
