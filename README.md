# Gemma 4 TQ4 vLLM WSL2

中文 | [English](#english)

在双 `RTX 2080 Ti` 上，通过 `WSL2 + vLLM + Tensor Parallel` 实际部署 `Gemma 4 31B AWQ`，并进一步把 `TurboQuant KV Q4` 路线推进到可真实生成。

这是一个偏实践、偏踩坑记录的开源仓库，目标不是“理论综述”，而是把下面这件事真正做通：

- `Gemma 4 31B AWQ`
- `vLLM 0.19.0`
- `WSL2 / Ubuntu 24.04`
- `2 x RTX 2080 Ti`
- `TP=2`
- `TRITON_ATTN`
- `TurboQuant KV-cache Q4`

## 项目状态

当前状态：

- 已跑通 `Gemma 4 31B AWQ + vLLM + WSL2 + 双卡 TP2`
- 已验证 `TurboQuant KV Q4` 在 `TRITON_ATTN` 路径上可以启动并完成真实生成
- 已确认在 `Turing / 2080 Ti / cc 7.5` 上需要额外 workaround
- 已确认这条路线不是“开箱即用”，需要对 `vLLM` 和 `turboquant-vllm` 做定向补丁

一句话结论：

`Gemma 4 31B AWQ` 可以在双 `2080 Ti` 的 `WSL2 + vLLM` 环境中运行，且 `TurboQuant KV Q4` 这条实验路线已经推进到“能实际返回结果”，但仍属于工程化 PoC，不是官方原生支持路径。

## 已验证配置

本仓库对应的已验证环境如下：

- OS: `Windows + WSL2 (Ubuntu 24.04.x)`
- GPU: `2 x RTX 2080 Ti 22GB`
- Python: `3.12`
- PyTorch: `2.10.0+cu129`
- vLLM: `0.19.0`
- 模型: `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- 权重量化: `AWQ / compressed-tensors`
- 推理并行: `--tensor-parallel-size 2`
- 注意力后端: `TRITON_ATTN`

## 当前已验证可用参数

### 1. 基线可用参数

这是最早稳定跑通的基线：

```bash
MODEL=ebircak/gemma-4-31B-it-4bit-W4A16-AWQ
PORT=8001
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
```

特征：

- 双卡张量并行已开启
- 上下文为 `8192`
- 权重为 `AWQ 4bit`
- KV 仍是默认精度路径
- 没有使用 TurboQuant KV Q4

### 2. TurboQuant KV Q4 已验证参数

这是当前实际成功返回生成结果的一组参数：

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

特征：

- 已启用 `TurboQuant KV Q4`
- `K/V` 都为 `4bit`
- 仍是双卡 `TP=2`
- 仍使用 `TRITON_ATTN`
- 通过收紧 scratch buffer 和显存利用率，为 `Turing` 卡留出推理余量

## 关键结果

当前已经验证过：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

最小生成测试返回：

```json
{
  "choices": [
    {
      "message": {
        "content": "ok"
      }
    }
  ]
}
```

在 `TurboQuant KV Q4` 路线上，GPU KV cache capacity 已提升到大约：

- `37k+ tokens`

相比此前默认 KV 路线的大约：

- `20k` 级别

这说明 Q4 KV 压缩已经带来了真实的缓存容量收益。

## 为什么这个仓库存在

因为这条路线有几个现实问题：

- `Gemma 4` 在 vLLM 中会被强制走 `TRITON_ATTN`
- `RTX 2080 Ti` 是 `Turing / cc 7.5`，不是 `Ampere`
- 现成的 `turboquant-vllm` 主要围绕 `FlashAttention` 设计
- 所以 `Gemma 4 + dual 2080 Ti + vLLM + TurboQuant` 并不是直接加参数就能跑

这次实际做过的关键工作包括：

- 给 `vLLM` 打 `Turing` shared-memory workaround
- 把 `TurboQuant` 后端从 `FlashAttention` 适配到 `TRITON_ATTN`
- 让 `FullAttention` 和 `SlidingWindow` 两类层都接入 `packed KV`
- 修掉生成期的 KV dtype / layout 相关问题
- 收敛 `Turing` 上首次生成的显存边界问题

## 当前限制

虽然已经跑通，但目前仍有这些限制：

- 这是实验性工程方案，不是上游官方支持配置
- 当前更适合 `8K` 级上下文，不建议一开始就追求极长上下文
- Windows 到 WSL 的 API 直连，可能还会受 `Hyper-V / WSL 防火墙` 影响
- 这条路线对 `2080 Ti` 的显存边界比较敏感，参数不能太激进
- 当前仓库更偏部署实践和补丁记录，不是通用一键安装器

## 推荐阅读顺序

如果你是第一次接触这个仓库，建议按这个顺序看：

1. 先看 README，了解整体结论和参数边界
2. 再看部署文档，复现 `Gemma 4 AWQ` 的基线服务
3. 然后看 `TurboQuant KV Q4` 的实验记录
4. 最后再看补丁细节和具体脚本

## 仓库计划

后续会逐步补齐：

- `README` 首页整理
- `docs/` 中文部署文档
- `docs/en/` 英文部署文档
- 启动脚本示例
- Patch 说明
- 常见报错排查

## License

本仓库使用 `Apache-2.0`。

原因：

- 更适合这类工程实现和补丁仓库
- 宽松，便于复用和二次开发
- 相比 `MIT`，专利授权条款更完整

## 致谢

这个仓库站在很多优秀开源项目之上，尤其是：

- `vLLM`
- `transformers`
- `turboquant-vllm`
- `PyTorch`

---

## English

Practical deployment notes for running `Gemma 4 31B AWQ` on dual `RTX 2080 Ti` under `WSL2 + vLLM`, and extending the stack to a working `TurboQuant KV-cache Q4` prototype.

This repository focuses on real-world engineering work rather than theory. The goal is to make the following stack actually work:

- `Gemma 4 31B AWQ`
- `vLLM 0.19.0`
- `WSL2 / Ubuntu 24.04`
- `2 x RTX 2080 Ti`
- `Tensor Parallel = 2`
- `TRITON_ATTN`
- `TurboQuant KV-cache Q4`

### Status

What has already been validated:

- A working baseline for `Gemma 4 31B AWQ + vLLM + WSL2 + TP2`
- A working `TurboQuant KV Q4` route on top of `TRITON_ATTN`
- Real generation succeeded, not just engine startup
- Turing-specific workarounds are required on `RTX 2080 Ti`

### Verified Setup

- OS: `Windows + WSL2 (Ubuntu 24.04.x)`
- GPU: `2 x RTX 2080 Ti 22GB`
- Python: `3.12`
- PyTorch: `2.10.0+cu129`
- vLLM: `0.19.0`
- Model: `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- Weight quantization: `AWQ / compressed-tensors`
- Parallelism: `--tensor-parallel-size 2`
- Attention backend: `TRITON_ATTN`

### Working TurboQuant KV Q4 Parameters

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

### Key Outcome

This route now supports:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The KV cache capacity with TurboQuant Q4 has increased to roughly `37k+ tokens`, which is materially higher than the earlier default-KV path.

### Why This Is Non-trivial

This is not a simple “enable one flag” setup because:

- `Gemma 4` is forced onto `TRITON_ATTN` in vLLM
- `RTX 2080 Ti` is `Turing / cc 7.5`
- upstream `turboquant-vllm` is primarily built around `FlashAttention`

As a result, this repository documents the engineering needed to bridge that gap.
