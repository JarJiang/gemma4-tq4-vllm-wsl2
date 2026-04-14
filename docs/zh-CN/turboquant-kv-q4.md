# TurboQuant KV Q4 说明

本文档描述的是这次已经推进到“可真实生成”的实验路线，而不是上游官方现成能力。

## 当前已验证参数

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

保持不变的部分：

- 模型仍为 `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`
- 并行仍为 `TP=2`
- 注意力后端仍为 `TRITON_ATTN`

## 这次实际做了什么

核心不是“开一个插件开关”，而是做了几类适配：

- 给 `vLLM` 加了面向 `Turing` 的 attention workaround
- 把 `turboquant-vllm` 从偏 `FlashAttention` 的实现，接到了 `TRITON_ATTN`
- 让 `FullAttention` 和 `SlidingWindow` 两类层都能走 packed KV
- 修掉生成期 KV dtype / cache layout 问题
- 收紧运行期 scratch buffer，给 `2080 Ti` 留出推理余量

## 当前结果

已经验证：

- `GET /v1/models` 正常
- `POST /v1/chat/completions` 正常
- 最小回复可返回 `ok`

KV cache 容量也明显提升到了约：

- `37k+ tokens`

相比默认 KV 路线的大约：

- `20k` 级别

## 当前边界

这条路虽然已经可用，但仍然是 PoC 级工程实践：

- 参数比较敏感，不能无脑拉满显存利用率
- `2080 Ti` 上建议先用 `8K` 级上下文
- 这不是上游 vLLM 官方原生支持的 Gemma 4 TurboQuant 路线

## 什么时候不建议直接启用

如果你还没完成下面这些前提，就别先开 `TurboQuant KV Q4`：

- 基线服务尚未稳定
- WSL 网络和代理还不稳定
- 你还没确认双卡 TP2 已经正常
- 你还没验证最小生成请求可以工作

## 推荐理解方式

可以把当前状态理解成：

- `Gemma 4 AWQ`：是生产可跑的基础
- `TurboQuant KV Q4`：是在这个基础上跑通的实验增强路径

它已经不是“纯研究概念”，但也还没有到“一键生产模板”的程度。
