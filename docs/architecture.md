# 推理引擎架构与数据流

## 1. 端到端推理流程

```mermaid
flowchart LR
    A[Prompt 文本] --> B[SentencePiece Tokenizer]
    B --> C[Token IDs]
    C --> D[Embedding]
    D --> E[Transformer 前向]
    E --> F[Logits]
    F --> G[Argmax Sampler]
    G --> H[下一个 Token]
    H --> I{达到结束条件?}
    I -- 否 --> D
    I -- 是 --> J[Decode 文本]
```

程序入口是 `demo/main.cpp`。每次生成会输出生成 Token 数、TTFT、Decode 延迟、总耗时和 tokens/s。

## 2. Transformer 单层

```mermaid
flowchart TD
    X[输入 hidden state] --> N1[Attention RMSNorm]
    N1 --> QKV[Wq / Wk / Wv MatMul]
    QKV --> R[RoPE]
    R --> KVC[写入 KV Cache]
    KVC --> MHA[Multi-Head Attention]
    MHA --> WO[Wo MatMul]
    X --> ADD1[Residual Add]
    WO --> ADD1
    ADD1 --> N2[FFN RMSNorm]
    N2 --> W1[W1 MatMul]
    N2 --> W3[W3 MatMul]
    W1 --> SG[SwiGLU]
    W3 --> SG
    SG --> W2[W2 MatMul]
    ADD1 --> ADD2[Residual Add]
    W2 --> ADD2
    ADD2 --> OUT[层输出]
```

Decode 时，历史 Token 的 K/V 保存在 KV Cache；当前 Token 只追加新的 K/V，避免重新计算历史状态。

## 3. FP32 与 Q8_0 权重路径

```mermaid
flowchart LR
    W[模型权重] --> SWITCH{模型格式}
    SWITCH -->|FP32| F[FP32 权重]
    SWITCH -->|Q8_0 + --quant| Q[INT8 权重 + group scale]
    F --> FM[FP32 MatMul]
    Q --> QM[INT8 weight-only MatMul]
    QM --> DQ[按 group scale 反量化累加]
    FM --> Y[FP32 激活输出]
    DQ --> Y
```

Q8_0 转换使用 group-size 64：

```text
scale = max(abs(weight_group)) / 127
quant = round(weight_group / scale)
```

Q8_0 只压缩 MatMul 权重；Embedding、RMSNorm 和 KV Cache 仍使用 FP32。

## 4. CUDA 与资源生命周期

```mermaid
sequenceDiagram
    participant M as Model::init
    participant A as CUDA Allocator
    participant K as CUDA Kernel
    participant S as ArgmaxSampler
    M->>A: 分配模型 Buffer / KV Cache
    M->>S: 创建 Sampler
    S->>A: 一次分配 Argmax workspace
    loop 每个生成 Token
        M->>K: Transformer CUDA kernels
        M->>S: 读取 logits
        S->>K: Argmax 写入 workspace
        K-->>S: Stream D2H 返回 Token ID
    end
    M->>S: 析构 Sampler
    S->>A: 释放 workspace
    M->>A: 释放模型 Buffer
```

Argmax workspace 复用避免了 Decode 热路径中的逐 Token `cudaMalloc`。

## 5. 性能分析闭环

```mermaid
flowchart LR
    B[端到端 Benchmark] --> P[CUDA Event Profiling]
    P --> L[定位阶段/算子瓶颈]
    L --> E[实现候选优化]
    E --> C[CPU/CUDA 正确性测试]
    C --> R[多轮 Kernel Benchmark]
    R --> A[端到端复测]
    A --> D{收益稳定?}
    D -- 是 --> S[保留并记录]
    D -- 否 --> F[记录失败实验，不替换默认路径]
```

当前实测主优化是 Q8_0 weight-only quantization；SwiGLU、RMSNorm 和 MatMul 的候选 Kernel 也保留为可复现实验，但没有因为微小或不稳定的差异而替换默认实现。
