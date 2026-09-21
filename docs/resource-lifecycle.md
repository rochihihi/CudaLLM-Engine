# 资源生命周期说明

## 1. 模型初始化

```text
Model 构造
→ 读取模型 Header
→ mmap 权重文件
→ 创建 Tensor / Layer
→ CUDA 权重拷贝
→ 创建 CUDA Stream
→ 创建 ArgmaxSampler workspace
→ 分配 KV Cache 和中间 Buffer
```

`RawModelData` 使用 `mmap` 映射模型文件，避免先把整个文件复制到另一块 CPU 缓冲。Layer 创建时，权重 Tensor 指向映射区域；调用 `to_cuda()` 后，CUDA 侧会持有可计算的 GPU 权重。

## 2. Tensor 与 Buffer

```text
Tensor
├── shape / dtype / device_type
└── Buffer
    ├── CPU：malloc 管理
    └── CUDA：CUDADeviceAllocator 管理
```

Tensor 描述形状和数据类型，Buffer 负责实际内存。CUDA Buffer 由 allocator 统一分配和回收，避免每个算子自行管理显存。

## 3. Decode 中的持久资源

以下资源在一次模型生命周期内复用：

- `KeyCache` / `ValueCache`：保存历史 Token 的 K/V；
- 中间 Tensor：Q、Attention 输出、FFN 输出和 logits；
- CUDA Stream：串联同一请求内的异步 Kernel；
- Argmax workspace：初始化时分配一个 `size_t` 输出缓冲，每个 Token 重复使用；
- Q8 scale：随量化权重加载，供 INT8 MatMul 反量化累加。

## 4. 单个 Token 的生命周期

```text
当前 Token Embedding
→ RMSNorm / QKV / RoPE
→ 写入 KV Cache
→ Attention / FFN
→ logits
→ Argmax workspace
→ D2H 得到 Token ID
→ 下一轮 Embedding
```

Argmax workspace 的复用避免了原先每轮 `cudaMalloc`。Stream D2H 拷贝完成后才读取 CPU 侧 Token ID，保证异步操作的可见性。

## 5. 释放顺序

```text
销毁 Model
→ 销毁 Sampler，归还 Argmax workspace
→ 销毁 CUDA Stream
→ 释放 GPU Tensor / Buffer
→ 解除 mmap 权重映射
```

Sampler 基类具有虚析构函数，确保通过 `unique_ptr<Sampler>` 析构 `ArgmaxSampler` 时能正确释放 workspace。

## 6. Q8_0 文件布局

Q8_0 模型包含：

```text
Header + group_size
→ 各 MatMul 的 INT8 权重
→ 各 MatMul 的 FP32 group scales
→ FP32 Embedding
→ FP32 Attention RMSNorm
→ FP32 FFN RMSNorm
→ FP32 Final RMSNorm
```

Embedding、Norm 和 KV Cache 保持 FP32，是为了控制改造范围并保持与 FP32 输出一致；当前主要压缩和优化 MatMul 权重读取。
