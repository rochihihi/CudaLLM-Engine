# Argmax Workspace 复用优化

## 问题

原始 Argmax CUDA 路径每生成一个 Token 都会：

1. 通过 CUDA allocator 申请一个 `size_t` 输出缓冲；
2. 启动 Argmax Kernel；
3. 将结果拷贝回 CPU；
4. 不归还申请的缓冲。

这会在 Decode 循环中重复调用 `cudaMalloc`，并持续把小缓冲标记为占用。

此外，原 Kernel 在 `size < 512` 时让部分线程在 block reduction 前提前返回，存在同步风险。

## 修改

- `ArgmaxSampler` 初始化时一次性分配 8 字节 GPU workspace；
- 每次采样复用同一 workspace；
- Sampler 析构时将 workspace 归还给 CUDA allocator；
- 为基类补充虚析构函数，保证通过基类指针正确析构；
- 异步 D2H 拷贝后显式同步对应 Stream，保证返回的 Token ID 已经可用；
- 无效线程使用 `SIZE_MAX` 参与归约，不再提前退出；
- 保持并列最大值选择较小索引的行为。

## Nsight Systems 结果

同一模型、同一提示词、生成 124 Tokens：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| `cudaMalloc` 调用次数 | 249 | 126 | -123 |
| 显式 `cudaStreamSynchronize` | 0 | 124 | +124 |
| Stream 同步累计时间 | - | 0.510 ms | 每 Token 约 0.004 ms |

减少的 123 次分配与生成阶段需要采样的 Token 数量相符。剩余 126 次 `cudaMalloc` 主要来自模型初始化和其他持久缓冲。

## 独立 Benchmark

测试条件：词表大小 32000，20 次预热，每轮重复 256 次。

三轮独立进程结果：

| 轮次 | 每次动态分配 | Workspace 复用 | 加速比 |
|---:|---:|---:|---:|
| 1 | 0.102654 ms | 0.089238 ms | 1.15× |
| 2 | 0.088818 ms | 0.083763 ms | 1.06× |
| 3 | 0.101060 ms | 0.084467 ms | 1.20× |

两种实现均返回正确索引 12345。独立算子加速范围为 1.06–1.20×。

## 正确性

新增测试使用 17 个元素覆盖不足一个 CUDA Block 的输入，并设置两个相同最大值。连续采样 256 次均返回较小索引 3。相关 MatMul、RMSNorm、SwiGLU 与 Argmax 共 9 个 CUDA 测试通过。

## 结论

该修改确定性地消除了 Decode 热路径中的逐 Token 动态分配，并修复了 workspace 泄漏和小输入归约风险。由于单次节省只有几十微秒、笔记本 GPU 频率存在波动，目前不宣称端到端吞吐有稳定提升；主要收益是更合理的资源生命周期和更稳定的 Decode 路径。
