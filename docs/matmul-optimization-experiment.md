# MatMul / GEMV 优化实验

## 背景

TinyStories 110M 的 Decode 阶段实际执行矩阵向量乘。模型维度为：

- `dim = 768`
- `hidden_dim = 2048`
- `vocab_size = 32000`

分阶段 Profiling 显示，W1、W3、W2 矩阵乘合计占 FFN 时间的 87.30%，因此优先检查 MatMul Kernel。

## 对比实现

1. `baseline`：一个 128 线程 CUDA Block 计算一个输出行，使用 CUB BlockReduce。
2. `warp-per-row`：一个 Warp 计算一个输出行，一个 Block 同时处理 8 行，使用 Warp Shuffle 归约。
3. `cuBLAS`：使用 `cublasSgemv` 作为厂商库参考。

测试采用 20 次预热、100 次重复、7 轮交替顺序测试，并报告中位数。

## 算子结果

测试设备：NVIDIA GeForce RTX 3050 Laptop GPU，FP32。

一次代表性进程内结果如下：

| 模型位置 | K × M | Baseline | Warp-per-row | 加速比 | cuBLAS | cuBLAS/Baseline |
|---|---:|---:|---:|---:|---:|---:|
| QKV / Wo | 768 × 768 | 0.0163 ms | 0.0161 ms | 1.01× | 0.0280 ms | 0.58× |
| W1 / W3 | 2048 × 768 | 0.0378 ms | 0.0372 ms | 1.02× | 0.0565 ms | 0.67× |
| W2 | 768 × 2048 | 0.0370 ms | 0.0370 ms | 1.00× | 0.0423 ms | 0.87× |
| 分类层 | 32000 × 768 | 0.6383 ms | 0.6249 ms | 1.02× | 0.6060 ms | 1.05× |

三种实现的测试输出最大绝对误差均为 0。实验 Kernel 的有效显存带宽约为 147–170 GB/s。

重新启动进程复测后，W1/W3 的实验版本从 `1.02×` 变为 `0.96×`。W2 的 baseline 与实验入口实际分派到同一 Kernel，但仍测出 `0.72×` 的假差异。这说明笔记本 GPU 动态频率带来的跨测量波动已经大于预期的 1–2% 微优化收益。在无法锁定 GPU 时钟的当前 WSL/WDDM 环境中，不能把这些差异作为有效加速结论。

## 端到端验证

尝试只对 `K >= 8192` 的分类层启用 cuBLAS。三个交替轮次显示：

- Baseline TTFT：17.3–19.6 ms；
- cuBLAS 混合版本 TTFT：124.8–156.9 ms；
- Baseline Decode：3.51–3.83 ms/token；
- cuBLAS 混合版本 Decode：3.65–3.90 ms/token。

cuBLAS 首次调用的懒初始化显著恶化 TTFT，而 Decode 收益不足以抵消这一代价。因此没有将实验实现替换为默认推理路径。

## 结论

1. 当前 FP32 GEMV 已明显受显存带宽限制，仅替换归约方式没有得到超出测量噪声的稳定收益。
2. cuBLAS 在一次测试中对大词表分类层快约 5%，但复测无法稳定重现，并且首次调用开销不适合当前低延迟场景。
3. 默认推理继续使用原 Kernel；保留独立 Benchmark 和实验 Kernel，确保失败实验可以复现。
4. Nsight Systems 的 CUDA API 统计还显示生成过程中存在反复 `cudaMalloc`。下一步应消除 Argmax 每 Token 的动态显存分配，这比继续微调 GEMV 更可能改善端到端延迟。
