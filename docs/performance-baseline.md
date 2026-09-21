# 推理性能基线与瓶颈分析

## 测试环境

- 日期：2026-09-20
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB
- Windows NVIDIA 驱动：527.99
- WSL2：Ubuntu
- 编译 CUDA Toolkit：12.8
- 模型：TinyStories 110M，FP32
- 提示词：`Once upon a time`（5 Tokens）
- 最大序列步数：128

## 五轮交替测试平均结果

```text
generated_tokens: 124
ttft_ms: 17.882
avg_decode_latency_ms: 3.655
total_latency_ms: 467.447
tokens_per_second: 265.397
```

以上为 5 轮交替测试的 FP32 平均结果。

## CUDA Event 分阶段结果

Profiling 模式会在每个阶段后同步 CUDA Stream，因此不能使用该模式的端到端吞吐与普通模式比较。阶段占比只在已统计的 GPU 模型阶段内部计算。

| 阶段 | 累计 GPU 时间 | 占比 |
|---|---:|---:|
| Attention RMSNorm | 7.246 ms | 1.79% |
| QKV 与 RoPE | 77.127 ms | 19.05% |
| Attention 与 Wo | 40.459 ms | 9.99% |
| FFN | 202.410 ms | 49.99% |
| 输出 Norm 与分类层 | 77.694 ms | 19.19% |

FFN 进一步拆分：

| FFN 阶段 | 累计 GPU 时间 | FFN 内占比 |
|---|---:|---:|
| 两次残差 Add | 12.164 ms | 6.01% |
| FFN RMSNorm | 7.215 ms | 3.56% |
| W1 与 W3 矩阵乘 | 117.877 ms | 58.24% |
| SwiGLU | 6.330 ms | 3.13% |
| W2 矩阵乘 | 58.823 ms | 29.06% |

## 当前结论

1. FFN 是当前最大的模型阶段，约占已统计 GPU 时间的一半。
2. FFN 内的 W1、W3、W2 矩阵乘合计占 87.30%，是主要耗时来源。
3. SwiGLU 只占 FFN 的 3.13%，约占全部已统计 GPU 阶段的 1.56%；单独优化它适合展示 CUDA 算子实验，但预计端到端收益有限。
4. 下一步应先检查矩阵向量乘 Kernel 的访存和线程映射，再决定将矩阵乘优化还是算子融合设为主优化项。

## Nsight 限制

- Nsight Systems 2024.6 可以生成报告，但当前 WSL/驱动组合未记录 GPU Kernel 明细。
- Nsight Compute 2025.1 明确报告当前 WSL 设备不支持硬件指标采集。
- 因此当前阶段使用项目内 CUDA Event 完成分阶段定位；后续若更新 Windows NVIDIA 驱动，再补充 Nsight Compute 的 Occupancy、内存吞吐和 Warp 指标。
