# Q8_0 Weight-only 推理优化

## 动机

前面的 CUDA Event 分析显示，FFN 占模型阶段耗时约 50%，其中 W1/W3/W2 矩阵乘占 FFN 的 87.30%。这些 Decode GEMV 主要受权重读取带宽影响，因此将 MatMul 权重从 FP32 压缩为 INT8，并在 Kernel 中按 group scale 反量化，是比继续微调线程归约更直接的优化路径。

## 实现

- 新增 `tools/quantize_legacy_model`，不依赖 Python/NumPy，直接读取现有 legacy FP32 `.bin`；
- 使用 group-size 64 的对称 Q8_0：`scale = max(abs(group)) / 127`；
- 写出 INT8 权重和每组 FP32 scale；
- 保留 Embedding、RMSNorm 和 KV Cache 为 FP32；
- Demo 新增 `--quant` 开关；
- 修复共享 Embedding/分类权重路径：量化文件中的共享分类权重仍使用 FP32 Embedding，不误当作 INT8；
- 新增 INT8 MatMul 正确性测试。

## 模型文件变化

测试模型：TinyStories 110M，`dim=768`、`hidden_dim=2048`、12 层、32000 词表。

| 格式 | 文件大小 |
|---|---:|
| FP32 | 438,381,596 bytes（约 418.1 MiB） |
| Q8_0 | 188,623,904 bytes（约 179.9 MiB） |

文件大小减少 **56.97%**，约为 FP32 的 2.32 倍压缩；量化过程报告的最大绝对权重误差为 `0.00758666`。

## 端到端结果

测试条件：RTX 3050 Laptop 4 GiB、WSL2、同一提示词 `Once upon a time`、生成 124 Tokens；FP32 和 Q8 交替运行 5 轮。

| 指标 | FP32 范围 | Q8 范围 | 代表性提升 |
|---|---:|---:|---:|
| TTFT | 16.69–18.53 ms | 13.10–14.43 ms | 约 21.28% |
| Decode 延迟 | 3.47–3.62 ms/token | 2.78–2.88 ms/token | 约 20.49% |
| 总耗时 | 444.03–463.38 ms | 355.05–368.40 ms | 约 20.52% |
| 吞吐 | 257.69–272.77 tokens/s | 319.60–344.86 tokens/s | 约 26.85% |

同一方法轮询 GPU 总显存时，进程增量峰值为：

```text
FP32: 758 MiB
Q8:   430 MiB
```

增量峰值减少约 **43.27%**。

## 正确性验证

- INT8 MatMul 与 CPU 参考结果误差不超过 `1e-4`；
- 10 个 Argmax、MatMul、RMSNorm、SwiGLU CUDA 测试全部通过；
- `hello`、`Once upon a time`、`The little boy` 三个提示词下，FP32/Q8 完整生成文本 SHA-256 完全一致；
- Q8 模型 CUDA Event 分阶段统计仍显示 FFN 是最大阶段，说明收益来自权重读取/量化 GEMV，而不是改变模型结构。

## 结论

这是当前项目第一个有稳定端到端收益的优化：在不改变模型层结构和贪心输出的前提下，Q8_0 weight-only 将 Decode 吞吐提高约 26.85%，同时降低模型文件和 GPU 权重占用。后续可继续研究 INT8 Kernel 的向量化加载与融合，但必须以同样的正确性和多轮端到端测试为准。
