# Qwen2.5-0.5B 推理验证

## 模型与构建

- 模型：Qwen2.5-0.5B
- hidden size：896
- Transformer 层数：24
- 词表大小：151936
- 构建：`QWEN2_SUPPORT=ON`、`USE_CPM=ON`
- Tokenizer：官方 `tokenizer.json`

```shell
cmake -S . -B build-qwen -DQWEN2_SUPPORT=ON -DUSE_CPM=ON
cmake --build build-qwen -j2 --target qwen_infer
```

## FP32 与 Q8_0 推理

```shell
./build-qwen/demo/qwen_infer models/Qwen2.5-0.5B.bin \
  models/Qwen2.5-0.5B/tokenizer.json "你好，我是一名人工智能工程师"
```

Q8 导出与运行：

```shell
.runtime/qwen-export-venv/bin/python tools/export_qwen2.py \
  models/Qwen2.5-0.5B-q8.bin --hf=models/Qwen2.5-0.5B --version=3
./build-qwen/demo/qwen_infer models/Qwen2.5-0.5B-q8.bin \
  models/Qwen2.5-0.5B/tokenizer.json --quant "你好，我是一名人工智能工程师"
```

中文 FP32/Q8 输出一致，FP32 单次约 23.93 tokens/s。

## 三轮端到端结果

| 指标 | FP32 | Q8_0 |
|---|---:|---:|
| TTFT | 201.810 ms | 176.232 ms |
| Decode | 13.852 ms/token | 10.448 ms/token |
| 总耗时 | 1877.904 ms | 1440.456 ms |
| 吞吐 | 64.967 tokens/s | 84.701 tokens/s |

Q8 平均吞吐提升约 **30.37%**。FP32 文件约 1.9 GiB，Q8 文件约 883 MiB。

## 显存对比

测试使用 Prompt `你好`，两种格式均生成 100 Tokens。通过 `nvidia-smi` 采样推理前显存和运行期间峰值：

| 格式 | 推理前显存 | 峰值显存 | 进程增量 |
|---|---:|---:|---:|
| FP32 | 1086 MiB | 3884 MiB | 2798 MiB |
| Q8_0 | 474 MiB | 2955 MiB | 2481 MiB |

Q8 的进程显存增量减少约 **11.33%**，同次测试吞吐从约 57.54 tokens/s 提升到 81.99 tokens/s。

该设备为 Windows/WSL 共享 GPU，推理前总显存会受桌面和其他程序影响，因此主要比较“峰值减去推理前”的进程增量，不把总显存直接解释为模型纯显存。

以下 Prompt 的 FP32/Q8 完整输出 SHA-256 一致：`你好，我是一名人工智能工程师`、`请用一句话介绍北京`、`Once upon a time`。

## 工程修复

- 恢复 Qwen BPE Tokenizer 创建逻辑；
- Qwen Demo 支持中文 Prompt、`--quant` 和统一 Benchmark 指标；
- 修复 Qwen 量化导出中 Q/K/V bias 未写入的问题；
- 修复量化 MatMul bias 的 FP32 读取路径；
- 共享 Embedding/分类权重按 FP32 路径处理；
- 不同 CMake 构建目录使用各自的 `lib` 输出，避免 Llama/Qwen 动态库互相覆盖。
