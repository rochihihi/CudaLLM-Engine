# CudaLLM-Engine：C++/CUDA 大模型推理与量化优化引擎

## 项目简介

CudaLLM-Engine 是我自主开发并持续迭代的 C++/CUDA 大模型推理项目，支持 Llama/Qwen 模型加载、Transformer 前向、KV Cache、量化推理和端到端性能分析。

```text
本地模型加载 → Tokenizer / Embedding → Transformer 前向与 KV Cache
→ CUDA Event Profiling → Q8_0 weight-only quantization
→ 正确性与端到端性能复测
```

### 我的工作

- 将固定 Prompt 改为命令行输入，并输出 Prompt Tokens、生成 Tokens、TTFT、Decode 延迟和 tokens/s；
- 增加 CUDA Event 分阶段 Profiling，定位 FFN、MatMul 和输出分类层的耗时占比；
- 实现纯 C++ 的 Q8_0 weight-only 转换工具，按 group-size 64 保存 INT8 权重和 FP32 scale；
- 增加 INT8 MatMul、Argmax workspace、SwiGLU 和 RMSNorm 的 CUDA 测试与 Benchmark；
- 对优化候选做多轮基准测试，只保留有数据支持的结论。

项目工作重点：性能测量、量化路径、资源复用、正确性验证和实验分析。

### Q8_0 实测结果

测试模型为 TinyStories 110M，GPU 为 RTX 3050 Laptop 4 GiB，FP32/Q8 交替运行 5 轮：

| 指标 | FP32 | Q8_0 | 变化 |
|---|---:|---:|---:|
| 模型文件 | 418.1 MiB | 179.9 MiB | -56.97% |
| Decode 延迟 | 3.47–3.62 ms/token | 2.78–2.88 ms/token | 约 -20.49% |
| 吞吐 | 257.69–272.77 tokens/s | 319.60–344.86 tokens/s | 约 +26.85% |
| GPU 增量峰值 | 758 MiB | 430 MiB | -43.27% |

三个 Prompt 的 FP32/Q8 完整生成文本 SHA-256 一致，相关 CUDA 正确性测试全部通过。

Qwen2.5-0.5B 也已完成中文 FP32/Q8 验证：Q8 平均吞吐提升约 30.37%，短请求的进程显存增量由 2798 MiB 降至 2481 MiB，减少约 11.33%。详见 [Qwen2.5 推理验证](docs/qwen2.5-inference.md)。

### 快速运行

```shell
cmake -S . -B build
cmake --build build -j2

./build/demo/llm_infer models/stories110M.bin models/tokenizer.model \
  "Once upon a time"

./build/tools/quantize_legacy_model models/stories110M.bin \
  models/stories110M-q8.bin 64

./build/demo/llm_infer models/stories110M-q8.bin models/tokenizer.model \
  --quant "Once upon a time"
```

### 性能与实验文档

- [Q8_0 量化优化报告](docs/int8-inference-optimization.md)
- [推理性能基线与瓶颈分析](docs/performance-baseline.md)
- [Argmax workspace 复用](docs/argmax-workspace-optimization.md)
- [MatMul 优化实验](docs/matmul-optimization-experiment.md)
- [SwiGLU 优化实验](docs/swiglu-optimization-experiment.md)
- [RMSNorm 优化实验](docs/rmsnorm-optimization-experiment.md)
- [架构与数据流图](docs/architecture.md)
- [资源生命周期说明](docs/resource-lifecycle.md)

## 当前实测效果

> TinyStories 110M 在当前 RTX 3050 Laptop WSL 环境中进行 5 轮交替测试：FP32 平均 265.397 tokens/s，Q8 平均 336.644 tokens/s，吞吐提升约 26.85%。

## 第三方依赖
> 借助企业级开发库，更快地搭建出大模型推理框架
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html
5. Cuda Toolkit


## 模型下载地址
1. LLama2 https://pan.baidu.com/s/1PF5KqvIvNFR8yDIY1HmTYA?pwd=ma8r 或 https://huggingface.co/fushenshen/lession_model/tree/main

2. Tiny LLama 
- TinyLLama模型 https://huggingface.co/karpathy/tinyllamas/tree/main
- TinyLLama分词器 https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model

3. Qwen2.5/LLama
   
   下载与导出方法见下方《Qwen2.5 推理》章节。


## 模型导出
```shell
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
# 使用--hf标签从hugging face中加载模型， 指定--version3可以导出量化模型
# 其他使用方法请看export.py中的命令行参数实例
```


## 编译方法
```shell
  mkdir build 
  cd build
  # 需要安装上述的第三方依赖
  cmake ..
  # 或者开启 USE_CPM 选项，自动下载第三方依赖
  cmake -DUSE_CPM=ON ..
  make -j16
```

## 生成文本的方法
```shell
./build/demo/llm_infer llama2_7b.bin tokenizer.model "Once upon a time"

```

第三个参数为可选提示词；省略时默认使用 `hello`。

程序生成文本后会输出以下端到端性能指标：

- `prompt_tokens`：提示词的 Token 数量；
- `generated_tokens`：实际生成的 Token 数量；
- `ttft_ms`：从开始处理请求到生成首个 Token 的时间；
- `avg_decode_latency_ms`：首个 Token 之后，每个 Token 的平均生成延迟；
- `total_latency_ms`：本次请求的总耗时；
- `tokens_per_second`：生成 Token 数除以请求总耗时。

如需使用 CUDA Event 统计模型各阶段的 GPU 时间，可以增加 `--profile`：

```shell
./build/demo/llm_infer llama2_7b.bin tokenizer.model --profile "Once upon a time"
```

Profiling 模式会同步每个模型阶段，只用于定位耗时占比，不用于测量正常运行吞吐。

矩阵向量乘实验可以独立运行：

```shell
./build/benchmark/benchmark_matmul
```

该程序使用 CUDA Event，对比原始 Kernel、实验 Kernel 和 cuBLAS SGEMV，并检查输出误差。

Argmax workspace 复用实验：

```shell
./build/benchmark/benchmark_argmax
```

该程序对比每次调用 `cudaMalloc` 与预分配 workspace 两种实现的延迟和输出索引。

SwiGLU 算子实验：

```shell
./build/benchmark/benchmark_swiglu
```

该程序对比原始 shared-memory Kernel 与寄存器直算实验版本；实验版本不会自动替换默认推理路径。

RMSNorm 算子实验：

```shell
./build/benchmark/benchmark_rmsnorm
```

该程序对比 CUB BlockReduce 与 warp-shuffle reduction 版本，并检查输出误差。

统一运行 FP32/Q8_0 多轮 Benchmark 并生成 Markdown 表格：

```shell
./benchmark/run_inference_benchmark.sh \
  models/stories110M.bin \
  models/tokenizer.model \
  models/stories110M-q8.bin \
  5
```

自动比较 FP32/Q8_0 的完整生成文本：

```shell
./benchmark/compare_inference_outputs.sh \
  models/stories110M.bin \
  models/tokenizer.model \
  models/stories110M-q8.bin
```

## Q8_0 weight-only 推理

可以把现有 FP32 Llama 二进制转换为 Q8_0 权重量化格式：

```shell
./build/tools/quantize_legacy_model \
  models/stories110M.bin \
  models/stories110M-q8.bin \
  64
```

然后使用 `--quant` 运行：

```shell
./build/demo/llm_infer \
  models/stories110M-q8.bin \
  models/tokenizer.model \
  --quant "Once upon a time"
```

Q8_0 只量化 MatMul 权重，RMSNorm、Embedding 和 KV Cache 保持 FP32。
`--quant` 必须与转换后的模型文件配套使用。

# LLama3.2 推理

- 以 meta-llama/Llama-3.2-1B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/llm_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/llama3_infer.py
```

# Qwen2.5 推理

当前已验证 Qwen2.5-0.5B 的中文 FP32/Q8 推理，详见 [Qwen2.5 推理验证](docs/qwen2.5-inference.md)。

- 以 Qwen2.5-0.5B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/qwen2_infer.py
```

## Qwen3推理
和上面同理，我们先从huggingface仓库中将模型下载到本地。
1. tools/export_qwen3/load.py中导出为pth，模型的输入`model_name`和输出地址`output_file`依次需要填写；
2. 导出pth格式的模型后，再用同文件夹下的write_bin.py导出qwen.bin；
3. 用CMake选项`QWEN3_SUPPORT`重新编译项目，其他步骤就都是一样的了。
