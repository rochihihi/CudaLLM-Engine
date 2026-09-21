#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <base/buffer.h>
#include <cuda_runtime_api.h>
#include <op/kernels/cuda/swiglu_kernel.cuh>
#include <tensor/tensor.h>

namespace {
using SwigluKernel = void (*)(const tensor::Tensor&, const tensor::Tensor&,
                              const tensor::Tensor&, void*);

struct Result {
  float latency_ms;
  float max_error;
};

Result measure(SwigluKernel fn, const tensor::Tensor& input1, const tensor::Tensor& input2,
              const tensor::Tensor& output, cudaStream_t stream, int warmups, int repeats) {
  for (int i = 0; i < warmups; ++i) fn(input1, input2, output, stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < repeats; ++i) fn(input1, input2, output, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float total = 0.0f;
  cudaEventElapsedTime(&total, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return {total / repeats, 0.0f};
}

void run_case(int size) {
  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto gpu = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor a_cpu(base::DataType::kDataTypeFp32, size, true, cpu);
  tensor::Tensor b_cpu(base::DataType::kDataTypeFp32, size, true, cpu);
  for (int i = 0; i < size; ++i) {
    a_cpu.index<float>(i) = static_cast<float>((i % 31) - 15) * 0.125f;
    b_cpu.index<float>(i) = static_cast<float>((i % 17) - 8) * 0.0625f;
  }
  tensor::Tensor a = a_cpu.clone();
  tensor::Tensor b = b_cpu.clone();
  tensor::Tensor baseline(base::DataType::kDataTypeFp32, size, true, gpu);
  tensor::Tensor optimized(base::DataType::kDataTypeFp32, size, true, gpu);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  a.to_cuda(stream);
  b.to_cuda(stream);
  constexpr int warmups = 20;
  constexpr int repeats = 200;
  Result baseline_result = measure(kernel::swiglu_kernel_cu_baseline, a, b, baseline, stream,
                                   warmups, repeats);
  Result optimized_result = measure(kernel::swiglu_kernel_cu_optimized, a, b, optimized, stream,
                                    warmups, repeats);
  baseline.to_cpu();
  optimized.to_cpu();
  float max_error = 0.0f;
  for (int i = 0; i < size; ++i) {
    max_error = std::max(max_error, std::abs(baseline.index<float>(i) - optimized.index<float>(i)));
  }
  printf("size=%-8d baseline=%8.4f ms optimized=%8.4f ms speedup=%5.2fx max_abs_error=%.8f\n",
         size, baseline_result.latency_ms, optimized_result.latency_ms,
         baseline_result.latency_ms / optimized_result.latency_ms, max_error);
  cudaStreamDestroy(stream);
}
}  // namespace

int main() {
  printf("SwiGLU CUDA Benchmark (warmups=20, repeats=200)\n");
  run_case(128);
  run_case(2048);
  run_case(11008);
  run_case(131072);
  return 0;
}
