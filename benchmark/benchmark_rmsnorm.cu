#include <algorithm>
#include <cmath>
#include <cstdio>

#include <base/buffer.h>
#include <cuda_runtime_api.h>
#include <op/kernels/cuda/rmsnorm_kernel.cuh>
#include <tensor/tensor.h>

namespace {
using Kernel = void (*)(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, void*);
float measure(Kernel fn, const tensor::Tensor& input, const tensor::Tensor& weight,
              const tensor::Tensor& output, cudaStream_t stream, int warmups, int repeats) {
  for (int i = 0; i < warmups; ++i) fn(input, weight, output, stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < repeats; ++i) fn(input, weight, output, stream);
  cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
  float total = 0.0f;
  cudaEventElapsedTime(&total, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return total / repeats;
}
void run_case(int size) {
  auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto gpu = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, cpu);
  tensor::Tensor w_cpu(base::DataType::kDataTypeFp32, size, true, cpu);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = static_cast<float>((i % 23) - 11) * 0.03125f;
    w_cpu.index<float>(i) = static_cast<float>((i % 19) - 9) * 0.0625f;
  }
  tensor::Tensor in = in_cpu.clone(), w = w_cpu.clone();
  tensor::Tensor baseline(base::DataType::kDataTypeFp32, size, true, gpu);
  tensor::Tensor optimized(base::DataType::kDataTypeFp32, size, true, gpu);
  cudaStream_t stream; cudaStreamCreate(&stream);
  in.to_cuda(stream); w.to_cuda(stream);
  const float base_ms = measure(kernel::rmsnorm_kernel_cu_baseline, in, w, baseline, stream, 20, 200);
  const float opt_ms = measure(kernel::rmsnorm_kernel_cu_optimized, in, w, optimized, stream, 20, 200);
  baseline.to_cpu(); optimized.to_cpu();
  float max_error = 0.0f;
  for (int i = 0; i < size; ++i) max_error = std::max(max_error, std::abs(baseline.index<float>(i) - optimized.index<float>(i)));
  printf("size=%-8d baseline=%8.4f ms optimized=%8.4f ms speedup=%5.2fx max_abs_error=%.8f\n",
         size, base_ms, opt_ms, base_ms / opt_ms, max_error);
  cudaStreamDestroy(stream);
}
}
int main() {
  printf("RMSNorm CUDA Benchmark (warmups=20, repeats=200)\n");
  run_case(32); run_case(768); run_case(2048); run_case(11008);
  return 0;
}
