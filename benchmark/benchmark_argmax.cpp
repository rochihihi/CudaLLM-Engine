#include <chrono>
#include <cstdio>
#include <vector>

#include <cuda_runtime_api.h>

#include "base/buffer.h"
#include "op/kernels/cuda/argmax_kernel.cuh"
#include "tensor/tensor.h"

namespace {
struct BenchmarkResult {
  double latency_ms = 0.0;
  size_t output_index = 0;
};

BenchmarkResult benchmark_allocating(const tensor::Tensor& logits, cudaStream_t stream,
                                     int repeats) {
  std::vector<size_t*> allocations;
  allocations.reserve(repeats);
  size_t output_index = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    size_t* device_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(size_t));
    allocations.push_back(device_output);
    output_index =
        kernel::argmax_kernel_cu(logits.ptr<float>(), logits.size(), device_output, stream);
  }
  const auto stop = std::chrono::steady_clock::now();
  for (size_t* allocation : allocations) {
    cudaFree(allocation);
  }
  return {std::chrono::duration<double, std::milli>(stop - start).count() / repeats,
          output_index};
}

BenchmarkResult benchmark_reusing(const tensor::Tensor& logits, cudaStream_t stream,
                                  int repeats) {
  size_t* device_output = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(size_t));
  size_t output_index = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    output_index =
        kernel::argmax_kernel_cu(logits.ptr<float>(), logits.size(), device_output, stream);
  }
  const auto stop = std::chrono::steady_clock::now();
  cudaFree(device_output);
  return {std::chrono::duration<double, std::milli>(stop - start).count() / repeats,
          output_index};
}
}  // namespace

int main() {
  constexpr int size = 32000;
  constexpr int repeats = 256;
  auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor logits_cpu(base::DataType::kDataTypeFp32, size, true, cpu_allocator);
  for (int i = 0; i < size; ++i) {
    logits_cpu.index<float>(i) = -static_cast<float>(i);
  }
  logits_cpu.index<float>(12345) = 100.0f;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  tensor::Tensor logits_cuda = logits_cpu.clone();
  logits_cuda.to_cuda(stream);

  size_t* warmup_output = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&warmup_output), sizeof(size_t));
  for (int i = 0; i < 20; ++i) {
    kernel::argmax_kernel_cu(logits_cuda.ptr<float>(), logits_cuda.size(), warmup_output, stream);
  }
  cudaFree(warmup_output);

  const BenchmarkResult allocating = benchmark_allocating(logits_cuda, stream, repeats);
  const BenchmarkResult reusing = benchmark_reusing(logits_cuda, stream, repeats);
  printf("Argmax CUDA Benchmark (size=%d, repeats=%d)\n", size, repeats);
  printf("  allocating_each_call: %.6f ms\n", allocating.latency_ms);
  printf("  reusing_workspace:    %.6f ms\n", reusing.latency_ms);
  printf("  speedup:              %.2fx\n", allocating.latency_ms / reusing.latency_ms);
  printf("  output_indices:       %zu / %zu\n", allocating.output_index,
         reusing.output_index);

  cudaStreamDestroy(stream);
  return allocating.output_index == 12345 && reusing.output_index == 12345 ? 0 : 1;
}
