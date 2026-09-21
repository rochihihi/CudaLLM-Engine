#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <base/buffer.h>
#include <base/cuda_config.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <op/kernels/cuda/matmul_kernel.cuh>
#include <tensor/tensor.h>

namespace {
using MatmulKernel = void (*)(const tensor::Tensor&, const tensor::Tensor&,
                              const tensor::Tensor&, float, const kernel::CudaConfig*);

struct BenchmarkResult {
  float latency_ms = 0.0f;
  double bandwidth_gb_s = 0.0;
};

BenchmarkResult benchmark_kernel(MatmulKernel kernel_fn, const tensor::Tensor& input,
                                 const tensor::Tensor& weight, const tensor::Tensor& output,
                                 const kernel::CudaConfig& config, int warmups, int repeats) {
  for (int i = 0; i < warmups; ++i) {
    kernel_fn(input, weight, output, 1.0f, &config);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, config.stream);
  for (int i = 0; i < repeats; ++i) {
    kernel_fn(input, weight, output, 1.0f, &config);
  }
  cudaEventRecord(stop, config.stream);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  BenchmarkResult result;
  result.latency_ms = total_ms / repeats;
  const double bytes = static_cast<double>(weight.size() + input.size() + output.size()) *
                       sizeof(float);
  result.bandwidth_gb_s = bytes / (result.latency_ms * 1.0e6);
  return result;
}

BenchmarkResult median_result(std::vector<BenchmarkResult> results) {
  std::sort(results.begin(), results.end(),
            [](const BenchmarkResult& lhs, const BenchmarkResult& rhs) {
              return lhs.latency_ms < rhs.latency_ms;
            });
  return results.at(results.size() / 2);
}

BenchmarkResult benchmark_cublas(cublasHandle_t handle, const tensor::Tensor& input,
                                 const tensor::Tensor& weight, const tensor::Tensor& output,
                                 const kernel::CudaConfig& config, int warmups, int repeats) {
  const int K = weight.get_dim(0);
  const int M = weight.get_dim(1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto launch = [&] {
    cublasSgemv(handle, CUBLAS_OP_T, M, K, &alpha, weight.ptr<float>(), M, input.ptr<float>(), 1,
                &beta, const_cast<float*>(output.ptr<float>()), 1);
  };
  for (int i = 0; i < warmups; ++i) {
    launch();
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, config.stream);
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  cudaEventRecord(stop, config.stream);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  BenchmarkResult result;
  result.latency_ms = total_ms / repeats;
  const double bytes = static_cast<double>(weight.size() + input.size() + output.size()) *
                       sizeof(float);
  result.bandwidth_gb_s = bytes / (result.latency_ms * 1.0e6);
  return result;
}

void run_case(int K, int M, int warmups, int repeats) {
  auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda_allocator = base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, M, true, cpu_allocator);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, K, M, true, cpu_allocator);
  for (int i = 0; i < M; ++i) {
    input_cpu.index<float>(i) = static_cast<float>((i % 17) - 8) * 0.03125f;
  }
  for (size_t i = 0; i < weight_cpu.size(); ++i) {
    weight_cpu.index<float>(i) = static_cast<float>((i % 13) - 6) * 0.015625f;
  }

  kernel::CudaConfig config;
  cudaStreamCreate(&config.stream);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, config.stream);
  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  input_cuda.to_cuda(config.stream);
  weight_cuda.to_cuda(config.stream);
  tensor::Tensor output_baseline(base::DataType::kDataTypeFp32, K, true, cuda_allocator);
  tensor::Tensor output_optimized(base::DataType::kDataTypeFp32, K, true, cuda_allocator);
  tensor::Tensor output_cublas(base::DataType::kDataTypeFp32, K, true, cuda_allocator);

  constexpr int rounds = 7;
  std::vector<BenchmarkResult> baseline_results;
  std::vector<BenchmarkResult> optimized_results;
  std::vector<BenchmarkResult> cublas_results;
  for (int round = 0; round < rounds; ++round) {
    if (round % 2 == 0) {
      baseline_results.push_back(benchmark_kernel(
          kernel::matmul_kernel_cu_baseline, input_cuda, weight_cuda, output_baseline, config,
          warmups, repeats));
      optimized_results.push_back(benchmark_kernel(
          kernel::matmul_kernel_cu_optimized, input_cuda, weight_cuda, output_optimized, config,
          warmups, repeats));
      cublas_results.push_back(benchmark_cublas(cublas_handle, input_cuda, weight_cuda,
                                                output_cublas, config, warmups, repeats));
    } else {
      cublas_results.push_back(benchmark_cublas(cublas_handle, input_cuda, weight_cuda,
                                                output_cublas, config, warmups, repeats));
      optimized_results.push_back(benchmark_kernel(
          kernel::matmul_kernel_cu_optimized, input_cuda, weight_cuda, output_optimized, config,
          warmups, repeats));
      baseline_results.push_back(benchmark_kernel(
          kernel::matmul_kernel_cu_baseline, input_cuda, weight_cuda, output_baseline, config,
          warmups, repeats));
    }
  }
  const BenchmarkResult baseline = median_result(std::move(baseline_results));
  const BenchmarkResult optimized = median_result(std::move(optimized_results));
  const BenchmarkResult cublas = median_result(std::move(cublas_results));

  output_baseline.to_cpu();
  output_optimized.to_cpu();
  output_cublas.to_cpu();
  float max_abs_error = 0.0f;
  float cublas_max_abs_error = 0.0f;
  for (int i = 0; i < K; ++i) {
    max_abs_error =
        std::max(max_abs_error, std::abs(output_baseline.index<float>(i) -
                                        output_optimized.index<float>(i)));
    cublas_max_abs_error =
        std::max(cublas_max_abs_error, std::abs(output_baseline.index<float>(i) -
                                               output_cublas.index<float>(i)));
  }

  printf("%-12s K=%-6d M=%-6d baseline=%8.4f ms  optimized=%8.4f ms (%5.2fx)  "
         "cuBLAS=%8.4f ms (%5.2fx)  bandwidth=%7.2f GB/s  errors=%.8f/%.8f\n",
         M == 2048 ? "w2" : (K == 32000 ? "cls" : (K == 2048 ? "w1/w3" : "qkv/wo")),
         K, M, baseline.latency_ms, optimized.latency_ms,
         baseline.latency_ms / optimized.latency_ms, cublas.latency_ms,
         baseline.latency_ms / cublas.latency_ms, optimized.bandwidth_gb_s, max_abs_error,
         cublas_max_abs_error);
  cublasDestroy(cublas_handle);
}
}  // namespace

int main() {
  constexpr int warmups = 20;
  constexpr int repeats = 100;
  printf("MatMul CUDA Benchmark (warmups=%d, repeats=%d, rounds=7, median)\n", warmups,
         repeats);
  run_case(768, 768, warmups, repeats);
  run_case(2048, 768, warmups, repeats);
  run_case(768, 2048, warmups, repeats);
  run_case(32000, 768, warmups, repeats);
  return 0;
}
