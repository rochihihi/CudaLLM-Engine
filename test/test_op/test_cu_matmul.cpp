#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;
TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          nullptr);

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 3, true, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_q8_matches_fp32_reference) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  constexpr int32_t M = 128;
  constexpr int32_t K = 65;
  constexpr int32_t group_size = 64;

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, M, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeInt8, K, M, true, alloc_cpu);
  tensor::Tensor scales_cpu(base::DataType::kDataTypeFp32, K * M / group_size, true, alloc_cpu);
  tensor::Tensor expected(base::DataType::kDataTypeFp32, K, true, alloc_cpu);
  for (int32_t i = 0; i < M; ++i) {
    input_cpu.index<float>(i) = static_cast<float>((i % 11) - 5) * 0.125f;
  }
  for (int32_t group = 0; group < scales_cpu.size(); ++group) {
    scales_cpu.index<float>(group) = 0.01f * static_cast<float>((group % 5) + 1);
  }
  for (int32_t i = 0; i < weight_cpu.size(); ++i) {
    weight_cpu.index<int8_t>(i) = static_cast<int8_t>((i % 15) - 7);
  }
  for (int32_t row = 0; row < K; ++row) {
    float sum = 0.0f;
    for (int32_t col = 0; col < M; ++col) {
      const int32_t weight_index = row * M + col;
      const int32_t group_index = weight_index / group_size;
      sum += input_cpu.index<float>(col) * scales_cpu.index<float>(group_index) *
             static_cast<float>(weight_cpu.index<int8_t>(weight_index));
    }
    expected.index<float>(row) = sum;
  }

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  tensor::Tensor scales_cuda = scales_cpu.clone();
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp32, K, true, alloc_cu);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  CudaConfig config;
  config.stream = stream;
  input_cuda.to_cuda(stream);
  weight_cuda.to_cuda(stream);
  scales_cuda.to_cuda(stream);

  kernel::get_matmul_kernel_quant8(base::DeviceType::kDeviceCUDA)(
      input_cuda, weight_cuda, output_cuda, group_size, scales_cuda, &config);
  output_cuda.to_cpu();
  for (int32_t row = 0; row < K; ++row) {
    ASSERT_NEAR(output_cuda.index<float>(row), expected.index<float>(row), 1e-4f);
  }
}
