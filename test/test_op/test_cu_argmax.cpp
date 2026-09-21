#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "sampler/argmax_sampler.h"
#include "tensor/tensor.h"

TEST(test_argmax_cu, reuses_workspace_and_handles_partial_block) {
  auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();
  constexpr int size = 17;
  tensor::Tensor logits_cpu(base::DataType::kDataTypeFp32, size, true, cpu_allocator);
  for (int i = 0; i < size; ++i) {
    logits_cpu.index<float>(i) = -static_cast<float>(i);
  }
  logits_cpu.index<float>(3) = 42.0f;
  logits_cpu.index<float>(16) = 42.0f;

  tensor::Tensor logits_cuda = logits_cpu.clone();
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  logits_cuda.to_cuda(stream);

  sampler::ArgmaxSampler sampler(base::DeviceType::kDeviceCUDA);
  for (int i = 0; i < 256; ++i) {
    ASSERT_EQ(sampler.sample(logits_cuda.ptr<float>(), logits_cuda.size(), stream), 3);
  }

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
