#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32_baseline(int size, const float* in1, const float* in2,
                                               float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

__global__ void swiglu_kernel_cu_fp32_optimized(int size, const float* __restrict__ in1,
                                                const float* __restrict__ in2,
                                                float* __restrict__ out) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    const float x = in1[idx];
    const float gate = in2[idx];
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    out[idx] = x * sigmoid * gate;
  }
}

void launch_swiglu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream, bool optimized) {
  const int size = static_cast<int32_t>(input1.size());
  constexpr int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (optimized) {
    swiglu_kernel_cu_fp32_optimized<<<blocks, threads, 0, cuda_stream>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    const size_t shmem = threads * sizeof(float) * 2;
    swiglu_kernel_cu_fp32_baseline<<<blocks, threads, shmem, cuda_stream>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  launch_swiglu(input1, input2, output, stream, false);
}

void swiglu_kernel_cu_baseline(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output, void* stream) {
  launch_swiglu(input1, input2, output, stream, false);
}

void swiglu_kernel_cu_optimized(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output, void* stream) {
  launch_swiglu(input1, input2, output, stream, true);
}
}  // namespace kernel
