#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32_baseline(const float* input, const float* weight,
                                               float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int WARPS_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32_optimized(const float* __restrict__ input,
                                                const float* __restrict__ weight,
                                                float* __restrict__ output, int M, int K) {
  constexpr int warp_size = 32;
  const int warp_id = threadIdx.x / warp_size;
  const int lane_id = threadIdx.x % warp_size;
  const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (row >= K) {
    return;
  }

  float thread_sum = 0.0f;
  const int row_offset = row * M;
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_offset = pack_num * pack_size;

  if (M % pack_size == 0) {
    const float4* input_float4 = reinterpret_cast<const float4*>(input);
    const float4* weight_float4 = reinterpret_cast<const float4*>(weight + row_offset);
    for (int i = lane_id; i < pack_num; i += warp_size) {
      const float4 input_value = input_float4[i];
      const float4 weight_value = weight_float4[i];
      thread_sum += input_value.x * weight_value.x + input_value.y * weight_value.y +
                    input_value.z * weight_value.z + input_value.w * weight_value.w;
    }
  } else {
    for (int i = lane_id; i < pack_offset; i += warp_size) {
      thread_sum += input[i] * weight[row_offset + i];
    }
  }

  for (int i = pack_offset + lane_id; i < M; i += warp_size) {
    thread_sum += input[i] * weight[row_offset + i];
  }

  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }
  if (lane_id == 0) {
    output[row] = thread_sum;
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void check_matmul_tensors(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  CHECK_EQ(M, input.get_dim(0));
  CHECK_EQ(K, output.get_dim(0));
}

void matmul_kernel_cu_baseline(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, const float scale,
                               const CudaConfig* config) {
  check_matmul_tensors(input, weight, output);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  if (config && config->stream) {
    matmul_kernel_cu_fp32_baseline<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32_baseline<128, 1><<<K, 128>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_optimized(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, const float scale,
                                const CudaConfig* config) {
  check_matmul_tensors(input, weight, output);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  cudaStream_t stream = config ? config->stream : nullptr;
  if (M >= 2048) {
    matmul_kernel_cu_fp32_baseline<128, 1><<<K, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    return;
  }
  constexpr int warps_per_block = 8;
  constexpr int threads_per_block = warps_per_block * 32;
  const int blocks = (K + warps_per_block - 1) / warps_per_block;
  matmul_kernel_cu_fp32_optimized<warps_per_block><<<blocks, threads_per_block, 0, stream>>>(
      input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  matmul_kernel_cu_baseline(input, weight, output, scale, config);
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel
