#ifndef CUDALLM_RMSNORM_KERNEL_H
#define CUDALLM_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // CUDALLM_RMSNORM_KERNEL_H
