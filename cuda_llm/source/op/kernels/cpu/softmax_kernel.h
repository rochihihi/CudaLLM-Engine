#ifndef CUDALLM_SOFTMAX_KERNEL_H
#define CUDALLM_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif  // CUDALLM_SOFTMAX_KERNEL_H
