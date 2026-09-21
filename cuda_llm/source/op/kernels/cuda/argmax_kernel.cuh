#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cstddef>
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, size_t* device_output_idx,
                        void* stream);
}
#endif  // ARGMAX_KERNEL_CUH
