#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
ArgmaxSampler::ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cuda_allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    device_output_index_ = static_cast<size_t*>(cuda_allocator_->allocate(sizeof(size_t)));
    CHECK_NE(device_output_index_, nullptr);
  }
}

ArgmaxSampler::~ArgmaxSampler() {
  if (cuda_allocator_ && device_output_index_) {
    cuda_allocator_->release(device_output_index_);
    device_output_index_ = nullptr;
  }
}

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  CHECK_NE(logits, nullptr);
  CHECK_GT(size, 0);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    CHECK_NE(device_output_index_, nullptr);
    size_t next = kernel::argmax_kernel_cu(logits, size, device_output_index_, stream);
    return next;
  }
}
}  // namespace sampler
