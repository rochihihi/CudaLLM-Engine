//
// Created by fss on 24-6-9.
//

#ifndef CUDALLM_NON_SAMPLER_H
#define CUDALLM_NON_SAMPLER_H
#include <base/alloc.h>
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type);

  ~ArgmaxSampler() override;

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  std::shared_ptr<base::CUDADeviceAllocator> cuda_allocator_;
  size_t* device_output_index_ = nullptr;
};
}  // namespace sampler
#endif  // CUDALLM_NON_SAMPLER_H
