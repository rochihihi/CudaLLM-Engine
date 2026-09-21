#ifndef CUDALLM_SAMPLER_H
#define CUDALLM_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual ~Sampler() = default;

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

 protected:
  base::DeviceType device_type_;
};
}  // namespace sampler
#endif  // CUDALLM_SAMPLER_H
