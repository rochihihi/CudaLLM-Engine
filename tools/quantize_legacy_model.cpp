#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct LegacyConfig {
  int32_t dim;
  int32_t hidden_dim;
  int32_t layer_num;
  int32_t head_num;
  int32_t kv_head_num;
  int32_t vocab_size;
  int32_t seq_len;
};

struct Layout {
  uint64_t embedding;
  uint64_t attention_norm;
  uint64_t wq;
  uint64_t wk;
  uint64_t wv;
  uint64_t wo;
  uint64_t ffn_norm;
  uint64_t w1;
  uint64_t w2;
  uint64_t w3;
  uint64_t final_norm;
  uint64_t classifier;
  uint64_t total_elements;
};

bool write_bytes(std::ofstream& output, const void* data, uint64_t bytes) {
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  return output.good();
}

bool quantize_matrix(std::ofstream& output, const float* values, uint64_t elements,
                     int32_t group_size, float& global_max_error) {
  if (elements % group_size != 0) {
    return false;
  }
  std::vector<int8_t> quantized(elements);
  std::vector<float> scales(elements / group_size);
  for (uint64_t group = 0; group < scales.size(); ++group) {
    const uint64_t offset = group * group_size;
    float max_abs = 0.0f;
    for (int32_t i = 0; i < group_size; ++i) {
      max_abs = std::max(max_abs, std::abs(values[offset + i]));
    }
    const float scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    scales[group] = scale;
    for (int32_t i = 0; i < group_size; ++i) {
      int32_t q = 0;
      if (scale != 0.0f) {
        q = static_cast<int32_t>(std::round(values[offset + i] / scale));
        q = std::clamp(q, -127, 127);
      }
      quantized[offset + i] = static_cast<int8_t>(q);
      global_max_error =
          std::max(global_max_error, std::abs(values[offset + i] - q * scale));
    }
  }
  return write_bytes(output, quantized.data(), quantized.size()) &&
         write_bytes(output, scales.data(), scales.size() * sizeof(float));
}

bool quantize_layers(std::ofstream& output, const float* weights, uint64_t section_offset,
                     uint64_t elements_per_layer, int32_t layers, int32_t group_size,
                     const std::string& name, float& global_max_error) {
  std::cout << "Quantizing " << name << " (" << layers << " matrices)..." << std::endl;
  for (int32_t layer = 0; layer < layers; ++layer) {
    const float* matrix = weights + section_offset + layer * elements_per_layer;
    if (!quantize_matrix(output, matrix, elements_per_layer, group_size, global_max_error)) {
      std::cerr << "Failed to quantize " << name << " layer " << layer << std::endl;
      return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: quantize_legacy_model <fp32_model.bin> <q8_model.bin> [group_size]"
              << std::endl;
    return 1;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const int32_t group_size = argc == 4 ? std::stoi(argv[3]) : 64;
  if (group_size <= 0) {
    std::cerr << "group_size must be positive" << std::endl;
    return 1;
  }

  const int fd = open(input_path.c_str(), O_RDONLY);
  struct stat file_stat {};
  if (fd < 0 || fstat(fd, &file_stat) != 0 || file_stat.st_size < sizeof(LegacyConfig)) {
    std::cerr << "Cannot open a valid input model: " << input_path << std::endl;
    if (fd >= 0) close(fd);
    return 1;
  }
  void* mapped = mmap(nullptr, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    std::cerr << "Cannot map input model" << std::endl;
    close(fd);
    return 1;
  }

  LegacyConfig config {};
  std::memcpy(&config, mapped, sizeof(config));
  const int64_t vocab = std::abs(static_cast<int64_t>(config.vocab_size));
  const int64_t kv_dim =
      static_cast<int64_t>(config.dim) * config.kv_head_num / config.head_num;
  const int64_t head_size = config.dim / config.head_num;
  if (config.dim <= 0 || config.hidden_dim <= 0 || config.layer_num <= 0 || vocab <= 0 ||
      kv_dim <= 0 || config.seq_len <= 0) {
    std::cerr << "Invalid model configuration" << std::endl;
    munmap(mapped, file_stat.st_size);
    close(fd);
    return 1;
  }

  uint64_t offset = 0;
  Layout layout {};
  layout.embedding = offset;
  offset += vocab * config.dim;
  layout.attention_norm = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.dim;
  layout.wq = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.dim * config.dim;
  layout.wk = offset;
  offset += static_cast<uint64_t>(config.layer_num) * kv_dim * config.dim;
  layout.wv = offset;
  offset += static_cast<uint64_t>(config.layer_num) * kv_dim * config.dim;
  layout.wo = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.dim * config.dim;
  layout.ffn_norm = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.dim;
  layout.w1 = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.hidden_dim * config.dim;
  layout.w2 = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.dim * config.hidden_dim;
  layout.w3 = offset;
  offset += static_cast<uint64_t>(config.layer_num) * config.hidden_dim * config.dim;
  layout.final_norm = offset;
  offset += config.dim;
  offset += static_cast<uint64_t>(config.seq_len) * head_size;
  layout.classifier = offset;
  if (config.vocab_size < 0) {
    offset += vocab * config.dim;
  }
  layout.total_elements = offset;

  const uint64_t expected_bytes = sizeof(LegacyConfig) + offset * sizeof(float);
  if (expected_bytes != static_cast<uint64_t>(file_stat.st_size)) {
    std::cerr << "Unexpected file size. Expected " << expected_bytes << ", got "
              << file_stat.st_size << std::endl;
    munmap(mapped, file_stat.st_size);
    close(fd);
    return 1;
  }

  const float* weights = reinterpret_cast<const float*>(
      static_cast<const uint8_t*>(mapped) + sizeof(LegacyConfig));
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "Cannot open output model: " << output_path << std::endl;
    munmap(mapped, file_stat.st_size);
    close(fd);
    return 1;
  }
  write_bytes(output, &config, sizeof(config));
  write_bytes(output, &group_size, sizeof(group_size));

  float max_error = 0.0f;
  const uint64_t dim_dim = static_cast<uint64_t>(config.dim) * config.dim;
  const uint64_t kv_dim_dim = static_cast<uint64_t>(kv_dim) * config.dim;
  const uint64_t hidden_dim = static_cast<uint64_t>(config.hidden_dim) * config.dim;
  bool ok = true;
  ok &= quantize_layers(output, weights, layout.wq, dim_dim, config.layer_num, group_size, "WQ",
                        max_error);
  ok &= quantize_layers(output, weights, layout.wk, kv_dim_dim, config.layer_num, group_size, "WK",
                        max_error);
  ok &= quantize_layers(output, weights, layout.wv, kv_dim_dim, config.layer_num, group_size, "WV",
                        max_error);
  ok &= quantize_layers(output, weights, layout.wo, dim_dim, config.layer_num, group_size, "WO",
                        max_error);
  ok &= quantize_layers(output, weights, layout.w1, hidden_dim, config.layer_num, group_size, "W1",
                        max_error);
  ok &= quantize_layers(output, weights, layout.w2, hidden_dim, config.layer_num, group_size, "W2",
                        max_error);
  ok &= quantize_layers(output, weights, layout.w3, hidden_dim, config.layer_num, group_size, "W3",
                        max_error);
  if (config.vocab_size < 0) {
    ok &= quantize_layers(output, weights, layout.classifier, vocab * config.dim, 1, group_size,
                          "classifier", max_error);
  }

  ok &= write_bytes(output, weights + layout.embedding, vocab * config.dim * sizeof(float));
  ok &= write_bytes(output, weights + layout.attention_norm,
                    static_cast<uint64_t>(config.layer_num) * config.dim * sizeof(float));
  ok &= write_bytes(output, weights + layout.ffn_norm,
                    static_cast<uint64_t>(config.layer_num) * config.dim * sizeof(float));
  ok &= write_bytes(output, weights + layout.final_norm,
                    static_cast<uint64_t>(config.dim) * sizeof(float));
  output.close();

  munmap(mapped, file_stat.st_size);
  close(fd);
  if (!ok) {
    std::cerr << "Failed while writing quantized model" << std::endl;
    return 1;
  }
  struct stat output_stat {};
  stat(output_path.c_str(), &output_stat);
  std::cout << "Wrote " << output_path << " (" << output_stat.st_size << " bytes)" << std::endl;
  std::cout << "Maximum absolute quantization error: " << max_error << std::endl;
  return 0;
}
