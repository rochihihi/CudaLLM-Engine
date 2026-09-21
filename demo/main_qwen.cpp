#include <algorithm>
#include <base/base.h>
#include <base/tick.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "model/qwen2.h"

struct GenerationMetrics {
  int32_t prompt_tokens = 0;
  int32_t generated_tokens = 0;
  double ttft_ms = 0.0;
  double decode_latency_ms = 0.0;
  double total_latency_ms = 0.0;
  double tokens_per_second = 0.0;
};

GenerationMetrics generate(const model::Qwen2Model& model, const std::string& sentence,
                           int total_steps, bool need_output = false) {
  cudaDeviceSynchronize();
  const auto benchmark_start = std::chrono::steady_clock::now();
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  GenerationMetrics metrics;
  metrics.prompt_tokens = prompt_len;
  std::chrono::steady_clock::time_point first_token_time;
  std::vector<int32_t> words;
  words.push_back(next);
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (!is_prompt) {
      ++metrics.generated_tokens;
      if (metrics.generated_tokens == 1) {
        cudaDeviceSynchronize();
        first_token_time = std::chrono::steady_clock::now();
        metrics.ttft_ms =
            std::chrono::duration<double, std::milli>(first_token_time - benchmark_start).count();
      }
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  cudaDeviceSynchronize();
  const auto benchmark_end = std::chrono::steady_clock::now();
  metrics.total_latency_ms =
      std::chrono::duration<double, std::milli>(benchmark_end - benchmark_start).count();
  const int32_t decode_tokens = std::max(0, metrics.generated_tokens - 1);
  if (decode_tokens > 0) {
    metrics.decode_latency_ms =
        std::chrono::duration<double, std::milli>(benchmark_end - first_token_time).count() /
        decode_tokens;
  }
  if (metrics.total_latency_ms > 0.0) {
    metrics.tokens_per_second =
        static_cast<double>(metrics.generated_tokens) * 1000.0 / metrics.total_latency_ms;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return metrics;
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen_infer <checkpoint_path> <tokenizer.json> [--quant] [prompt]";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  bool quantized_model = false;
  bool has_prompt = false;
  std::string sentence = "你好，我是一名";
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--quant") {
      quantized_model = true;
    } else if (!has_prompt) {
      sentence = argument;
      has_prompt = true;
    } else {
      sentence += " ";
      sentence += argument;
    }
  }

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path,
                          quantized_model);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  printf("Model: Qwen2.5 (%s)\n", quantized_model ? "Q8_0 weight-only" : "FP32");
  printf("Prompt: %s\n", sentence.c_str());
  printf("Generating...\n");
  fflush(stdout);
  const GenerationMetrics metrics = generate(model, sentence, 128, true);
  printf("\n\nBenchmark:\n");
  printf("  prompt_tokens: %d\n", metrics.prompt_tokens);
  printf("  generated_tokens: %d\n", metrics.generated_tokens);
  printf("  ttft_ms: %.3f\n", metrics.ttft_ms);
  printf("  avg_decode_latency_ms: %.3f\n", metrics.decode_latency_ms);
  printf("  total_latency_ms: %.3f\n", metrics.total_latency_ms);
  printf("  tokens_per_second: %.3f\n", metrics.tokens_per_second);
  fflush(stdout);
  return 0;
}
