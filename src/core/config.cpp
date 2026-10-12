#include "trtbench/config.hpp"

namespace trtbench {

const char* to_string(Mode m) {
  switch (m) {
    case Mode::Latency: return "latency";
    case Mode::Throughput: return "throughput";
  }
  return "?";
}

const char* to_string(Precision p) {
  switch (p) {
    case Precision::FP32: return "fp32";
    case Precision::FP16: return "fp16";
    case Precision::BF16: return "bf16";
    case Precision::INT8: return "int8";
    case Precision::Mixed: return "mixed";
  }
  return "?";
}

std::string validate(const BenchConfig& c) {
  if (c.warmup_iters < 0) return "warmup_iters must be >= 0";
  if (c.warmup_min_ms < 0) return "warmup_min_ms must be >= 0";
  if (c.min_iters < 1) return "min_iters must be >= 1";
  if (c.max_iters < c.min_iters) return "max_iters must be >= min_iters";
  if (c.min_time_ms < 0) return "min_time_ms must be >= 0";
  if (c.launches_per_sample < 1) return "launches_per_sample must be >= 1";
  if (c.clock_sample_every < 0) return "clock_sample_every must be >= 0";
  if (!(c.clock_drift_tol > 0)) return "clock_drift_tol must be > 0";
  return {};
}

}  // namespace trtbench
