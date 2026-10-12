#pragma once

#include <optional>
#include <string>

#include "trtbench/result.hpp"

namespace trtbench {

// An accuracy measurement on the same inputs for both sides of a comparison,
// e.g. top-1 on a held-out set, or max |logit difference| vs FP32.
struct AccuracyDelta {
  std::string metric;
  double baseline = 0;
  double candidate = 0;
  double delta() const { return candidate - baseline; }
};

enum class ComparisonStatus {
  Ok,
  Refused,        // the comparison is not meaningful as asked
  Untrustworthy,  // at least one side failed its own audit
};
const char* to_string(ComparisonStatus s);

struct Comparison {
  ComparisonStatus status = ComparisonStatus::Refused;
  double speedup_p50 = 0;  // baseline p50 / candidate p50; only set when Ok
  double speedup_p99 = 0;
  std::string reason;
  std::optional<AccuracyDelta> accuracy;
};

// Refuses rather than prints when:
//   - either result is not trustworthy;
//   - the modes differ (latency vs throughput answer different questions);
//   - the precisions differ and no accuracy delta is supplied: a faster FP16
//     engine is only a result alongside what it cost in accuracy.
Comparison compare(const Result& baseline, const Result& candidate,
                   std::optional<AccuracyDelta> accuracy = std::nullopt);

}  // namespace trtbench
