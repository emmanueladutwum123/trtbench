#include "trtbench/compare.hpp"

namespace trtbench {

const char* to_string(ComparisonStatus s) {
  switch (s) {
    case ComparisonStatus::Ok: return "ok";
    case ComparisonStatus::Refused: return "refused";
    case ComparisonStatus::Untrustworthy: return "untrustworthy";
  }
  return "?";
}

namespace {
std::string invalid_flags(const Result& r) {
  std::string out;
  for (const auto& f : r.findings) {
    if (f.severity != Severity::Invalid) continue;
    if (!out.empty()) out += ", ";
    out += to_string(f.flag);
  }
  return out;
}
}  // namespace

Comparison compare(const Result& base, const Result& cand,
                   std::optional<AccuracyDelta> accuracy) {
  Comparison c;
  c.accuracy = std::move(accuracy);

  if (!base.trustworthy() || !cand.trustworthy()) {
    c.status = ComparisonStatus::Untrustworthy;
    const Result& bad = !base.trustworthy() ? base : cand;
    c.reason = std::string(&bad == &base ? "baseline '" : "candidate '") + bad.name +
               "' failed its own audit (" + invalid_flags(bad) +
               "); no speedup is reported from an invalid measurement";
    return c;
  }
  if (base.config.mode != cand.config.mode) {
    c.status = ComparisonStatus::Refused;
    c.reason = std::string("modes differ (") + to_string(base.config.mode) + " vs " +
               to_string(cand.config.mode) + "); latency and throughput are not comparable";
    return c;
  }
  if (base.precision != cand.precision && !c.accuracy) {
    c.status = ComparisonStatus::Refused;
    c.reason = std::string("precisions differ (") + to_string(base.precision) + " vs " +
               to_string(cand.precision) +
               ") and no accuracy delta was supplied; a lower-precision speedup is only "
               "reported next to what it cost in accuracy";
    return c;
  }
  if (base.device_ms.p50 <= 0 || cand.device_ms.p50 <= 0) {
    c.status = ComparisonStatus::Refused;
    c.reason = "a p50 of zero cannot be compared";
    return c;
  }

  c.status = ComparisonStatus::Ok;
  c.speedup_p50 = base.device_ms.p50 / cand.device_ms.p50;
  c.speedup_p99 = cand.device_ms.p99 > 0 ? base.device_ms.p99 / cand.device_ms.p99 : 0;
  return c;
}

}  // namespace trtbench
