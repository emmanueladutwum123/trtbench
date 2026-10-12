#include "trtbench/clocks.hpp"

#include <algorithm>

namespace trtbench {

namespace throttle {
std::string describe(std::uint64_t r) {
  struct Bit { std::uint64_t mask; const char* name; };
  static constexpr Bit kBits[] = {
      {kGpuIdle, "gpu_idle"},
      {kAppClocksSetting, "app_clocks_setting"},
      {kSwPowerCap, "sw_power_cap"},
      {kHwSlowdown, "hw_slowdown"},
      {kSyncBoost, "sync_boost"},
      {kSwThermal, "sw_thermal"},
      {kHwThermal, "hw_thermal"},
      {kHwPowerBrake, "hw_power_brake"},
      {kDisplayClock, "display_clock"},
  };
  std::string out;
  for (const auto& b : kBits) {
    if (r & b.mask) {
      if (!out.empty()) out += "|";
      out += b.name;
    }
  }
  return out.empty() ? "none" : out;
}
}  // namespace throttle

namespace {
double drift(unsigned lo, unsigned hi) {
  return hi == 0 ? 0.0 : static_cast<double>(hi - lo) / static_cast<double>(hi);
}
}  // namespace

ClockReport summarize_clocks(bool available, std::string source,
                             const ClockSnapshot& pre_warmup,
                             std::vector<ClockSnapshot> steady) {
  ClockReport r;
  r.available = available;
  r.source = std::move(source);
  r.pre_warmup = pre_warmup;
  r.steady = std::move(steady);
  if (!available || r.steady.empty()) return r;

  std::vector<unsigned> sm;
  sm.reserve(r.steady.size());
  r.sm_min = r.mem_min = ~0u;
  for (const auto& s : r.steady) {
    r.sm_min = std::min(r.sm_min, s.sm_mhz);
    r.sm_max = std::max(r.sm_max, s.sm_mhz);
    r.mem_min = std::min(r.mem_min, s.mem_mhz);
    r.mem_max = std::max(r.mem_max, s.mem_mhz);
    r.throttle_union |= s.throttle_reasons;
    sm.push_back(s.sm_mhz);
  }
  std::sort(sm.begin(), sm.end());
  r.sm_median = sm[(sm.size() - 1) / 2];
  r.sm_drift = drift(r.sm_min, r.sm_max);
  r.mem_drift = drift(r.mem_min, r.mem_max);
  return r;
}

}  // namespace trtbench
