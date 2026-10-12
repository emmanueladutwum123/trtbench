#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtbench {

// NVML clock-event ("throttle") reason bits, duplicated here so CPU-only code
// and tests do not need nvml.h. Values are NVML's stable ABI.
namespace throttle {
inline constexpr std::uint64_t kGpuIdle = 0x1;
inline constexpr std::uint64_t kAppClocksSetting = 0x2;
inline constexpr std::uint64_t kSwPowerCap = 0x4;
inline constexpr std::uint64_t kHwSlowdown = 0x8;
inline constexpr std::uint64_t kSyncBoost = 0x10;
inline constexpr std::uint64_t kSwThermal = 0x20;
inline constexpr std::uint64_t kHwThermal = 0x40;
inline constexpr std::uint64_t kHwPowerBrake = 0x80;
inline constexpr std::uint64_t kDisplayClock = 0x100;

// Reasons that mean the hardware pulled clocks down underneath the run.
inline constexpr std::uint64_t kThermalOrHardware =
    kHwSlowdown | kSwThermal | kHwThermal | kHwPowerBrake;

std::string describe(std::uint64_t reasons);
}  // namespace throttle

struct ClockSnapshot {
  double t_ms = 0;  // since the start of the run, host clock
  unsigned sm_mhz = 0;
  unsigned mem_mhz = 0;
  std::uint64_t throttle_reasons = 0;
  double temp_c = 0;
  double power_w = 0;
};

// Abstract so tests can inject a throttling GPU without having one.
class ClockSource {
 public:
  virtual ~ClockSource() = default;
  virtual bool available() const = 0;
  virtual ClockSnapshot sample() = 0;
  virtual std::string name() const = 0;
};

class NullClockSource final : public ClockSource {
 public:
  explicit NullClockSource(std::string why) : why_(std::move(why)) {}
  bool available() const override { return false; }
  ClockSnapshot sample() override { return {}; }
  std::string name() const override { return "none (" + why_ + ")"; }

 private:
  std::string why_;
};

struct ClockReport {
  bool available = false;
  std::string source;
  bool locked = false;
  std::string lock_status;

  // Snapshot taken before warmup. The GPU is usually idle here, so it is kept
  // for the record but excluded from drift: idle -> boost is not throttling.
  ClockSnapshot pre_warmup;
  // Steady-state snapshots: first = after warmup, last = after the run.
  std::vector<ClockSnapshot> steady;

  unsigned sm_min = 0, sm_max = 0, sm_median = 0;
  unsigned mem_min = 0, mem_max = 0;
  double sm_drift = 0;   // (max - min) / max
  double mem_drift = 0;
  std::uint64_t throttle_union = 0;  // OR of reasons over steady state

  double max_drift() const { return sm_drift > mem_drift ? sm_drift : mem_drift; }
};

ClockReport summarize_clocks(bool available, std::string source,
                             const ClockSnapshot& pre_warmup,
                             std::vector<ClockSnapshot> steady);

}  // namespace trtbench
