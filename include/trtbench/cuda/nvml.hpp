#pragma once

#include <memory>
#include <string>

#include "trtbench/clocks.hpp"

namespace trtbench::cuda {

// Static facts NVML can tell us about the device; -1 / "" when unavailable.
struct NvmlDeviceInfo {
  int index = -1;  // NVML / nvidia-smi index (may differ from the CUDA ordinal)
  std::string driver_version;
  int max_sm_clock_mhz = -1;
  int default_app_sm_clock_mhz = -1;
  int persistence_mode = -1;
  int ecc_enabled = -1;
  double power_limit_w = -1;
};

// NVML is loaded with dlopen rather than linked, so binaries start on
// machines without a driver (CI) and report clocks as unavailable instead of
// failing to load.
class NvmlClockSource final : public ClockSource {
 public:
  // Opens NVML and finds the device by PCI bus ID, which is the only reliable
  // mapping between CUDA ordinals and NVML indices. Returns nullptr and sets
  // `why` on failure.
  static std::unique_ptr<NvmlClockSource> open(const std::string& pci_bus_id, std::string* why);
  ~NvmlClockSource() override;

  NvmlClockSource(const NvmlClockSource&) = delete;
  NvmlClockSource& operator=(const NvmlClockSource&) = delete;

  bool available() const override { return true; }
  ClockSnapshot sample() override;
  std::string name() const override { return "nvml"; }
  const NvmlDeviceInfo& info() const { return info_; }

  struct Api;

 private:
  NvmlClockSource() = default;
  std::unique_ptr<Api> api_;
  NvmlDeviceInfo info_;
};

}  // namespace trtbench::cuda
