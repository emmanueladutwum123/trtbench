#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "trtbench/audit.hpp"
#include "trtbench/clocks.hpp"
#include "trtbench/config.hpp"
#include "trtbench/roofline.hpp"
#include "trtbench/stats.hpp"

namespace trtbench {

// Everything a reviewer needs to reproduce or reject a number. Integer fields
// use -1 and strings use "" for "could not be queried".
struct Environment {
  std::string gpu_name;
  std::string pci_bus_id;
  int cc_major = -1, cc_minor = -1;
  int sm_count = -1;
  int max_threads_per_sm = -1;
  long long l2_bytes = -1;
  int mem_bus_width_bits = -1;
  double mem_clock_khz = -1;
  int max_sm_clock_mhz = -1;
  int default_app_sm_clock_mhz = -1;
  std::string driver_version;
  int cuda_driver_api_version = -1;
  int cuda_runtime_version = -1;
  int persistence_mode = -1;  // 1 on, 0 off
  int ecc_enabled = -1;
  double power_limit_w = -1;
  std::string cuda_module_loading;  // CUDA_MODULE_LOADING env var
  std::string nvml_status;
  std::string hostname;
  std::string timestamp_utc;
  std::string trtbench_version;
  std::string git_sha;
  std::string compiler;
};

struct Result {
  std::string name;
  Precision precision = Precision::FP32;
  BenchConfig config;
  int launches_per_sample = 1;
  std::uint64_t bytes_per_launch = 0;
  std::uint64_t flops_per_launch = 0;

  // Two distinct numbers, never summed: device time from CUDA events, and the
  // host-side cost of the enqueue calls from std::chrono.
  Percentiles device_ms;
  Percentiles host_enqueue_ms;
  Percentiles wall_ms;  // cross-check only; see Flag::WallclockBelowDeviceTime

  // Reported separately and excluded from the percentiles above.
  double first_call_device_ms = -1;
  double first_call_host_ms = -1;

  std::vector<double> device_samples_ms;  // raw steady-state samples, for diffing
  double launch_floor_ms = 0;
  ClockReport clocks;
  RooflinePoint roofline;  // placed at device p50
  Checksum checksum;
  std::vector<Finding> findings;

  bool trustworthy() const;
  bool has(Flag f) const;
};

struct FinalizeInput {
  std::string name;
  Precision precision = Precision::FP32;
  BenchConfig config;
  std::uint64_t bytes = 0;
  std::uint64_t flops = 0;
  RawSamples raw;
  ClockReport clocks;
  DevicePeaks peaks;
  double launch_floor_ms = 0;
  Checksum checksum;
};

// Turns raw samples into a Result: percentiles, roofline, audit. Pure.
Result finalize(FinalizeInput in, const AuditThresholds& th = {});

struct RunReport {
  int schema_version = 1;
  Environment env;
  DevicePeaks peaks;
  double launch_floor_latency_ms = 0;
  double launch_floor_throughput_ms = 0;
  std::vector<Result> calibration;  // empty-kernel floors and copy peak, kept for audit
  std::vector<Result> results;

  bool all_trustworthy() const;
};

// Build metadata baked in at configure time.
std::string build_version();
std::string build_git_sha();
std::string build_compiler();

}  // namespace trtbench
