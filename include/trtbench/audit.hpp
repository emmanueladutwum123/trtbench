#pragma once

// The lie detector. Pure functions over collected samples: no CUDA, no
// clocks, no I/O, so every rule here is unit-tested on a CPU-only runner and
// the Phase 5 trap suite can feed it deliberately wrong measurements.

#include <cstdint>
#include <string>
#include <vector>

#include "trtbench/clocks.hpp"
#include "trtbench/config.hpp"

namespace trtbench {

// Severity decides what a finding does to the result:
//   Invalid -> Result::trustworthy() is false; the CLI exits non-zero and
//              compare() refuses to compute a speedup from it.
//   Warning -> number is reported, finding is printed next to it.
//   Info    -> recorded, printed, never changes a verdict.
enum class Severity { Info, Warning, Invalid };

enum class Flag {
  // Invalid
  NoSamples,
  ImpossibleBandwidth,       // achieved DRAM GB/s above theoretical peak
  ImpossibleFlops,           // achieved FLOP/s above theoretical peak
  WallclockBelowDeviceTime,  // host interval that encloses the work is shorter than the work
  ClockDrift,                // SM or memory clock moved more than tolerance
  ThermalOrHwThrottle,       // NVML reported thermal / hardware slowdown
  ChecksumFailed,            // output did not match the reference
  // Warning
  ClocksUnavailable,         // could not observe clocks: drift is unverified
  PowerCapThrottle,          // clocks set by power limit, not by clock policy
  AtLaunchFloor,             // indistinguishable from an empty kernel launch
  LaunchBound,               // host enqueue time comparable to device time
  TooFewSamples,             // too few samples for p99 to mean anything
  HighVariance,
  NoWarmup,
  OutputUnverified,          // workload cannot check its output
  // Info
  FirstCallOutlier,          // first call much slower than steady state
};

const char* to_string(Flag f);
const char* to_string(Severity s);
Severity severity_of(Flag f);

struct Finding {
  Flag flag;
  Severity severity;
  std::string detail;
};

// Per-launch values in milliseconds. The runner divides each sample by the
// number of launches it contained.
struct RawSamples {
  std::vector<double> device_ms;        // CUDA events
  std::vector<double> host_enqueue_ms;  // std::chrono around the enqueue calls only
  std::vector<double> wall_ms;          // std::chrono enclosing record(start) .. sync(stop)
  double first_device_ms = -1;
  double first_host_ms = -1;
  double first_wall_ms = -1;
};

struct Checksum {
  bool verifiable = false;  // the workload knows the right answer
  bool passed = false;
  double value = 0;
  std::string note;
};

struct AuditThresholds {
  // Theoretical peaks come from nominal clocks; allow 5% for clock rounding
  // before calling a number physically impossible.
  double peak_tolerance = 1.05;
  // First call more than this many times the steady-state p50 is surfaced.
  double first_call_outlier_ratio = 2.0;
  // Latency mode: host enqueue p50 at or above this fraction of device p50.
  double launch_bound_ratio = 0.8;
  // p90 / p50 above this is a noisy distribution.
  double high_variance_ratio = 1.25;
  // p99 of fewer than 100 samples is the max of a handful of points.
  std::size_t min_samples_for_p99 = 100;
  // Event timer resolution is ~0.5 us; wall can trail device by that much.
  double wall_tolerance_ms = 0.001;
  // p50 within 10% of the empty-kernel floor.
  double launch_floor_margin = 1.10;
};

struct AuditInput {
  Mode mode = Mode::Latency;
  int warmup_iters = 0;
  const RawSamples* samples = nullptr;
  const ClockReport* clocks = nullptr;
  double clock_drift_tol = 0.05;
  std::uint64_t bytes = 0;
  std::uint64_t flops = 0;
  double dram_gbps_theoretical = 0;  // 0 = unknown, skip the check
  double peak_gflops_max_clock = 0;  // 0 = unknown, skip the check
  double launch_floor_ms = 0;        // 0 = not calibrated, skip the check
  Checksum checksum;
};

std::vector<Finding> audit(const AuditInput& in, const AuditThresholds& th = {});

}  // namespace trtbench
