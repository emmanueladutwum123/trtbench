#include "trtbench/result.hpp"

#include <algorithm>

namespace trtbench {

bool Result::trustworthy() const {
  return std::none_of(findings.begin(), findings.end(),
                      [](const Finding& f) { return f.severity == Severity::Invalid; });
}

bool Result::has(Flag f) const {
  return std::any_of(findings.begin(), findings.end(),
                     [f](const Finding& x) { return x.flag == f; });
}

bool RunReport::all_trustworthy() const {
  return std::all_of(results.begin(), results.end(),
                     [](const Result& r) { return r.trustworthy(); });
}

Result finalize(FinalizeInput in, const AuditThresholds& th) {
  Result r;
  r.name = std::move(in.name);
  r.precision = in.precision;
  r.config = in.config;
  r.launches_per_sample = in.config.mode == Mode::Latency ? 1 : in.config.launches_per_sample;
  r.bytes_per_launch = in.bytes;
  r.flops_per_launch = in.flops;

  r.device_ms = summarize(in.raw.device_ms);
  r.host_enqueue_ms = summarize(in.raw.host_enqueue_ms);
  r.wall_ms = summarize(in.raw.wall_ms);
  r.first_call_device_ms = in.raw.first_device_ms;
  r.first_call_host_ms = in.raw.first_host_ms;
  r.launch_floor_ms = in.launch_floor_ms;
  r.checksum = in.checksum;

  // Compute % of peak against the clock the GPU actually ran at.
  const unsigned run_mhz = in.clocks.available ? in.clocks.sm_median : 0;
  r.roofline = place(in.bytes, in.flops, r.device_ms.p50, in.peaks.dram_gbps_reference(),
                     in.peaks.dram_gbps_theoretical,
                     compute_peak_gflops(in.peaks, in.precision, run_mhz));

  AuditInput a;
  a.mode = in.config.mode;
  a.warmup_iters = in.config.warmup_iters;
  a.samples = &in.raw;
  a.clocks = &in.clocks;
  a.clock_drift_tol = in.config.clock_drift_tol;
  a.bytes = in.bytes;
  a.flops = in.flops;
  a.dram_gbps_theoretical = in.peaks.dram_gbps_theoretical;
  // Impossibility is judged against the maximum clock, not the run clock:
  // that bound cannot be beaten under any clock state.
  a.peak_gflops_max_clock = compute_peak_gflops(in.peaks, in.precision, 0);
  a.launch_floor_ms = in.launch_floor_ms;
  a.checksum = in.checksum;
  r.findings = audit(a, th);

  r.device_samples_ms = std::move(in.raw.device_ms);
  r.clocks = std::move(in.clocks);
  return r;
}

#ifndef TRTBENCH_GIT_SHA
#define TRTBENCH_GIT_SHA "unknown"
#endif
#ifndef TRTBENCH_VERSION
#define TRTBENCH_VERSION "unknown"
#endif

std::string build_version() { return TRTBENCH_VERSION; }
std::string build_git_sha() { return TRTBENCH_GIT_SHA; }

std::string build_compiler() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "unknown";
#endif
}

}  // namespace trtbench
