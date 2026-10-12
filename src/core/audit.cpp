#include "trtbench/audit.hpp"

#include <algorithm>
#include <cstdio>

#include "trtbench/stats.hpp"

namespace trtbench {

const char* to_string(Flag f) {
  switch (f) {
    case Flag::NoSamples: return "no_samples";
    case Flag::ImpossibleBandwidth: return "impossible_bandwidth";
    case Flag::ImpossibleFlops: return "impossible_flops";
    case Flag::WallclockBelowDeviceTime: return "wallclock_below_device_time";
    case Flag::ClockDrift: return "clock_drift";
    case Flag::ThermalOrHwThrottle: return "thermal_or_hw_throttle";
    case Flag::ChecksumFailed: return "checksum_failed";
    case Flag::ClocksUnavailable: return "clocks_unavailable";
    case Flag::PowerCapThrottle: return "power_cap_throttle";
    case Flag::AtLaunchFloor: return "at_launch_floor";
    case Flag::LaunchBound: return "launch_bound";
    case Flag::TooFewSamples: return "too_few_samples";
    case Flag::HighVariance: return "high_variance";
    case Flag::NoWarmup: return "no_warmup";
    case Flag::OutputUnverified: return "output_unverified";
    case Flag::FirstCallOutlier: return "first_call_outlier";
  }
  return "?";
}

const char* to_string(Severity s) {
  switch (s) {
    case Severity::Info: return "info";
    case Severity::Warning: return "warning";
    case Severity::Invalid: return "invalid";
  }
  return "?";
}

Severity severity_of(Flag f) {
  switch (f) {
    case Flag::NoSamples:
    case Flag::ImpossibleBandwidth:
    case Flag::ImpossibleFlops:
    case Flag::WallclockBelowDeviceTime:
    case Flag::ClockDrift:
    case Flag::ThermalOrHwThrottle:
    case Flag::ChecksumFailed:
      return Severity::Invalid;
    case Flag::ClocksUnavailable:
    case Flag::PowerCapThrottle:
    case Flag::AtLaunchFloor:
    case Flag::LaunchBound:
    case Flag::TooFewSamples:
    case Flag::HighVariance:
    case Flag::NoWarmup:
    case Flag::OutputUnverified:
      return Severity::Warning;
    case Flag::FirstCallOutlier:
      return Severity::Info;
  }
  return Severity::Invalid;
}

namespace {

template <typename... Args>
std::string fmt(const char* f, Args... args) {
  char buf[512];
  std::snprintf(buf, sizeof buf, f, args...);
  return buf;
}

void add(std::vector<Finding>& out, Flag f, std::string detail) {
  out.push_back({f, severity_of(f), std::move(detail)});
}

}  // namespace

std::vector<Finding> audit(const AuditInput& in, const AuditThresholds& th) {
  std::vector<Finding> out;
  const RawSamples& s = *in.samples;

  if (s.device_ms.empty()) {
    add(out, Flag::NoSamples, "no steady-state samples were collected");
    return out;
  }
  const Percentiles dev = summarize(s.device_ms);

  // --- Physical impossibility -------------------------------------------
  // Judged on the fastest sample: if any single sample beats physics, the
  // measurement is wrong regardless of what the median says.
  if (dev.min > 0 && in.bytes > 0 && in.dram_gbps_theoretical > 0) {
    const double gbps = static_cast<double>(in.bytes) / (dev.min / 1e3) / 1e9;
    const double limit = in.dram_gbps_theoretical * th.peak_tolerance;
    if (gbps > limit) {
      add(out, Flag::ImpossibleBandwidth,
          fmt("fastest sample implies %.1f GB/s of DRAM traffic, above the %.1f GB/s "
              "theoretical peak. Either the work was eliminated, or the data was "
              "L2-resident and this is not a DRAM bandwidth number.",
              gbps, in.dram_gbps_theoretical));
    }
  }
  if (dev.min > 0 && in.flops > 0 && in.peak_gflops_max_clock > 0) {
    const double gflops = static_cast<double>(in.flops) / (dev.min / 1e3) / 1e9;
    if (gflops > in.peak_gflops_max_clock * th.peak_tolerance) {
      add(out, Flag::ImpossibleFlops,
          fmt("fastest sample implies %.0f GFLOP/s, above the %.0f GFLOP/s theoretical "
              "peak. The declared work did not all execute.",
              gflops, in.peak_gflops_max_clock));
    }
  }

  // --- Timing consistency -------------------------------------------------
  // wall_ms is a host interval that starts before record(start) and ends after
  // sync(stop), so it encloses the device interval. Shorter means the host
  // timer stopped before the work finished.
  if (!s.wall_ms.empty() && s.wall_ms.size() == s.device_ms.size()) {
    std::size_t bad = 0;
    double worst = 0;
    for (std::size_t i = 0; i < s.device_ms.size(); ++i) {
      const double gap = s.device_ms[i] - s.wall_ms[i];
      if (gap > th.wall_tolerance_ms) {
        ++bad;
        worst = std::max(worst, gap);
      }
    }
    if (bad > 0) {
      add(out, Flag::WallclockBelowDeviceTime,
          fmt("%zu of %zu samples have a host wall-clock interval shorter than the device "
              "event interval (worst by %.4f ms). The host timer did not wait for the "
              "device; it measured launch, not execution.",
              bad, s.device_ms.size(), worst));
    }
  }

  // --- Clocks ---------------------------------------------------------------
  const ClockReport* c = in.clocks;
  if (c == nullptr || !c->available) {
    add(out, Flag::ClocksUnavailable,
        "SM/memory clocks could not be observed; throttling during the run cannot be ruled out");
  } else if (!c->steady.empty()) {
    if (c->max_drift() > in.clock_drift_tol) {
      add(out, Flag::ClockDrift,
          fmt("clocks moved during steady state: SM %u-%u MHz (%.1f%%), memory %u-%u MHz "
              "(%.1f%%); tolerance %.1f%%",
              c->sm_min, c->sm_max, 100 * c->sm_drift, c->mem_min, c->mem_max,
              100 * c->mem_drift, 100 * in.clock_drift_tol));
    }
    if (c->throttle_union & throttle::kThermalOrHardware) {
      add(out, Flag::ThermalOrHwThrottle,
          "NVML reported " + throttle::describe(c->throttle_union & throttle::kThermalOrHardware) +
              " during steady state");
    }
    if (c->throttle_union & throttle::kSwPowerCap) {
      add(out, Flag::PowerCapThrottle,
          fmt("clocks were limited by the power cap during steady state (SM median %u MHz); "
              "compare only against runs at the same power limit",
              c->sm_median));
    }
  }

  // --- Output ---------------------------------------------------------------
  if (!in.checksum.verifiable) {
    add(out, Flag::OutputUnverified,
        "workload has no reference output; elimination of work is only caught if it "
        "produces an impossible rate");
  } else if (!in.checksum.passed) {
    add(out, Flag::ChecksumFailed, "output did not match reference: " + in.checksum.note);
  }

  // --- Distribution ---------------------------------------------------------
  if (in.warmup_iters == 0) {
    add(out, Flag::NoWarmup,
        "no warmup: early steady-state samples may include lazy module loading and clock ramp");
  }
  if (s.first_device_ms > 0 && dev.p50 > 0 &&
      s.first_device_ms > th.first_call_outlier_ratio * dev.p50) {
    add(out, Flag::FirstCallOutlier,
        fmt("first call took %.4f ms on device (%.1fx the steady-state p50 of %.4f ms); "
            "excluded from percentiles",
            s.first_device_ms, s.first_device_ms / dev.p50, dev.p50));
  }
  if (dev.n < th.min_samples_for_p99) {
    add(out, Flag::TooFewSamples,
        fmt("%zu samples; p99 needs at least %zu to be more than the maximum", dev.n,
            th.min_samples_for_p99));
  }
  if (dev.p50 > 0 && dev.p90 / dev.p50 > th.high_variance_ratio) {
    add(out, Flag::HighVariance,
        fmt("p90/p50 = %.2f (threshold %.2f)", dev.p90 / dev.p50, th.high_variance_ratio));
  }

  // --- Launch overhead ------------------------------------------------------
  if (in.mode == Mode::Latency && !s.host_enqueue_ms.empty()) {
    const Percentiles host = summarize(s.host_enqueue_ms);
    if (dev.p50 > 0 && host.p50 >= th.launch_bound_ratio * dev.p50) {
      add(out, Flag::LaunchBound,
          fmt("host enqueue p50 %.4f ms is %.0f%% of device p50 %.4f ms; device time "
              "includes the GPU waiting for the host to submit work",
              host.p50, 100 * host.p50 / dev.p50, dev.p50));
    }
  }
  if ((in.bytes > 0 || in.flops > 0) && in.launch_floor_ms > 0 &&
      dev.p50 <= th.launch_floor_margin * in.launch_floor_ms) {
    add(out, Flag::AtLaunchFloor,
        fmt("p50 %.4f ms is within %.0f%% of an empty kernel (%.4f ms); this measures launch "
            "overhead, not the declared work",
            dev.p50, 100 * (th.launch_floor_margin - 1), in.launch_floor_ms));
  }

  return out;
}

}  // namespace trtbench
