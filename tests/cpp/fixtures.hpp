#pragma once

// Synthetic measurements shaped like a real L4 run, used to drive the pure
// audit/finalize code. Nothing here pretends to be a measured result.

#include <cstdint>
#include <vector>

#include "trtbench/result.hpp"

namespace trtbench::testing {

// NVIDIA L4 datasheet inputs: 58 SMs, 2040 MHz boost, 6251 MHz GDDR6, 192-bit.
inline DevicePeaks l4_peaks() {
  DevicePeaks p = theoretical_peaks(8, 9, 58, 2040, 6251000.0, 192);
  p.dram_gbps_measured = 280.0;
  return p;
}

inline ClockReport steady_clocks(unsigned sm = 2040, unsigned mem = 6251, int n = 10) {
  std::vector<ClockSnapshot> s;
  for (int i = 0; i < n; ++i) s.push_back({i * 100.0, sm, mem, 0, 60.0, 70.0});
  return summarize_clocks(true, "fake", {0, 210, 405, throttle::kGpuIdle, 40, 15}, s);
}

// Deterministic jitter: +/- 1% around `center`, no randomness in tests.
inline std::vector<double> around(double center, std::size_t n) {
  std::vector<double> v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double j = (static_cast<double>(i % 21) - 10.0) / 1000.0;
    v.push_back(center * (1.0 + j));
  }
  return v;
}

// A 256 MiB device copy (read + write) at ~250 GB/s: physically plausible on L4.
inline FinalizeInput clean_copy() {
  FinalizeInput in;
  in.name = "copy_256MiB";
  in.precision = Precision::FP32;
  in.config.mode = Mode::Latency;
  in.bytes = 2ull * 256 * 1024 * 1024;
  in.flops = 0;
  const double ms = static_cast<double>(in.bytes) / 250e9 * 1e3;
  in.raw.device_ms = around(ms, 300);
  in.raw.host_enqueue_ms = around(0.005, 300);
  for (double d : in.raw.device_ms) in.raw.wall_ms.push_back(d + 0.01);
  in.raw.first_device_ms = ms * 1.1;
  in.raw.first_host_ms = 0.02;
  in.raw.first_wall_ms = ms * 1.1 + 0.03;
  in.clocks = steady_clocks();
  in.peaks = l4_peaks();
  in.launch_floor_ms = 0.004;
  in.checksum = {true, true, 1.0, "sampled 4096 positions"};
  return in;
}

}  // namespace trtbench::testing
