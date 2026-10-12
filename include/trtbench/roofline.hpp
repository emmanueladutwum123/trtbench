#pragma once

#include <cstdint>
#include <string>

#include "trtbench/config.hpp"

namespace trtbench {

enum class Bound { Memory, Compute, Unknown };
const char* to_string(Bound b);

// Per-SM throughput by architecture. CUDA exposes no "peak FLOP/s" query, so
// this table is the source of every compute peak; values are from NVIDIA's
// architecture whitepapers and are cross-checked in tests against datasheet
// TFLOPS for specific parts.
struct ArchInfo {
  bool known = false;
  int fp32_lanes_per_sm = 0;            // FP32 FMA units per SM
  int fp16_tc_flops_per_clk_per_sm = 0; // dense FP16 tensor-core FLOP/clk, FP32 accumulate,
                                        // datacenter rate (GeForce parts are half this)
};
ArchInfo arch_info(int cc_major, int cc_minor);

struct DevicePeaks {
  double dram_gbps_theoretical = 0;  // memory clock x bus width x 2 (DDR)
  double dram_gbps_measured = 0;     // large device-to-device copy, measured by this harness
  double fp32_gflops = 0;            // at peak_sm_clock_mhz
  double fp16_tc_gflops = 0;         // at peak_sm_clock_mhz
  unsigned peak_sm_clock_mhz = 0;
  std::string source;

  // Achievable reference for "% of peak": the measured copy if we have it.
  double dram_gbps_reference() const {
    return dram_gbps_measured > 0 ? dram_gbps_measured : dram_gbps_theoretical;
  }
};

DevicePeaks theoretical_peaks(int cc_major, int cc_minor, int sm_count,
                              unsigned peak_sm_clock_mhz, double mem_clock_khz,
                              int bus_width_bits);

// Compute peak relevant to an op of this precision, scaled to the SM clock
// observed during the run (a GPU locked or throttled to 1500 MHz cannot reach
// its 2040 MHz datasheet peak, and % of peak must not pretend otherwise).
// Returns 0 when unknown (INT8 until Phase 2, unknown arch).
double compute_peak_gflops(const DevicePeaks& peaks, Precision p, unsigned run_sm_mhz);

struct RooflinePoint {
  double time_ms = 0;
  std::uint64_t bytes = 0;
  std::uint64_t flops = 0;
  double achieved_gbps = 0;
  double achieved_gflops = 0;
  double intensity = 0;          // FLOP per byte of DRAM traffic
  double ridge = 0;              // peak_gflops / peak_gbps
  double peak_gbps = 0;          // reference used for pct_bw
  double peak_gflops = 0;
  double pct_bw_peak = 0;        // of measured copy peak when available
  double pct_bw_theoretical = 0;
  double pct_compute_peak = 0;
  Bound bound = Bound::Unknown;
};

RooflinePoint place(std::uint64_t bytes, std::uint64_t flops, double time_ms,
                    double peak_gbps, double peak_gbps_theoretical, double peak_gflops);

}  // namespace trtbench
