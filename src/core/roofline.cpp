#include "trtbench/roofline.hpp"

namespace trtbench {

const char* to_string(Bound b) {
  switch (b) {
    case Bound::Memory: return "memory";
    case Bound::Compute: return "compute";
    case Bound::Unknown: return "unknown";
  }
  return "?";
}

ArchInfo arch_info(int major, int minor) {
  // fp32 lanes: Volta/Turing/GA100/GH100 = 64 per SM; GA10x/AD10x and later
  // consumer/inference parts = 128 per SM. GH100 = 128.
  // FP16 tensor FLOP/clk/SM (dense, FP32 accumulate, datacenter rate):
  //   sm_70 1024, sm_75 1024, sm_80 2048, sm_86 1024, sm_89 1024, sm_90 4096.
  const int cc = major * 10 + minor;
  switch (cc) {
    case 70: return {true, 64, 1024, 0};
    case 75: return {true, 64, 1024, 2048};
    case 80: return {true, 64, 2048, 4096};
    case 86: return {true, 128, 1024, 2048};
    case 87: return {true, 128, 1024, 2048};
    case 89: return {true, 128, 1024, 2048};
    case 90: return {true, 128, 4096, 8192};
    default: return {};
  }
}

DevicePeaks theoretical_peaks(int major, int minor, int sm_count,
                              unsigned peak_sm_clock_mhz, double mem_clock_khz,
                              int bus_width_bits) {
  DevicePeaks p;
  p.peak_sm_clock_mhz = peak_sm_clock_mhz;
  if (mem_clock_khz > 0 && bus_width_bits > 0) {
    // x2: DDR transfers per clock as reported by cudaDevAttrMemoryClockRate.
    p.dram_gbps_theoretical = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
  }
  const ArchInfo a = arch_info(major, minor);
  if (a.known && sm_count > 0 && peak_sm_clock_mhz > 0) {
    const double clk_ghz = peak_sm_clock_mhz / 1e3;
    p.fp32_gflops = sm_count * a.fp32_lanes_per_sm * 2.0 * clk_ghz;  // FMA = 2 FLOP
    p.fp16_tc_gflops = sm_count * a.fp16_tc_flops_per_clk_per_sm * clk_ghz;
    p.int8_tc_gops = sm_count * a.int8_tc_ops_per_clk_per_sm * clk_ghz;
    p.source = "arch table sm_" + std::to_string(major) + std::to_string(minor) +
               " x " + std::to_string(sm_count) + " SMs @ " +
               std::to_string(peak_sm_clock_mhz) + " MHz; DRAM = 2 x mem clock x bus width";
  } else {
    p.source = "compute peak unknown for sm_" + std::to_string(major) + std::to_string(minor) +
               "; DRAM = 2 x mem clock x bus width";
  }
  return p;
}

double compute_peak_gflops(const DevicePeaks& peaks, Precision prec, unsigned run_sm_mhz) {
  double at_peak = 0;
  switch (prec) {
    case Precision::FP32: at_peak = peaks.fp32_gflops; break;
    case Precision::FP16:
    case Precision::BF16: at_peak = peaks.fp16_tc_gflops; break;
    case Precision::INT8: at_peak = peaks.int8_tc_gops; break;
    case Precision::Mixed: return 0;
  }
  if (run_sm_mhz == 0 || peaks.peak_sm_clock_mhz == 0) return at_peak;
  return at_peak * run_sm_mhz / peaks.peak_sm_clock_mhz;
}

RooflinePoint place(std::uint64_t bytes, std::uint64_t flops, double time_ms,
                    double peak_gbps, double peak_gbps_theoretical, double peak_gflops) {
  RooflinePoint r;
  r.time_ms = time_ms;
  r.bytes = bytes;
  r.flops = flops;
  r.peak_gbps = peak_gbps;
  r.peak_gflops = peak_gflops;
  if (time_ms > 0) {
    const double s = time_ms / 1e3;
    r.achieved_gbps = static_cast<double>(bytes) / s / 1e9;
    r.achieved_gflops = static_cast<double>(flops) / s / 1e9;
  }
  if (bytes > 0) r.intensity = static_cast<double>(flops) / static_cast<double>(bytes);
  if (peak_gbps > 0) r.pct_bw_peak = 100.0 * r.achieved_gbps / peak_gbps;
  if (peak_gbps_theoretical > 0) r.pct_bw_theoretical = 100.0 * r.achieved_gbps / peak_gbps_theoretical;
  if (peak_gflops > 0) r.pct_compute_peak = 100.0 * r.achieved_gflops / peak_gflops;

  if (peak_gbps > 0 && peak_gflops > 0) {
    r.ridge = peak_gflops / peak_gbps;
    if (bytes == 0 && flops == 0) {
      r.bound = Bound::Unknown;
    } else if (bytes == 0) {
      r.bound = Bound::Compute;
    } else {
      r.bound = r.intensity < r.ridge ? Bound::Memory : Bound::Compute;
    }
  } else if (flops == 0 && bytes > 0) {
    r.bound = Bound::Memory;  // no compute peak needed to know this
  }
  return r;
}

}  // namespace trtbench
