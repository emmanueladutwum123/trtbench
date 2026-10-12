#include <gtest/gtest.h>

#include "trtbench/roofline.hpp"

using namespace trtbench;

// Datasheet cross-checks: the arch table must reproduce NVIDIA's published
// peaks for real parts, within 1%.

TEST(TheoreticalPeaks, L4MatchesDatasheet) {
  // L4: 300 GB/s, 30.3 TFLOPS FP32, 121 TFLOPS FP16 tensor (dense).
  const auto p = theoretical_peaks(8, 9, 58, 2040, 6251000.0, 192);
  EXPECT_NEAR(p.dram_gbps_theoretical, 300.0, 3.0);
  EXPECT_NEAR(p.fp32_gflops, 30300.0, 303.0);
  EXPECT_NEAR(p.fp16_tc_gflops, 121000.0, 1210.0);
}

TEST(TheoreticalPeaks, A100_40GBMatchesDatasheet) {
  // A100 40GB SXM: 1555 GB/s, 19.5 TFLOPS FP32, 312 TFLOPS FP16 tensor.
  const auto p = theoretical_peaks(8, 0, 108, 1410, 1215000.0, 5120);
  EXPECT_NEAR(p.dram_gbps_theoretical, 1555.0, 15.6);
  EXPECT_NEAR(p.fp32_gflops, 19500.0, 195.0);
  EXPECT_NEAR(p.fp16_tc_gflops, 312000.0, 3120.0);
}

TEST(TheoreticalPeaks, A10MatchesDatasheet) {
  // A10: 600 GB/s, 31.2 TFLOPS FP32, 125 TFLOPS FP16 tensor.
  const auto p = theoretical_peaks(8, 6, 72, 1695, 6251000.0, 384);
  EXPECT_NEAR(p.dram_gbps_theoretical, 600.0, 6.0);
  EXPECT_NEAR(p.fp32_gflops, 31200.0, 312.0);
  EXPECT_NEAR(p.fp16_tc_gflops, 125000.0, 1250.0);
}

TEST(TheoreticalPeaks, UnknownArchHasNoComputePeak) {
  const auto p = theoretical_peaks(6, 1, 28, 1600, 5000000.0, 352);
  EXPECT_EQ(p.fp32_gflops, 0.0);
  EXPECT_GT(p.dram_gbps_theoretical, 0.0);
  EXPECT_NE(p.source.find("unknown"), std::string::npos);
}

TEST(ComputePeak, ScalesToRunClock) {
  const auto p = theoretical_peaks(8, 9, 58, 2040, 6251000.0, 192);
  EXPECT_NEAR(compute_peak_gflops(p, Precision::FP32, 1020), p.fp32_gflops / 2, 1e-6);
  EXPECT_DOUBLE_EQ(compute_peak_gflops(p, Precision::FP32, 0), p.fp32_gflops);
  EXPECT_DOUBLE_EQ(compute_peak_gflops(p, Precision::FP16, 0), p.fp16_tc_gflops);
  EXPECT_DOUBLE_EQ(compute_peak_gflops(p, Precision::INT8, 0), 0.0);
}

TEST(Place, CopyIsMemoryBound) {
  // 1 GB moved in 4 ms = 250 GB/s; no FLOPs.
  const auto r = place(1'000'000'000, 0, 4.0, 280.0, 300.0, 30000.0);
  EXPECT_NEAR(r.achieved_gbps, 250.0, 1e-9);
  EXPECT_NEAR(r.pct_bw_peak, 100.0 * 250 / 280, 1e-9);
  EXPECT_NEAR(r.pct_bw_theoretical, 100.0 * 250 / 300, 1e-9);
  EXPECT_EQ(r.intensity, 0.0);
  EXPECT_EQ(r.bound, Bound::Memory);
}

TEST(Place, RidgeSeparatesMemoryFromCompute) {
  // Ridge = 30000 / 300 = 100 FLOP/byte.
  const auto lo = place(1000, 99'000, 1.0, 300.0, 300.0, 30000.0);
  const auto hi = place(1000, 101'000, 1.0, 300.0, 300.0, 30000.0);
  EXPECT_DOUBLE_EQ(lo.ridge, 100.0);
  EXPECT_EQ(lo.bound, Bound::Memory);
  EXPECT_EQ(hi.bound, Bound::Compute);
}

TEST(Place, PureComputeAndNoWork) {
  EXPECT_EQ(place(0, 1000, 1.0, 300.0, 300.0, 30000.0).bound, Bound::Compute);
  EXPECT_EQ(place(0, 0, 1.0, 300.0, 300.0, 30000.0).bound, Bound::Unknown);
}

TEST(Place, UnknownComputePeakStillClassifiesPureMemoryOps) {
  EXPECT_EQ(place(1000, 0, 1.0, 300.0, 300.0, 0.0).bound, Bound::Memory);
  EXPECT_EQ(place(1000, 10, 1.0, 300.0, 300.0, 0.0).bound, Bound::Unknown);
}
