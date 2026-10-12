#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "trtbench/result.hpp"

using namespace trtbench;
using trtbench::testing::clean_copy;

TEST(Finalize, DeviceAndHostAreReportedSeparately) {
  const Result r = finalize(clean_copy());
  EXPECT_GT(r.device_ms.p50, 1.0);
  EXPECT_LT(r.host_enqueue_ms.p50, 0.01);
  EXPECT_EQ(r.device_ms.n, 300u);
  EXPECT_EQ(r.host_enqueue_ms.n, 300u);
}

TEST(Finalize, RooflineIsPlacedAtP50) {
  const Result r = finalize(clean_copy());
  EXPECT_DOUBLE_EQ(r.roofline.time_ms, r.device_ms.p50);
  EXPECT_NEAR(r.roofline.achieved_gbps, 250.0, 2.5);
  EXPECT_EQ(r.roofline.bound, Bound::Memory);
  // % of peak uses the measured copy peak (280), with theoretical alongside.
  EXPECT_NEAR(r.roofline.pct_bw_peak, 100.0 * r.roofline.achieved_gbps / 280.0, 1e-9);
  EXPECT_GT(r.roofline.pct_bw_theoretical, 0.0);
}

TEST(Finalize, ComputePeakUsesObservedClock) {
  auto in = clean_copy();
  in.flops = 1000;
  in.clocks = trtbench::testing::steady_clocks(1020);
  const Result r = finalize(in);
  EXPECT_NEAR(r.roofline.peak_gflops, in.peaks.fp32_gflops / 2, 1e-6);
}

TEST(Finalize, LatencyModeForcesOneLaunchPerSample) {
  auto in = clean_copy();
  in.config.launches_per_sample = 64;
  EXPECT_EQ(finalize(in).launches_per_sample, 1);
  in.config.mode = Mode::Throughput;
  EXPECT_EQ(finalize(in).launches_per_sample, 64);
}

TEST(Finalize, RawSamplesAreKept) {
  const Result r = finalize(clean_copy());
  EXPECT_EQ(r.device_samples_ms.size(), 300u);
}

TEST(Config, ValidateRejectsNonsense) {
  BenchConfig c;
  EXPECT_EQ(validate(c), "");
  c.max_iters = c.min_iters - 1;
  EXPECT_NE(validate(c), "");
  c = {};
  c.launches_per_sample = 0;
  EXPECT_NE(validate(c), "");
  c = {};
  c.clock_drift_tol = 0;
  EXPECT_NE(validate(c), "");
}
