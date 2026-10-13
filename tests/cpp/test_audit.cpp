#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "trtbench/result.hpp"

using namespace trtbench;
using trtbench::testing::clean_copy;

namespace {
Result run(FinalizeInput in) { return finalize(std::move(in)); }

std::string dump(const Result& r) {
  std::string s;
  for (const auto& f : r.findings) s += std::string(to_string(f.flag)) + ": " + f.detail + "\n";
  return s;
}
}  // namespace

TEST(Audit, CleanRunIsTrustworthyWithNoWarnings) {
  const Result r = run(clean_copy());
  EXPECT_TRUE(r.trustworthy()) << dump(r);
  for (const auto& f : r.findings) EXPECT_EQ(f.severity, Severity::Info) << dump(r);
}

TEST(Audit, NoSamplesIsInvalid) {
  auto in = clean_copy();
  in.raw.device_ms.clear();
  in.raw.wall_ms.clear();
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::NoSamples));
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, BandwidthAboveDramPeakIsImpossible) {
  // Same declared bytes, 4x faster: ~1000 GB/s on a 300 GB/s part.
  auto in = clean_copy();
  for (auto& d : in.raw.device_ms) d /= 4;
  for (auto& w : in.raw.wall_ms) w /= 4;
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ImpossibleBandwidth)) << dump(r);
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, BandwidthWithinToleranceOfPeakIsAccepted) {
  // 1.04x theoretical is inside the 5% clock-rounding allowance.
  auto in = clean_copy();
  const double ms = static_cast<double>(in.bytes) / (in.peaks.dram_gbps_theoretical * 1.04e9) * 1e3;
  in.raw.device_ms.assign(300, ms);
  in.raw.wall_ms.assign(300, ms + 0.01);
  EXPECT_FALSE(run(in).has(Flag::ImpossibleBandwidth));
}

TEST(Audit, OneImpossibleSampleIsEnough) {
  auto in = clean_copy();
  in.raw.device_ms[17] = in.raw.device_ms[17] / 10;
  EXPECT_TRUE(run(in).has(Flag::ImpossibleBandwidth));
}

TEST(Audit, FlopsAbovePeakIsImpossible) {
  // Declare 1 TFLOP of FP32 work finishing in 1 ms on a ~30 TFLOPS part.
  auto in = clean_copy();
  in.bytes = 4096;
  in.flops = 1'000'000'000'000ull;
  in.raw.device_ms.assign(300, 1.0);
  in.raw.wall_ms.assign(300, 1.01);
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ImpossibleFlops)) << dump(r);
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, FlopsCheckSkippedWhenPeakUnknown) {
  auto in = clean_copy();
  in.precision = Precision::Mixed;  // no peak for mixed precision
  in.flops = 1'000'000'000'000ull;
  EXPECT_FALSE(run(in).has(Flag::ImpossibleFlops));
}

TEST(Audit, WallclockShorterThanDeviceTimeIsInvalid) {
  // A host timer that stopped at launch: 5 us of wall time around 2 ms of work.
  auto in = clean_copy();
  in.raw.wall_ms.assign(in.raw.device_ms.size(), 0.005);
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::WallclockBelowDeviceTime)) << dump(r);
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, WallclockWithinTimerResolutionIsAccepted) {
  auto in = clean_copy();
  for (std::size_t i = 0; i < in.raw.device_ms.size(); ++i)
    in.raw.wall_ms[i] = in.raw.device_ms[i] - 0.0005;  // 0.5 us under: event resolution
  EXPECT_FALSE(run(in).has(Flag::WallclockBelowDeviceTime));
}

TEST(Audit, ClockDriftAboveToleranceIsInvalid) {
  auto in = clean_copy();
  in.clocks = summarize_clocks(true, "fake", {},
                               {{0, 2040, 6251, 0, 70, 70}, {500, 1800, 6251, 0, 85, 70},
                                {1000, 1785, 6251, 0, 88, 70}});
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ClockDrift)) << dump(r);
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, ClockDriftWithinToleranceIsAccepted) {
  auto in = clean_copy();
  in.clocks = summarize_clocks(true, "fake", {},
                               {{0, 2040, 6251, 0, 70, 70}, {1000, 1995, 6251, 0, 72, 70}});
  EXPECT_FALSE(run(in).has(Flag::ClockDrift));  // 2.2%
}

TEST(Audit, ThermalThrottleIsInvalidEvenWithoutDrift) {
  auto in = clean_copy();
  in.clocks = summarize_clocks(true, "fake", {},
                               {{0, 2040, 6251, throttle::kHwThermal, 90, 70},
                                {100, 2040, 6251, 0, 90, 70}});
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ThermalOrHwThrottle));
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, PowerCapIsAWarningNotInvalid) {
  // Steady power-capped clocks are normal on a 72 W L4 under load; the drift
  // check is what catches them moving.
  auto in = clean_copy();
  in.clocks = summarize_clocks(true, "fake", {},
                               {{0, 1800, 6251, throttle::kSwPowerCap, 70, 72},
                                {100, 1800, 6251, throttle::kSwPowerCap, 70, 72}});
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::PowerCapThrottle));
  EXPECT_TRUE(r.trustworthy()) << dump(r);
}

TEST(Audit, UnobservableClocksAreAWarning) {
  auto in = clean_copy();
  in.clocks = summarize_clocks(false, "none", {}, {});
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ClocksUnavailable));
  EXPECT_TRUE(r.trustworthy());
}

TEST(Audit, ChecksumFailureIsInvalid) {
  auto in = clean_copy();
  in.checksum = {true, false, 0.0, "out[4096] = 0, expected 4096"};
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::ChecksumFailed));
  EXPECT_FALSE(r.trustworthy());
}

TEST(Audit, UnverifiableOutputIsAWarning) {
  auto in = clean_copy();
  in.checksum = {};
  EXPECT_TRUE(run(in).has(Flag::OutputUnverified));
}

TEST(Audit, FirstCallOutlierIsSurfacedAndExcluded) {
  auto in = clean_copy();
  in.raw.first_device_ms = 250.0;  // lazy module load + context work
  const Result r = run(in);
  EXPECT_TRUE(r.has(Flag::FirstCallOutlier));
  EXPECT_TRUE(r.trustworthy());
  EXPECT_DOUBLE_EQ(r.first_call_device_ms, 250.0);
  EXPECT_LT(r.device_ms.max, 250.0);  // not averaged into the distribution
}

TEST(Audit, NoWarmupIsAWarning) {
  auto in = clean_copy();
  in.config.warmup_iters = 0;
  EXPECT_TRUE(run(in).has(Flag::NoWarmup));
}

TEST(Audit, TooFewSamplesForP99) {
  auto in = clean_copy();
  in.raw.device_ms.resize(50);
  in.raw.wall_ms.resize(50);
  in.raw.host_enqueue_ms.resize(50);
  EXPECT_TRUE(run(in).has(Flag::TooFewSamples));
}

TEST(Audit, HighVariance) {
  auto in = clean_copy();
  for (std::size_t i = 0; i < in.raw.device_ms.size(); i += 5) {
    in.raw.device_ms[i] *= 2;
    in.raw.wall_ms[i] = in.raw.device_ms[i] + 0.01;
  }
  EXPECT_TRUE(run(in).has(Flag::HighVariance));
}

TEST(Audit, LaunchBoundInLatencyMode) {
  auto in = clean_copy();
  in.bytes = 4096;
  in.raw.device_ms.assign(300, 0.010);
  in.raw.wall_ms.assign(300, 0.020);
  in.raw.host_enqueue_ms.assign(300, 0.009);
  EXPECT_TRUE(run(in).has(Flag::LaunchBound));
}

TEST(Audit, LaunchBoundNotJudgedInThroughputMode) {
  auto in = clean_copy();
  in.config.mode = Mode::Throughput;
  in.bytes = 4096;
  in.raw.device_ms.assign(300, 0.010);
  in.raw.wall_ms.assign(300, 0.020);
  in.raw.host_enqueue_ms.assign(300, 0.009);
  EXPECT_FALSE(run(in).has(Flag::LaunchBound));
}

TEST(Audit, AtLaunchFloorWhenIndistinguishableFromEmptyKernel) {
  auto in = clean_copy();
  in.bytes = 4096;
  in.raw.device_ms.assign(300, 0.0042);
  in.raw.wall_ms.assign(300, 0.010);
  in.raw.host_enqueue_ms.assign(300, 0.001);
  EXPECT_TRUE(run(in).has(Flag::AtLaunchFloor));
}

TEST(Audit, EmptyKernelItselfIsNotAtLaunchFloor) {
  // No declared work, so being at the floor is the expected answer.
  auto in = clean_copy();
  in.bytes = 0;
  in.raw.device_ms.assign(300, 0.004);
  EXPECT_FALSE(run(in).has(Flag::AtLaunchFloor));
}
