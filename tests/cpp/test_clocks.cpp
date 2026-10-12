#include <gtest/gtest.h>

#include "trtbench/clocks.hpp"

using namespace trtbench;

namespace {
ClockSnapshot snap(unsigned sm, unsigned mem, std::uint64_t reasons = 0) {
  ClockSnapshot s;
  s.sm_mhz = sm;
  s.mem_mhz = mem;
  s.throttle_reasons = reasons;
  return s;
}
}  // namespace

TEST(Clocks, SteadyClocksHaveNoDrift) {
  const auto r = summarize_clocks(true, "t", {}, {snap(2040, 6251), snap(2040, 6251)});
  EXPECT_DOUBLE_EQ(r.sm_drift, 0.0);
  EXPECT_DOUBLE_EQ(r.mem_drift, 0.0);
  EXPECT_EQ(r.sm_median, 2040u);
}

TEST(Clocks, DriftIsRangeOverMax) {
  // 2040 -> 1785 MHz is a 12.5% drop.
  const auto r = summarize_clocks(true, "t", {},
                                  {snap(2040, 6251), snap(1900, 6251), snap(1785, 6251)});
  EXPECT_NEAR(r.sm_drift, (2040.0 - 1785.0) / 2040.0, 1e-12);
  EXPECT_EQ(r.sm_min, 1785u);
  EXPECT_EQ(r.sm_max, 2040u);
  EXPECT_EQ(r.sm_median, 1900u);
}

TEST(Clocks, DipInTheMiddleIsCaughtNotJustBeforeAfter) {
  // Before and after agree; a before/after comparison alone would miss this.
  const auto r = summarize_clocks(true, "t", {},
                                  {snap(2040, 6251), snap(1500, 6251), snap(2040, 6251)});
  EXPECT_GT(r.max_drift(), 0.25);
}

TEST(Clocks, PreWarmupIdleClockIsExcludedFromDrift) {
  // Idle at 210 MHz before warmup, then a steady 2040 MHz: not throttling.
  const auto r = summarize_clocks(true, "t", snap(210, 405, throttle::kGpuIdle),
                                  {snap(2040, 6251), snap(2040, 6251)});
  EXPECT_DOUBLE_EQ(r.max_drift(), 0.0);
  EXPECT_EQ(r.pre_warmup.sm_mhz, 210u);
}

TEST(Clocks, ThrottleReasonsAreUnioned) {
  const auto r = summarize_clocks(true, "t", {},
                                  {snap(2040, 6251, throttle::kSwPowerCap),
                                   snap(2040, 6251, throttle::kHwThermal)});
  EXPECT_EQ(r.throttle_union, throttle::kSwPowerCap | throttle::kHwThermal);
  EXPECT_EQ(throttle::describe(r.throttle_union), "sw_power_cap|hw_thermal");
}

TEST(Clocks, UnavailableSourceProducesEmptyReport) {
  const auto r = summarize_clocks(false, "none", {}, {});
  EXPECT_FALSE(r.available);
  EXPECT_DOUBLE_EQ(r.max_drift(), 0.0);
}

TEST(Clocks, NullSourceIsUnavailable) {
  NullClockSource n("no driver");
  EXPECT_FALSE(n.available());
  EXPECT_EQ(n.name(), "none (no driver)");
}
