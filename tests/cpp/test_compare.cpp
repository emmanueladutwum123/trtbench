#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "trtbench/compare.hpp"

using namespace trtbench;
using trtbench::testing::clean_copy;

namespace {
// Scales the clean 250 GB/s copy. Scale < 1 makes it faster; anything below ~0.83
// exceeds the L4 DRAM peak and is (correctly) rejected by the auditor, so the
// speedup tests slow the baseline down instead of speeding the candidate up.
Result with(Precision p, double scale) {
  auto in = clean_copy();
  in.precision = p;
  for (auto& d : in.raw.device_ms) d *= scale;
  for (auto& w : in.raw.wall_ms) w *= scale;
  return finalize(in);
}
}  // namespace

TEST(Compare, SamePrecisionReportsSpeedup) {
  const auto c = compare(with(Precision::FP32, 2.0), with(Precision::FP32, 4.0 / 3.0));
  ASSERT_EQ(c.status, ComparisonStatus::Ok) << c.reason;
  EXPECT_NEAR(c.speedup_p50, 1.5, 1e-9);
}

TEST(Compare, PrecisionChangeWithoutAccuracyIsRefused) {
  const auto c = compare(with(Precision::FP32, 2.0), with(Precision::FP16, 1.0));
  EXPECT_EQ(c.status, ComparisonStatus::Refused);
  EXPECT_EQ(c.speedup_p50, 0.0);  // no number leaks out of a refused comparison
  EXPECT_NE(c.reason.find("accuracy"), std::string::npos);
}

TEST(Compare, PrecisionChangeWithAccuracyIsReportedTogether) {
  const auto c = compare(with(Precision::FP32, 2.0), with(Precision::FP16, 1.0),
                         AccuracyDelta{"top1", 0.7613, 0.7609});
  ASSERT_EQ(c.status, ComparisonStatus::Ok) << c.reason;
  EXPECT_NEAR(c.speedup_p50, 2.0, 1e-9);
  ASSERT_TRUE(c.accuracy.has_value());
  EXPECT_NEAR(c.accuracy->delta(), -0.0004, 1e-12);
}

TEST(Compare, UntrustworthySideIsRefused) {
  auto bad = clean_copy();
  for (auto& d : bad.raw.device_ms) d /= 10;  // impossible bandwidth
  const auto c = compare(with(Precision::FP32, 1.0), finalize(bad));
  EXPECT_EQ(c.status, ComparisonStatus::Untrustworthy);
  EXPECT_EQ(c.speedup_p50, 0.0);
  EXPECT_NE(c.reason.find("impossible_bandwidth"), std::string::npos) << c.reason;
}

TEST(Compare, DifferentModesAreRefused) {
  auto t = clean_copy();
  t.config.mode = Mode::Throughput;
  const auto c = compare(with(Precision::FP32, 1.0), finalize(t));
  EXPECT_EQ(c.status, ComparisonStatus::Refused);
}
