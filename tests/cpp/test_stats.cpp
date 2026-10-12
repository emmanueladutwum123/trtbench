#include <gtest/gtest.h>

#include "trtbench/stats.hpp"

using namespace trtbench;

TEST(NearestRank, OneToHundred) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  EXPECT_EQ(nearest_rank(v, 0.50), 50);
  EXPECT_EQ(nearest_rank(v, 0.90), 90);
  EXPECT_EQ(nearest_rank(v, 0.99), 99);
  EXPECT_EQ(nearest_rank(v, 1.00), 100);
}

TEST(NearestRank, ReportsOnlyObservedValues) {
  // Interpolation would give 15 for the median of {10, 20}; nearest-rank
  // reports a value that was actually measured.
  EXPECT_EQ(nearest_rank({10, 20}, 0.5), 10);
}

TEST(NearestRank, SmallSampleP99IsTheMax) {
  // With 10 samples, p99 is the maximum. This is why the auditor flags
  // TooFewSamples below 100.
  std::vector<double> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(nearest_rank(v, 0.99), 10);
}

TEST(NearestRank, EmptyIsZero) { EXPECT_EQ(nearest_rank({}, 0.5), 0); }

TEST(Summarize, SortsUnsortedInput) {
  const Percentiles p = summarize({5, 1, 4, 2, 3});
  EXPECT_EQ(p.n, 5u);
  EXPECT_EQ(p.min, 1);
  EXPECT_EQ(p.max, 5);
  EXPECT_EQ(p.p50, 3);
  EXPECT_EQ(p.p99, 5);
}

TEST(Summarize, SingleSample) {
  const Percentiles p = summarize({7});
  EXPECT_EQ(p.p50, 7);
  EXPECT_EQ(p.p90, 7);
  EXPECT_EQ(p.p99, 7);
}

TEST(Summarize, Empty) {
  const Percentiles p = summarize({});
  EXPECT_EQ(p.n, 0u);
  EXPECT_EQ(p.p50, 0);
}
