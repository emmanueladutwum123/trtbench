#include <gtest/gtest.h>

#include "trtbench/layer_profile.hpp"

using trtbench::LayerProfile;

namespace {
void run(LayerProfile& p, std::vector<std::pair<std::string, double>> layers) {
  p.begin_run();
  for (const auto& l : layers) p.add(l.first, l.second);
  p.end_run();
}
}  // namespace

TEST(LayerProfile, PerLayerMedianOverRunsInEngineOrder) {
  LayerProfile p;
  run(p, {{"conv", 1.0}, {"relu", 0.1}});
  run(p, {{"conv", 3.0}, {"relu", 0.3}});
  run(p, {{"conv", 2.0}, {"relu", 0.2}});
  const auto ls = p.layers();
  ASSERT_EQ(ls.size(), 2u);
  EXPECT_EQ(ls[0].name, "conv");
  EXPECT_DOUBLE_EQ(ls[0].ms.p50, 2.0);
  EXPECT_DOUBLE_EQ(ls[1].ms.p50, 0.2);
  EXPECT_EQ(ls[0].ms.n, 3u);
  EXPECT_TRUE(p.consistent());
  const auto tot = p.run_totals_ms();
  EXPECT_DOUBLE_EQ(tot[1], 3.3);
}

TEST(LayerProfile, RepeatedNameInOneRunIsKeptSeparateNotSummed) {
  LayerProfile p;
  run(p, {{"reformat", 0.5}, {"gemm", 1.0}, {"reformat", 0.7}});
  const auto ls = p.layers();
  ASSERT_EQ(ls.size(), 3u);
  EXPECT_EQ(ls[2].name, "reformat#2");
  EXPECT_DOUBLE_EQ(ls[2].ms.p50, 0.7);
}

TEST(LayerProfile, RunWithDifferentLayersIsFlaggedNotAveraged) {
  LayerProfile p;
  run(p, {{"a", 1.0}, {"b", 1.0}});
  run(p, {{"a", 1.0}});  // a lost callback
  EXPECT_FALSE(p.consistent());
  EXPECT_NE(p.inconsistency().find("1 layers"), std::string::npos);

  LayerProfile q;
  run(q, {{"a", 1.0}, {"b", 1.0}});
  run(q, {{"b", 1.0}, {"a", 1.0}});
  EXPECT_FALSE(q.consistent());
}

TEST(LayerProfile, CallbackOutsideARunIsAnError) {
  LayerProfile p;
  EXPECT_THROW(p.add("a", 1.0), std::logic_error);
  p.begin_run();
  EXPECT_THROW(p.begin_run(), std::logic_error);
}

TEST(LayerProfile, JsonCarriesConsistencyAndTotals) {
  LayerProfile p;
  run(p, {{"a", 1.0}, {"b", 2.0}});
  const auto j = to_json(p);
  EXPECT_EQ(j["runs"], 1);
  EXPECT_TRUE(j["consistent"].get<bool>());
  EXPECT_DOUBLE_EQ(j["sum_of_layers_ms"]["p50"].get<double>(), 3.0);
  EXPECT_EQ(j["layers"][1]["name"], "b");
}
