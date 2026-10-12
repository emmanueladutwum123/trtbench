#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include "fixtures.hpp"
#include "trtbench/json.hpp"

using namespace trtbench;
using trtbench::testing::clean_copy;

namespace {
RunReport sample_report() {
  RunReport rep;
  rep.env.gpu_name = "SYNTHETIC (unit test fixture, not a measurement)";
  rep.env.cc_major = 8;
  rep.env.cc_minor = 9;
  rep.env.git_sha = build_git_sha();
  rep.peaks = trtbench::testing::l4_peaks();
  rep.launch_floor_latency_ms = 0.004;
  rep.results.push_back(finalize(clean_copy()));
  auto bad = clean_copy();
  bad.name = "copy_impossible";
  for (auto& d : bad.raw.device_ms) d /= 10;
  rep.results.push_back(finalize(bad));
  return rep;
}
}  // namespace

TEST(Json, ReportHasTheFieldsDownstreamToolsRead) {
  const auto j = to_json(sample_report());
  EXPECT_EQ(j["schema_version"], 1);
  EXPECT_FALSE(j["all_trustworthy"].get<bool>());
  const auto& r0 = j["results"][0];
  for (const char* k : {"name", "precision", "config", "device_ms", "host_enqueue_ms",
                        "first_call_device_ms", "clocks", "roofline", "findings",
                        "trustworthy", "device_samples_ms"}) {
    EXPECT_TRUE(r0.contains(k)) << k;
  }
  EXPECT_TRUE(r0["trustworthy"].get<bool>());
  EXPECT_FALSE(j["results"][1]["trustworthy"].get<bool>());
  EXPECT_EQ(j["results"][1]["findings"][0]["flag"], "impossible_bandwidth");
  EXPECT_EQ(r0["roofline"]["bound"], "memory");
}

TEST(Json, SamplesCanBeOmitted) {
  const auto j = to_json(sample_report(), /*include_samples=*/false);
  EXPECT_FALSE(j["results"][0].contains("device_samples_ms"));
}

TEST(Json, ComparisonSerializesRefusal) {
  const auto rep = sample_report();
  const auto j = to_json(compare(rep.results[0], rep.results[1]));
  EXPECT_EQ(j["status"], "untrustworthy");
  EXPECT_TRUE(j["accuracy"].is_null());
}

// Writes a schema sample for the Python tests to consume, so the C++ writer
// and the Python reader cannot drift apart unnoticed.
TEST(Json, WriteSampleForPythonTests) {
  const char* path = std::getenv("TRTBENCH_SAMPLE_JSON_OUT");
  if (path == nullptr) GTEST_SKIP() << "TRTBENCH_SAMPLE_JSON_OUT not set";
  std::ofstream(path) << to_json(sample_report()).dump(2);
  std::ifstream in(path);
  EXPECT_TRUE(in.good());
}
