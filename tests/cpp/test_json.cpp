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

namespace {
nlohmann::json samples_doc() {
  nlohmann::json snap = {{"t_ms", 0}, {"sm_mhz", 1200}, {"mem_mhz", 5000},
                         {"throttle_reasons", throttle::kSwPowerCap}, {"temp_c", 60}, {"power_w", 69}};
  std::vector<double> dev(150, 1.0), host(150, 0.05), wall(150, 1.02);
  return {{"schema", "trtbench-samples-1"},
          {"name", "torch_eager/resnet50_b1/fp32"},
          {"labels", {{"backend", "torch_eager"}, {"model", "resnet50"}}},
          {"precision", "fp32"},
          {"config", {{"mode", "latency"}, {"warmup_iters", 50}, {"warmup_min_ms", 250},
                      {"min_iters", 150}, {"min_time_ms", 100}, {"max_iters", 100000},
                      {"launches_per_sample", 1}, {"clock_sample_every", 25},
                      {"clock_drift_tol", 0.05}}},
          {"bytes", 102'000'000},
          {"flops", 8'200'000'000ull},
          {"raw", {{"device_ms", dev}, {"host_enqueue_ms", host}, {"wall_ms", wall},
                   {"first_device_ms", 30.0}, {"first_host_ms", 29.0}, {"first_wall_ms", 31.0}}},
          {"clocks", {{"available", true}, {"source", "pynvml"}, {"pre_warmup", snap},
                      {"steady", {snap, snap, snap}}}},
          {"checksum", {{"verifiable", true}, {"passed", true}, {"note", "top-1 matches"}}}};
}
}  // namespace

TEST(Ingest, PythonSamplesGoThroughTheSameAudit) {
  FinalizeInput in = samples_from_json(samples_doc());
  in.peaks = trtbench::testing::l4_peaks();
  const Result r = finalize(std::move(in));
  EXPECT_EQ(r.name, "torch_eager/resnet50_b1/fp32");
  EXPECT_EQ(r.labels.at("backend"), "torch_eager");
  EXPECT_EQ(r.device_ms.n, 150u);
  EXPECT_TRUE(r.has(Flag::PowerCapThrottle));
  EXPECT_TRUE(r.has(Flag::FirstCallOutlier));
  EXPECT_TRUE(r.trustworthy());
}

TEST(Ingest, ImpossibleIngestedSamplesAreRejected) {
  auto doc = samples_doc();
  doc["raw"]["device_ms"] = std::vector<double>(150, 0.001);  // 102 MB in 1 us
  doc["raw"]["wall_ms"] = std::vector<double>(150, 0.002);
  FinalizeInput in = samples_from_json(doc);
  in.peaks = trtbench::testing::l4_peaks();
  EXPECT_TRUE(finalize(std::move(in)).has(Flag::ImpossibleBandwidth));
}

TEST(Ingest, MalformedDocumentsAreRejected) {
  auto doc = samples_doc();
  doc["schema"] = "something-else";
  EXPECT_THROW(samples_from_json(doc), std::invalid_argument);
  doc = samples_doc();
  doc["raw"]["wall_ms"] = std::vector<double>(3, 1.0);
  EXPECT_THROW(samples_from_json(doc), std::invalid_argument);
  doc = samples_doc();
  doc["precision"] = "fp8";
  EXPECT_THROW(samples_from_json(doc), std::invalid_argument);
  doc = samples_doc();
  doc.erase("flops");
  EXPECT_THROW(samples_from_json(doc), std::invalid_argument);
}
