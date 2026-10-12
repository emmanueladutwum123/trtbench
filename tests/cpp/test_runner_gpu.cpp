// Runs on a real GPU; every test skips when none is present. These check the
// runner's invariants end to end, not performance.

#include <gtest/gtest.h>

#include "trtbench/cuda/builtin_workloads.hpp"
#include "trtbench/cuda/device_context.hpp"
#include "trtbench/cuda/runner.hpp"

using namespace trtbench;
using namespace trtbench::cuda;

namespace {

class GpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string why;
    if (cuda_device_count(&why) == 0) {
      skip_reason_ = "no CUDA device: " + why;
      return;
    }
    ContextOptions o;
    ctx_ = DeviceContext::create(o).release();
  }
  static void TearDownTestSuite() {
    delete ctx_;
    ctx_ = nullptr;
  }
  void SetUp() override {
    if (ctx_ == nullptr) GTEST_SKIP() << skip_reason_;
  }

  static BenchConfig quick(Mode m = Mode::Latency) {
    BenchConfig c;
    c.mode = m;
    c.min_iters = 200;
    c.min_time_ms = 100;
    c.warmup_min_ms = 100;
    return c;
  }

  static DeviceContext* ctx_;
  static std::string skip_reason_;
};

DeviceContext* GpuTest::ctx_ = nullptr;
std::string GpuTest::skip_reason_;

std::string dump(const Result& r) {
  std::string s;
  for (const auto& f : r.findings) s += std::string(to_string(f.flag)) + ": " + f.detail + "\n";
  return s;
}

}  // namespace

TEST(GpuAvailability, ReportsCountWithoutThrowing) {
  std::string why;
  EXPECT_GE(cuda_device_count(&why), 0);
}

TEST_F(GpuTest, CalibrationProducesPlausibleFloorsAndPeak) {
  EXPECT_GT(ctx_->launch_floor_ms(Mode::Latency), 0.0);
  EXPECT_GT(ctx_->launch_floor_ms(Mode::Throughput), 0.0);
  ASSERT_GT(ctx_->peaks().dram_gbps_theoretical, 0.0);
  EXPECT_GT(ctx_->peaks().dram_gbps_measured, 0.0) << "copy calibration was rejected";
  EXPECT_LE(ctx_->peaks().dram_gbps_measured, ctx_->peaks().dram_gbps_theoretical * 1.05);
}

TEST_F(GpuTest, CopyIsCorrectMemoryBoundAndBelowPeak) {
  auto w = make_copy(std::size_t{256} << 20);
  const Result r = run(*w, quick(Mode::Throughput), *ctx_);
  EXPECT_TRUE(r.checksum.passed) << r.checksum.note;
  EXPECT_TRUE(r.trustworthy()) << dump(r);
  EXPECT_EQ(r.roofline.bound, Bound::Memory);
  EXPECT_LE(r.roofline.pct_bw_theoretical, 105.0);
}

TEST_F(GpuTest, FmaIsBitExactAndComputeBound) {
  auto w = make_fma(4096);
  const Result r = run(*w, quick(), *ctx_);
  EXPECT_TRUE(r.checksum.passed) << r.checksum.note;
  EXPECT_FALSE(r.has(Flag::ImpossibleFlops)) << dump(r);
  EXPECT_EQ(r.roofline.bound, Bound::Compute);
}

TEST_F(GpuTest, WallClockEnclosesDeviceTimeForEverySample) {
  auto w = make_copy(std::size_t{64} << 20);
  const Result r = run(*w, quick(), *ctx_);
  EXPECT_FALSE(r.has(Flag::WallclockBelowDeviceTime)) << dump(r);
  EXPECT_GE(r.wall_ms.p50, r.device_ms.p50);
}

TEST_F(GpuTest, FirstCallIsRecordedSeparately) {
  auto w = make_fma(256);
  const Result r = run(*w, quick(), *ctx_);
  EXPECT_GT(r.first_call_device_ms, 0.0);
  EXPECT_EQ(r.device_ms.n, r.device_samples_ms.size());
}

TEST_F(GpuTest, SteadyStateRespectsMinimums) {
  auto w = make_empty_kernel();
  BenchConfig c = quick();
  c.min_iters = 321;
  const Result r = run(*w, c, *ctx_);
  EXPECT_GE(r.device_ms.n, 321u);
}
