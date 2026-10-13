#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "trtbench/traps.hpp"

using namespace trtbench;
using trtbench::testing::clean_copy;

TEST(Traps, EverySyntheticTrapPasses) {
  const auto rows = run_synthetic_traps();
  ASSERT_EQ(rows.size(), synthetic_trap_specs().size());
  for (const auto& t : rows) {
    EXPECT_EQ(t.outcome, TrapOutcome::Pass)
        << t.spec.id << " " << t.spec.name << "\n  wrong: " << t.wrong
        << "\n  control: " << t.control << "\n  " << t.detail;
  }
}

TEST(Traps, WrongBenchmarkThatIsNotFlaggedFails) {
  const Result clean = finalize(clean_copy());
  const auto t = judge_flag({"X", "x", "clock_drift"}, "unit", Flag::ClockDrift, clean, nullptr);
  EXPECT_EQ(t.outcome, TrapOutcome::Fail);
}

TEST(Traps, FlaggingTheControlTooFails) {
  // A harness that flags everything would "catch" every wrong benchmark. The
  // control is what stops that from counting as a pass.
  auto in = clean_copy();
  for (auto& d : in.raw.device_ms) d /= 10;
  const Result impossible = finalize(in);
  const auto t = judge_flag({"X", "x", "impossible_bandwidth"}, "unit", Flag::ImpossibleBandwidth,
                            impossible, &impossible);
  EXPECT_EQ(t.outcome, TrapOutcome::Fail);
  EXPECT_NE(t.detail.find("control"), std::string::npos);
}

TEST(Traps, CaughtWithCleanControlPasses) {
  auto in = clean_copy();
  for (auto& d : in.raw.device_ms) d /= 10;
  const Result wrong = finalize(in);
  const Result control = finalize(clean_copy());
  const auto t = judge_flag({"X", "x", "impossible_bandwidth"}, "unit", Flag::ImpossibleBandwidth,
                            wrong, &control);
  EXPECT_EQ(t.outcome, TrapOutcome::Pass);
  EXPECT_NE(t.detail.find("GB/s"), std::string::npos);  // carries the auditor's explanation
}

TEST(Traps, SkipsAreNotFailures) {
  auto rows = skipped_traps(gpu_trap_specs(), "gpu", "no device");
  EXPECT_TRUE(no_failures(rows));
  rows.push_back(run_synthetic_traps()[0]);
  EXPECT_TRUE(no_failures(rows));
  rows.back().outcome = TrapOutcome::Fail;
  EXPECT_FALSE(no_failures(rows));
}

TEST(Traps, TableReportsCounts) {
  auto rows = run_synthetic_traps();
  for (auto& t : skipped_traps(gpu_trap_specs(), "gpu", "no device")) rows.push_back(t);
  const std::string table = render_trap_table(rows);
  EXPECT_NE(table.find("6 passed, 0 failed, 6 skipped"), std::string::npos) << table;
}
