#pragma once

// The trap suite: deliberately wrong benchmarks that the harness must reject.
//
// Each trap pairs a WRONG benchmark with a CONTROL, a correct version of the
// same measurement. A trap passes only if the wrong one is flagged AND the
// control is not flagged for the same reason; a harness that flags everything
// fails its own self-test.
//
// Two layers run the same traps:
//   synthetic  pure CPU: the auditor is fed the data shape each wrong
//              benchmark produces. Runs everywhere, including CI.
//   gpu        the wrong benchmark really executes on the device and the
//              harness must catch it end to end. Skipped, with the reason,
//              when there is no device.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trtbench/result.hpp"

namespace trtbench {

enum class TrapOutcome { Pass, Fail, Skip };
const char* to_string(TrapOutcome o);

struct TrapSpec {
  std::string id;
  std::string name;
  std::string expect;  // what the harness must report
};

// Traps that run in each layer, in display order.
const std::vector<TrapSpec>& synthetic_trap_specs();
const std::vector<TrapSpec>& gpu_trap_specs();

struct TrapResult {
  TrapSpec spec;
  std::string layer;  // "synthetic" | "gpu"
  TrapOutcome outcome = TrapOutcome::Skip;
  std::string wrong;    // what the harness said about the wrong benchmark
  std::string control;  // what it said about the control ("none" if there is none)
  std::string detail;
};

// Pass iff `wrong` carries `flag` and `control` (when given) does not.
TrapResult judge_flag(const TrapSpec& spec, const std::string& layer, Flag flag,
                      const Result& wrong, const Result* control);

// Short human summary of a result's findings, e.g. "REJECTED: impossible_flops".
std::string verdict(const Result& r);

std::vector<TrapResult> run_synthetic_traps();
std::vector<TrapResult> skipped_traps(const std::vector<TrapSpec>& specs,
                                      const std::string& layer, const std::string& why);

bool no_failures(const std::vector<TrapResult>& rows);  // Skip is not a failure
std::string render_trap_table(const std::vector<TrapResult>& rows);
nlohmann::json to_json(const TrapResult& r);

}  // namespace trtbench
