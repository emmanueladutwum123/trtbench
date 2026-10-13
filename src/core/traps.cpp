#include "trtbench/traps.hpp"

#include <algorithm>
#include <cstdio>

namespace trtbench {

const char* to_string(TrapOutcome o) {
  switch (o) {
    case TrapOutcome::Pass: return "PASS";
    case TrapOutcome::Fail: return "FAIL";
    case TrapOutcome::Skip: return "SKIP";
  }
  return "?";
}

const std::vector<TrapSpec>& synthetic_trap_specs() {
  static const std::vector<TrapSpec> specs = {
      {"T1", "result never used (dead-code-eliminated kernel)", "impossible_flops"},
      {"T2", "host timer stopped without synchronize", "wallclock_below_device_time"},
      {"T3", "no warmup: first-call cost", "first_call_outlier, excluded from percentiles"},
      {"T4", "clocks drop during the run", "clock_drift"},
      {"T4t", "thermal slowdown at steady clocks", "thermal_or_hw_throttle"},
      {"T5", "FP16 vs FP32 speedup without accuracy", "comparison refused"},
  };
  return specs;
}

const std::vector<TrapSpec>& gpu_trap_specs() {
  static const std::vector<TrapSpec> specs = {
      {"T1", "result never used (dead-code-eliminated kernel)", "impossible_flops"},
      {"T2", "host timer stopped without synchronize", "wallclock_below_device_time"},
      {"T3", "no warmup: first-call cost (simulated autotuning)",
       "first_call_outlier, excluded from percentiles"},
      {"T4", "clocks drop during the run (scripted clock readings)", "clock_drift"},
      {"T4r", "clocks drop during the run (real, nvidia-smi -lgc)", "clock_drift"},
      {"T5", "FP16 vs FP32 speedup without accuracy", "comparison refused"},
  };
  return specs;
}

std::string verdict(const Result& r) {
  std::string flags, info;
  for (const auto& f : r.findings) {
    std::string& dst = f.severity == Severity::Info ? info : flags;
    if (!dst.empty()) dst += ", ";
    dst += to_string(f.flag);
  }
  std::string v = r.trustworthy() ? "accepted" : "REJECTED";
  if (!flags.empty()) v += ": " + flags;
  if (!info.empty()) v += " (info: " + info + ")";
  return v;
}

TrapResult judge_flag(const TrapSpec& spec, const std::string& layer, Flag flag,
                      const Result& wrong, const Result* control) {
  TrapResult t;
  t.spec = spec;
  t.layer = layer;
  t.wrong = verdict(wrong);
  t.control = control ? verdict(*control) : "none";

  const bool caught = wrong.has(flag);
  const bool false_positive = control != nullptr && control->has(flag);
  if (!caught) {
    t.outcome = TrapOutcome::Fail;
    t.detail = std::string("harness did not report ") + to_string(flag) + " for the wrong benchmark";
  } else if (false_positive) {
    t.outcome = TrapOutcome::Fail;
    t.detail = std::string("control was also flagged ") + to_string(flag) +
               ": the check cannot tell right from wrong";
  } else {
    t.outcome = TrapOutcome::Pass;
    for (const auto& f : wrong.findings)
      if (f.flag == flag) t.detail = f.detail;
  }
  return t;
}

std::vector<TrapResult> skipped_traps(const std::vector<TrapSpec>& specs,
                                      const std::string& layer, const std::string& why) {
  std::vector<TrapResult> out;
  for (const auto& s : specs) {
    TrapResult t;
    t.spec = s;
    t.layer = layer;
    t.outcome = TrapOutcome::Skip;
    t.wrong = t.control = "-";
    t.detail = why;
    out.push_back(t);
  }
  return out;
}

bool no_failures(const std::vector<TrapResult>& rows) {
  return std::none_of(rows.begin(), rows.end(),
                      [](const TrapResult& t) { return t.outcome == TrapOutcome::Fail; });
}

std::string render_trap_table(const std::vector<TrapResult>& rows) {
  std::string out;
  char buf[512];
  std::snprintf(buf, sizeof buf, "%-5s %-10s %-52s %-6s\n", "ID", "LAYER", "TRAP", "RESULT");
  out += buf;
  out += std::string(76, '-') + "\n";
  int pass = 0, fail = 0, skip = 0;
  for (const auto& t : rows) {
    std::snprintf(buf, sizeof buf, "%-5s %-10s %-52s %-6s\n", t.spec.id.c_str(), t.layer.c_str(),
                  t.spec.name.c_str(), to_string(t.outcome));
    out += buf;
    if (t.outcome != TrapOutcome::Skip) {
      out += "      expect:  " + t.spec.expect + "\n";
      out += "      wrong:   " + t.wrong + "\n";
      out += "      control: " + t.control + "\n";
    }
    if (!t.detail.empty()) out += "      " + t.detail + "\n";
    (t.outcome == TrapOutcome::Pass ? pass : t.outcome == TrapOutcome::Fail ? fail : skip)++;
  }
  std::snprintf(buf, sizeof buf, "\n%d passed, %d failed, %d skipped\n", pass, fail, skip);
  out += buf;
  return out;
}

nlohmann::json to_json(const TrapResult& t) {
  return {{"id", t.spec.id},     {"name", t.spec.name},     {"expect", t.spec.expect},
          {"layer", t.layer},    {"outcome", to_string(t.outcome)},
          {"wrong", t.wrong},    {"control", t.control},    {"detail", t.detail}};
}

}  // namespace trtbench
