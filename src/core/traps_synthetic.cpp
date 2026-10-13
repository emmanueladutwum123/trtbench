// Synthetic layer of the trap suite. Each wrong benchmark is represented by
// the samples it would produce on an L4-class GPU; nothing here is presented
// as a measurement. The point is to pin down the auditor's rules with exact,
// deterministic inputs on any machine.

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "trtbench/compare.hpp"
#include "trtbench/traps.hpp"

namespace trtbench {

namespace {

const char* kLayer = "synthetic";

DevicePeaks l4() {
  DevicePeaks p = theoretical_peaks(8, 9, 58, 2040, 6251000.0, 192);
  p.dram_gbps_measured = 280.0;
  return p;
}

ClockReport clocks(const std::vector<unsigned>& sm, std::uint64_t reasons_from = 0,
                   std::size_t reasons_at = SIZE_MAX) {
  std::vector<ClockSnapshot> s;
  for (std::size_t i = 0; i < sm.size(); ++i) {
    ClockSnapshot c;
    c.t_ms = 100.0 * static_cast<double>(i);
    c.sm_mhz = sm[i];
    c.mem_mhz = 6251;
    c.throttle_reasons = i >= reasons_at ? reasons_from : 0;
    s.push_back(c);
  }
  return summarize_clocks(true, "synthetic", {}, s);
}

std::vector<double> around(double center, std::size_t n) {
  std::vector<double> v;
  for (std::size_t i = 0; i < n; ++i)
    v.push_back(center * (1.0 + (static_cast<double>(i % 21) - 10.0) / 1000.0));
  return v;
}

FinalizeInput base(const std::string& name, std::uint64_t bytes, std::uint64_t flops, double ms) {
  FinalizeInput in;
  in.name = name;
  in.bytes = bytes;
  in.flops = flops;
  in.raw.device_ms = around(ms, 300);
  in.raw.host_enqueue_ms = around(0.005, 300);
  for (double d : in.raw.device_ms) in.raw.wall_ms.push_back(d + 0.01);
  in.raw.first_device_ms = ms * 1.05;
  in.raw.first_host_ms = 0.02;
  in.clocks = clocks(std::vector<unsigned>(10, 2040));
  in.peaks = l4();
  in.launch_floor_ms = 0.004;
  in.checksum = {true, true, 0, "synthetic"};
  return in;
}

const TrapSpec& spec(const char* id) {
  for (const auto& s : synthetic_trap_specs())
    if (s.id == id) return s;
  throw std::logic_error(std::string("no trap spec ") + id);
}

// T1: an FMA kernel whose result is never stored. The compiler deletes the
// body, so it runs as fast as an empty kernel while still "declaring" the
// work it was supposed to do. L4: 58 SMs x 1536 threads x 65536 iters x 16
// FLOP = 9.3e10 FLOP; the live kernel takes ~3.4 ms (27 TFLOPS), the dead one
// takes the 4 us launch floor (23 PFLOPS).
TrapResult t1() {
  const std::uint64_t threads = 58ull * 1536;
  const std::uint64_t flops = threads * 65536 * 16;
  FinalizeInput wrong = base("dead_fma", 0, flops, 0.004);
  wrong.checksum = {false, false, 0, "result never stored"};
  FinalizeInput control = base("live_fma", threads * 4, flops, 3.4);
  const Result w = finalize(wrong), c = finalize(control);
  return judge_flag(spec("T1"), kLayer, Flag::ImpossibleFlops, w, &c);
}

// T2: t0 = now; launch; t1 = now, no synchronize. The host timer measures
// the ~6 us launch of a 2.1 ms copy.
TrapResult t2() {
  const std::uint64_t bytes = 2ull * 256 * 1024 * 1024;
  FinalizeInput wrong = base("copy_timed_without_sync", bytes, 0, 2.15);
  wrong.raw.wall_ms.assign(wrong.raw.device_ms.size(), 0.006);
  FinalizeInput control = base("copy_timed_with_sync", bytes, 0, 2.15);
  const Result w = finalize(wrong), c = finalize(control);
  return judge_flag(spec("T2"), kLayer, Flag::WallclockBelowDeviceTime, w, &c);
}

// T3: no warmup; the first call carries 40 ms of module loading / tuning on
// top of a 0.15 ms op. A mean over all 301 calls would report 0.28 ms.
TrapResult t3() {
  FinalizeInput in = base("cold_start", 2ull * 16 * 1024 * 1024, 0, 0.15);
  in.config.warmup_iters = 0;
  in.config.warmup_min_ms = 0;
  in.raw.first_device_ms = 40.0;
  const Result r = finalize(in);

  TrapResult t;
  t.spec = spec("T3");
  t.layer = kLayer;
  t.wrong = verdict(r);
  t.control = "none";
  const bool surfaced = r.has(Flag::FirstCallOutlier);
  const bool excluded = r.device_ms.max < r.first_call_device_ms && r.device_ms.n == 300;
  const bool warned = r.has(Flag::NoWarmup);
  double sum = r.first_call_device_ms;
  for (double d : r.device_samples_ms) sum += d;
  const double naive_mean = sum / static_cast<double>(r.device_samples_ms.size() + 1);
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "first call %.2f ms reported separately; p50 %.3f ms; a mean over every call "
                "would have said %.3f ms",
                r.first_call_device_ms, r.device_ms.p50, naive_mean);
  t.detail = buf;
  t.outcome = surfaced && excluded && warned ? TrapOutcome::Pass : TrapOutcome::Fail;
  if (t.outcome == TrapOutcome::Fail)
    t.detail = std::string("surfaced=") + (surfaced ? "yes" : "no") +
               " excluded=" + (excluded ? "yes" : "no") + " no_warmup=" + (warned ? "yes" : "no");
  return t;
}

// T4: SM clock falls from 2040 to 1500 MHz partway through steady state.
TrapResult t4() {
  FinalizeInput wrong = base("copy_while_throttling", 1 << 28, 0, 1.1);
  wrong.clocks = clocks({2040, 2040, 2040, 1905, 1785, 1650, 1500, 1500, 1500, 1500});
  FinalizeInput control = base("copy_steady_clocks", 1 << 28, 0, 1.1);
  const Result w = finalize(wrong), c = finalize(control);
  return judge_flag(spec("T4"), kLayer, Flag::ClockDrift, w, &c);
}

// T4t: clocks look steady but NVML reports a thermal slowdown midway.
TrapResult t4t() {
  FinalizeInput wrong = base("copy_thermal", 1 << 28, 0, 1.1);
  wrong.clocks = clocks(std::vector<unsigned>(10, 2040), throttle::kSwThermal, 6);
  FinalizeInput control = base("copy_no_throttle_reason", 1 << 28, 0, 1.1);
  const Result w = finalize(wrong), c = finalize(control);
  return judge_flag(spec("T4t"), kLayer, Flag::ThermalOrHwThrottle, w, &c);
}

// T5: a memory-bound elementwise op in FP32 and FP16. FP16 moves half the
// bytes and is ~2x faster; that is only a result next to its accuracy cost.
TrapResult t5() {
  const std::uint64_t n = 64ull << 20;
  FinalizeInput f32 = base("scale_fp32", 2 * n * 4, n, 2.0);
  FinalizeInput f16 = base("scale_fp16", 2 * n * 2, n, 1.02);
  f16.precision = Precision::FP16;
  const Result a = finalize(f32), b = finalize(f16);
  const Comparison without = compare(a, b);
  const Comparison with = compare(a, b, AccuracyDelta{"max_rel_err_vs_fp64", 6e-8, 4.9e-4});

  TrapResult t;
  t.spec = spec("T5");
  t.layer = kLayer;
  t.wrong = std::string("compare(fp32, fp16): ") + to_string(without.status);
  char buf[160];
  std::snprintf(buf, sizeof buf, "compare(fp32, fp16, accuracy): %s, %.2fx with max rel err %.1e -> %.1e",
                to_string(with.status), with.speedup_p50, 6e-8, 4.9e-4);
  t.control = buf;
  const bool refused = without.status == ComparisonStatus::Refused && without.speedup_p50 == 0;
  const bool reported = with.status == ComparisonStatus::Ok;
  t.outcome = refused && reported ? TrapOutcome::Pass : TrapOutcome::Fail;
  t.detail = refused ? without.reason : "a speedup was reported without an accuracy delta";
  return t;
}

}  // namespace

std::vector<TrapResult> run_synthetic_traps() {
  return {t1(), t2(), t3(), t4(), t4t(), t5()};
}

}  // namespace trtbench
