// trtbench: run the reference workloads through the measurement harness and
// emit a JSON report, or run the trap suite with --self-test.
//
// Exit status: 0 success, 1 runtime error, 2 usage error, 3 at least one
// result failed its audit, 4 at least one self-test trap failed. A run that
// produced an impossible number fails; it does not just print.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "trtbench/context_options.hpp"
#include "trtbench/json.hpp"
#include "trtbench/traps.hpp"

#ifdef TRTBENCH_HAVE_CUDA
#include "trtbench/cuda/builtin_workloads.hpp"
#include "trtbench/cuda/device_context.hpp"
#include "trtbench/cuda/runner.hpp"
#include "trtbench/cuda/traps_gpu.hpp"
#endif
#ifdef TRTBENCH_HAVE_TENSORRT
#include "trtbench/trt/engine_workload.hpp"
#endif

using namespace trtbench;

namespace {

constexpr const char* kUsage = R"(usage: trtbench [options]

Self-test:
  --self-test         run the trap suite: deliberately wrong benchmarks the
                      harness must reject. GPU traps skip without a device.

Workloads:
  --op NAME           empty | copy | fma | all | none
                      (default: all, or none when --engine/--ingest is given)
  --copy-mib N        copy size in MiB            (default: max(256, 16 x L2))
  --fma-iters N       FMA loop iterations         (default: 16384)

TensorRT (builds with -DTB_ENABLE_TENSORRT=ON):
  --engine META       time the engine described by META (engines.py writes
                      it); repeatable
  --cuda-graph        also time each engine replayed from a captured CUDA graph
  --engine-eval META  run the engine over its evaluation set and write the
                      outputs to --eval-out (accuracy support; untimed)
  --eval-out PATH

Baselines timed outside this binary:
  --ingest SAMPLES    audit samples written by python/trtbench/pyrunner.py
                      against this process's calibration; repeatable

Measurement:
  --mode M            latency | throughput        (default: latency)
  --launches N        launches per sample, throughput mode (default: 100)
  --warmup N          warmup iterations           (default: 50)
  --warmup-ms X       warmup time floor           (default: 250)
  --min-iters N       steady-state samples        (default: 200)
  --min-time-ms X     steady-state time floor     (default: 1000)
  --max-iters N                                   (default: 100000)
  --flush-l2          overwrite 2 x L2 before every sample (cold cache)
  --drift-tol X       clock drift tolerance, fraction (default: 0.05)

Device:
  --device N          CUDA device ordinal         (default: 0)
  --lock-clocks=MHZ   lock the SM clock via nvidia-smi -lgc. Needs root;
                      degrades to drift detection when not permitted. Choose
                      a clock the GPU sustains under load: at or below
                      clocks.sm_mhz.min from an unlocked run's JSON.
  --clocks-locked-externally=MHZ
                      the caller already locked the SM clock (run_phase2.sh
                      locks it once for every process); recorded, and still
                      checked by the drift audit
  --no-calibrate      skip launch-floor and copy-peak calibration

Output:
  --json PATH         write the full report (with raw samples) to PATH
  --no-samples        omit raw per-sample arrays from the JSON
  -h, --help
)";

struct Args {
  std::string op = "all";
  std::size_t copy_mib = 0;
  int fma_iters = 16384;
  BenchConfig cfg;
  cuda::ContextOptions ctx;
  std::string json_path;
  bool samples = true;
  bool self_test = false;
  bool op_given = false;
  std::vector<std::string> engines;
  bool cuda_graph = false;
  std::vector<std::string> ingest;
  std::string engine_eval;
  std::string eval_out;
};

[[noreturn]] void usage_error(const std::string& msg) {
  std::fprintf(stderr, "trtbench: %s\n\n%s", msg.c_str(), kUsage);
  std::exit(2);
}

long to_long(const std::string& flag, const char* v) {
  char* end = nullptr;
  const long x = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') usage_error(flag + " expects an integer, got '" + v + "'");
  return x;
}

double to_double(const std::string& flag, const char* v) {
  char* end = nullptr;
  const double x = std::strtod(v, &end);
  if (end == v || *end != '\0') usage_error(flag + " expects a number, got '" + v + "'");
  return x;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage_error(f + " needs a value");
      return argv[++i];
    };
    if (f == "-h" || f == "--help") {
      std::fputs(kUsage, stdout);
      std::exit(0);
    } else if (f == "--self-test") {
      a.self_test = true;
    } else if (f == "--op") {
      a.op = next();
      a.op_given = true;
    } else if (f == "--engine") {
      a.engines.push_back(next());
    } else if (f == "--cuda-graph") {
      a.cuda_graph = true;
    } else if (f == "--engine-eval") {
      a.engine_eval = next();
    } else if (f == "--eval-out") {
      a.eval_out = next();
    } else if (f == "--ingest") {
      a.ingest.push_back(next());
    } else if (f.rfind("--clocks-locked-externally=", 0) == 0) {
      a.ctx.external_lock_mhz = static_cast<unsigned>(
          to_long(f, f.c_str() + std::strlen("--clocks-locked-externally=")));
    } else if (f == "--copy-mib") {
      a.copy_mib = static_cast<std::size_t>(to_long(f, next()));
    } else if (f == "--fma-iters") {
      a.fma_iters = static_cast<int>(to_long(f, next()));
    } else if (f == "--mode") {
      const std::string m = next();
      if (m == "latency") a.cfg.mode = Mode::Latency;
      else if (m == "throughput") a.cfg.mode = Mode::Throughput;
      else usage_error("unknown mode '" + m + "'");
    } else if (f == "--launches") {
      a.cfg.launches_per_sample = static_cast<int>(to_long(f, next()));
    } else if (f == "--warmup") {
      a.cfg.warmup_iters = static_cast<int>(to_long(f, next()));
    } else if (f == "--warmup-ms") {
      a.cfg.warmup_min_ms = to_double(f, next());
    } else if (f == "--min-iters") {
      a.cfg.min_iters = static_cast<int>(to_long(f, next()));
    } else if (f == "--min-time-ms") {
      a.cfg.min_time_ms = to_double(f, next());
    } else if (f == "--max-iters") {
      a.cfg.max_iters = static_cast<int>(to_long(f, next()));
    } else if (f == "--flush-l2") {
      a.cfg.flush_l2 = true;
    } else if (f == "--drift-tol") {
      a.cfg.clock_drift_tol = to_double(f, next());
    } else if (f == "--device") {
      a.ctx.device = static_cast<int>(to_long(f, next()));
    } else if (f == "--lock-clocks") {
      usage_error("--lock-clocks needs a target, e.g. --lock-clocks=1100 (see --help)");
    } else if (f.rfind("--lock-clocks=", 0) == 0) {
      a.ctx.lock_clocks = true;
      a.ctx.lock_mhz = static_cast<unsigned>(to_long(f, f.c_str() + std::strlen("--lock-clocks=")));
    } else if (f == "--no-calibrate") {
      a.ctx.calibrate = false;
    } else if (f == "--json") {
      a.json_path = next();
    } else if (f == "--no-samples") {
      a.samples = false;
    } else {
      usage_error("unknown option '" + f + "'");
    }
  }
  if (const std::string e = validate(a.cfg); !e.empty()) usage_error(e);
  if (!a.op_given && (!a.engines.empty() || !a.ingest.empty())) a.op = "none";
  if (a.op != "all" && a.op != "empty" && a.op != "copy" && a.op != "fma" && a.op != "none")
    usage_error("unknown op '" + a.op + "'");
  if (!a.engine_eval.empty() && a.eval_out.empty()) usage_error("--engine-eval needs --eval-out");
#ifndef TRTBENCH_HAVE_TENSORRT
  if (!a.engines.empty() || !a.engine_eval.empty())
    usage_error("this build has no TensorRT support (configure with -DTB_ENABLE_TENSORRT=ON)");
#endif
  return a;
}

#ifdef TRTBENCH_HAVE_CUDA
std::string us(double ms) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f", ms * 1e3);
  return b;
}

void print_env(const RunReport& r) {
  const Environment& e = r.env;
  std::printf("GPU        %s (sm_%d%d, %d SMs, PCI %s)\n", e.gpu_name.c_str(), e.cc_major,
              e.cc_minor, e.sm_count, e.pci_bus_id.c_str());
  std::printf("Driver     %s   CUDA driver API %d   runtime %d\n",
              e.driver_version.empty() ? "?" : e.driver_version.c_str(), e.cuda_driver_api_version,
              e.cuda_runtime_version);
  std::printf("NVML       %s\n", e.nvml_status.c_str());
  std::printf("Peaks      DRAM %.1f GB/s theoretical, %.1f GB/s measured copy; FP32 %.1f TFLOPS; "
              "FP16 TC %.1f TFLOPS @ %u MHz\n",
              r.peaks.dram_gbps_theoretical, r.peaks.dram_gbps_measured,
              r.peaks.fp32_gflops / 1e3, r.peaks.fp16_tc_gflops / 1e3, r.peaks.peak_sm_clock_mhz);
  std::printf("Floors     empty kernel %s us (latency), %s us/launch (throughput)\n",
              us(r.launch_floor_latency_ms).c_str(), us(r.launch_floor_throughput_ms).c_str());
  std::printf("git        %s\n\n", e.git_sha.c_str());
}

void print_result(const Result& r) {
  std::printf("== %s  [%s, %s, %d launch/sample, n=%zu]  %s\n", r.name.c_str(),
              to_string(r.config.mode), to_string(r.precision), r.launches_per_sample,
              r.device_ms.n, r.trustworthy() ? "TRUSTWORTHY" : "REJECTED");
  std::printf("   device us   p50 %s  p90 %s  p99 %s  (first call %s, excluded)\n",
              us(r.device_ms.p50).c_str(), us(r.device_ms.p90).c_str(), us(r.device_ms.p99).c_str(),
              us(r.first_call_device_ms).c_str());
  std::printf("   host   us   enqueue p50 %s  p99 %s\n", us(r.host_enqueue_ms.p50).c_str(),
              us(r.host_enqueue_ms.p99).c_str());
  const RooflinePoint& f = r.roofline;
  std::printf("   roofline    %.1f GB/s (%.1f%% of measured, %.1f%% of theoretical)  "
              "%.1f GFLOP/s (%.1f%% of peak @ run clock)  AI %.3f  ridge %.1f  -> %s-bound\n",
              f.achieved_gbps, f.pct_bw_peak, f.pct_bw_theoretical, f.achieved_gflops,
              f.pct_compute_peak, f.intensity, f.ridge, to_string(f.bound));
  if (r.clocks.available) {
    std::printf("   clocks      SM %u-%u MHz (drift %.1f%%)  mem %u-%u MHz  reasons: %s  %s\n",
                r.clocks.sm_min, r.clocks.sm_max, 100 * r.clocks.sm_drift, r.clocks.mem_min,
                r.clocks.mem_max, throttle::describe(r.clocks.throttle_union).c_str(),
                r.clocks.locked ? "[locked]" : "[unlocked]");
  }
  std::printf("   output      %s\n", r.checksum.note.c_str());
  for (const auto& x : r.findings)
    std::printf("   [%s] %s: %s\n", to_string(x.severity), to_string(x.flag), x.detail.c_str());
  std::printf("\n");
}
#endif

int self_test(const Args& a) {
  std::vector<TrapResult> rows = run_synthetic_traps();
  nlohmann::json env = nullptr;
  std::string device = "none";
#ifdef TRTBENCH_HAVE_CUDA
  std::string why;
  if (cuda::cuda_device_count(&why) > 0) {
    auto ctx = cuda::DeviceContext::create(a.ctx);
    device = ctx->env().gpu_name;
    env = to_json(ctx->env());
    for (auto& t : cuda::run_gpu_traps(*ctx)) rows.push_back(std::move(t));
  } else {
    for (auto& t : skipped_traps(gpu_trap_specs(), "gpu", "no CUDA device: " + why))
      rows.push_back(std::move(t));
  }
#else
  for (auto& t : skipped_traps(gpu_trap_specs(), "gpu",
                               "built without CUDA (configure with -DTB_ENABLE_CUDA=ON)"))
    rows.push_back(std::move(t));
#endif
  std::printf("trtbench self-test: deliberately wrong benchmarks the harness must reject\n");
  std::printf("device: %s   git: %s\n\n", device.c_str(), build_git_sha().c_str());
  std::fputs(render_trap_table(rows).c_str(), stdout);

  if (!a.json_path.empty()) {
    nlohmann::json j = {{"git_sha", build_git_sha()}, {"environment", env}};
    j["traps"] = nlohmann::json::array();
    for (const auto& t : rows) j["traps"].push_back(to_json(t));
    j["all_passed"] = no_failures(rows);
    std::ofstream(a.json_path) << j.dump(2) << "\n";
    std::printf("wrote %s\n", a.json_path.c_str());
  }
  return no_failures(rows) ? 0 : 4;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (a.self_test) {
    try {
      return self_test(a);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "trtbench: self-test aborted: %s\n", e.what());
      return 1;
    }
  }
#ifndef TRTBENCH_HAVE_CUDA
  std::fprintf(stderr,
               "trtbench: this build has no CUDA support (configure with -DTB_ENABLE_CUDA=ON); "
               "only --self-test is available\n");
  return 2;
#else
  try {
#ifdef TRTBENCH_HAVE_TENSORRT
    if (!a.engine_eval.empty()) {
      const std::size_t n = trt::evaluate_engine(a.engine_eval, a.eval_out);
      std::printf("wrote %zu evaluation outputs to %s\n", n, a.eval_out.c_str());
      return 0;
    }
#endif
    auto ctx = cuda::DeviceContext::create(a.ctx);
    if (a.ctx.lock_clocks) std::printf("Clock lock %s\n", ctx->lock_status().c_str());

    std::vector<std::unique_ptr<cuda::Workload>> ws;
    if (a.op == "empty" || a.op == "all") ws.push_back(cuda::make_empty_kernel());
    if (a.op == "copy" || a.op == "all") {
      std::size_t bytes = a.copy_mib << 20;
      if (bytes == 0)
        bytes = std::max<std::size_t>(std::size_t{256} << 20,
                                      static_cast<std::size_t>(std::max(0LL, ctx->env().l2_bytes)) * 16);
      ws.push_back(cuda::make_copy(bytes));
    }
    if (a.op == "fma" || a.op == "all") ws.push_back(cuda::make_fma(a.fma_iters));
#ifdef TRTBENCH_HAVE_TENSORRT
    for (const auto& m : a.engines) {
      ws.push_back(trt::make_engine_workload(m, false));
      if (a.cuda_graph) ws.push_back(trt::make_engine_workload(m, true));
    }
#endif

    std::vector<Result> results;
    for (auto& w : ws) {
      std::fprintf(stderr, "trtbench: timing %s\n", w->name().c_str());
      results.push_back(cuda::run(*w, a.cfg, *ctx));
      w.reset();  // free device memory before the next engine
    }
    for (const auto& path : a.ingest) {
      std::ifstream f(path);
      if (!f) throw std::runtime_error("cannot open " + path);
      FinalizeInput in = samples_from_json(nlohmann::json::parse(f));
      // Every backend is judged against this process's calibration.
      in.peaks = ctx->peaks();
      in.launch_floor_ms = ctx->launch_floor_ms(in.config.mode);
      results.push_back(finalize(std::move(in)));
    }

    const RunReport rep = ctx->make_report(std::move(results));
    print_env(rep);
    for (const auto& r : rep.calibration) print_result(r);
    for (const auto& r : rep.results) print_result(r);

    if (!a.json_path.empty()) {
      std::ofstream out(a.json_path);
      out << to_json(rep, a.samples).dump(2) << "\n";
      if (!out) throw std::runtime_error("could not write " + a.json_path);
      std::printf("wrote %s\n", a.json_path.c_str());
    }
    if (!rep.all_trustworthy()) {
      std::fprintf(stderr, "trtbench: at least one result failed its audit (see REJECTED above)\n");
      return 3;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "trtbench: %s\n", e.what());
    return 1;
  }
#endif
}
