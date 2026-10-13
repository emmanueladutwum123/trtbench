// GPU layer of the trap suite. Every wrong benchmark here really runs; the
// harness has to catch it from what it observes, exactly as it would in a
// real measurement.

#include "trtbench/cuda/traps_gpu.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "trtbench/compare.hpp"
#include "trtbench/cuda/builtin_workloads.hpp"
#include "trtbench/cuda/check.hpp"
#include "trtbench/cuda/clock_lock.hpp"
#include "trtbench/cuda/runner.hpp"

namespace trtbench::cuda {

namespace {

constexpr unsigned kThreads = 256;
const char* kLayer = "gpu";

const TrapSpec& spec(const char* id) {
  for (const auto& s : gpu_trap_specs())
    if (s.id == id) return s;
  throw std::logic_error(std::string("no trap spec ") + id);
}

int attr(cudaDeviceAttr a) {
  int dev = 0, v = 0;
  TB_CUDA_CHECK(cudaGetDevice(&dev));
  TB_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

unsigned one_wave_blocks() {
  return static_cast<unsigned>(attr(cudaDevAttrMultiProcessorCount) *
                               (attr(cudaDevAttrMaxThreadsPerMultiProcessor) / kThreads));
}

unsigned grid_for(std::size_t n) {
  const std::size_t want = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(std::min<std::size_t>(want, std::size_t(one_wave_blocks()) * 4));
}

BenchConfig quick(Mode m) {
  BenchConfig c;
  c.mode = m;
  c.min_iters = 200;
  c.min_time_ms = 300;
  c.warmup_min_ms = 100;
  c.launches_per_sample = 10;
  return c;
}

std::string fmt2(const char* f, double a, double b) {
  char buf[256];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}

// ===========================================================================
// T1: result never used
// ===========================================================================

// The same FMA chains as the calibration fma_kernel, but the result is never
// stored. Nothing observable depends on the loop, so the compiler deletes it
// and the kernel is empty. scripts/check_dce_sass.py verifies that in CI by
// reading the SASS: no FFMA here, FFMA present in the live kernel.
#pragma nv_diagnostic push
#pragma nv_diag_suppress 550  // "set but never used": that is the point
__global__ void trap_dead_fma_kernel(int iters, float a, float b) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  float x[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) x[k] = static_cast<float>(t & 1023u) + static_cast<float>(k);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = fmaf(x[k], a, b);
  }
  // x is discarded here. This is the bug.
}
#pragma nv_diagnostic pop

class DeadFma final : public Workload {
 public:
  explicit DeadFma(int iters) : iters_(iters) {}
  std::string name() const override { return "trap/fma_result_unused_x" + std::to_string(iters_); }
  void setup(cudaStream_t) override { blocks_ = one_wave_blocks(); }
  void enqueue(cudaStream_t s) override {
    trap_dead_fma_kernel<<<blocks_, kThreads, 0, s>>>(iters_, 0.9999f, 0.0001f);
  }
  std::uint64_t bytes_moved() const override { return 0; }
  // The work the benchmark author believes is being timed.
  std::uint64_t flops() const override {
    return std::uint64_t(blocks_) * kThreads * std::uint64_t(iters_) * 8 * 2;
  }
  Checksum consume_output(cudaStream_t) override {
    return {false, false, 0, "result is never stored, so there is nothing to check"};
  }

 private:
  int iters_;
  unsigned blocks_ = 0;
};

TrapResult t1(DeviceContext& ctx) {
  constexpr int kIters = 32768;
  DeadFma dead(kIters);
  auto live = make_fma(kIters);
  const Result w = run(dead, quick(Mode::Latency), ctx);
  const Result c = run(*live, quick(Mode::Latency), ctx);
  TrapResult t = judge_flag(spec("T1"), kLayer, Flag::ImpossibleFlops, w, &c);
  t.detail += fmt2(" [dead p50 %.4f ms vs live p50 %.4f ms]", w.device_ms.p50, c.device_ms.p50);
  return t;
}

// ===========================================================================
// T2: host timer stopped without synchronize
// ===========================================================================

// The naive benchmark: t0 = now; launch; t1 = now; report t1 - t0. CUDA
// launches are asynchronous, so this is the launch cost. The harness records
// events around the same launch and reads them only after the naive timer has
// already stopped.
Result measure_without_sync(Workload& w, DeviceContext& ctx, int samples) {
  using Clock = std::chrono::steady_clock;
  cudaStream_t s = ctx.stream();
  w.setup(s);
  TB_CUDA_CHECK(cudaStreamSynchronize(s));
  for (int i = 0; i < 5; ++i) w.enqueue(s);  // keep first-call cost out of it
  TB_CUDA_CHECK(cudaStreamSynchronize(s));

  ClockSource& clocks = ctx.clocks();
  std::vector<ClockSnapshot> snaps;
  if (clocks.available()) snaps.push_back(clocks.sample());

  RawSamples raw;
  for (int i = 0; i < samples; ++i) {
    const auto t0 = Clock::now();
    TB_CUDA_CHECK(cudaEventRecord(ctx.start_event(), s));
    w.enqueue(s);
    TB_CUDA_CHECK(cudaEventRecord(ctx.stop_event(), s));
    const auto t1 = Clock::now();  // BUG under test: no synchronize before stopping the timer
    TB_CUDA_CHECK(cudaEventSynchronize(ctx.stop_event()));  // harness: read events afterwards
    TB_CUDA_CHECK(cudaGetLastError());
    float ev = 0;
    TB_CUDA_CHECK(cudaEventElapsedTime(&ev, ctx.start_event(), ctx.stop_event()));
    const double naive = std::chrono::duration<double, std::milli>(t1 - t0).count();
    raw.device_ms.push_back(ev);
    raw.host_enqueue_ms.push_back(naive);
    raw.wall_ms.push_back(naive);  // what the naive benchmark claims encloses the work
  }
  if (clocks.available()) snaps.push_back(clocks.sample());

  FinalizeInput in;
  in.name = "trap/" + w.name() + "_timed_without_sync";
  in.precision = w.precision();
  in.config.warmup_iters = 5;
  in.bytes = w.bytes_moved();
  in.flops = w.flops();
  in.checksum = w.consume_output(s);
  TB_CUDA_CHECK(cudaStreamSynchronize(s));
  in.raw = std::move(raw);
  in.clocks = summarize_clocks(clocks.available(), clocks.name(), {}, std::move(snaps));
  in.peaks = ctx.peaks();
  in.launch_floor_ms = ctx.launch_floor_ms(Mode::Latency);
  return finalize(std::move(in));
}

TrapResult t2(DeviceContext& ctx) {
  auto copy = make_copy(std::size_t{256} << 20);
  const Result w = measure_without_sync(*copy, ctx, 200);
  const Result c = run(*copy, quick(Mode::Latency), ctx);
  TrapResult t = judge_flag(spec("T2"), kLayer, Flag::WallclockBelowDeviceTime, w, &c);
  t.detail += fmt2(" [naive timer p50 %.4f ms; events p50 %.4f ms]", w.wall_ms.p50, w.device_ms.p50);
  return t;
}

// ===========================================================================
// T3: no warmup, expensive first call
// ===========================================================================

__device__ __host__ inline float trap_input(std::size_t i) {
  return 1.0f + static_cast<float>(i & 1023u) * 0.0009765625f;  // exact in fp32 and fp16
}

__global__ void trap_fill_f32(float* p, std::size_t n) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = trap_input(i);
}

__global__ void trap_cold_scale_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       std::size_t n, float a) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    out[i] = in[i] * a;
}

// Models a framework that tunes on its first call (cuDNN/TensorRT tactic
// selection): the first enqueue runs the kernel 50 times to "pick" a config.
// The kernel is launched nowhere else, so its lazy module load also lands on
// the first call.
class ColdStart final : public Workload {
 public:
  ~ColdStart() override {
    cudaFree(in_);
    cudaFree(out_);
  }
  std::string name() const override { return "trap/cold_start_autotune"; }
  void setup(cudaStream_t s) override {
    if (in_ == nullptr) {
      TB_CUDA_CHECK(cudaMalloc(&in_, kN * sizeof(float)));
      TB_CUDA_CHECK(cudaMalloc(&out_, kN * sizeof(float)));
    }
    grid_ = grid_for(kN);
    trap_fill_f32<<<grid_, kThreads, 0, s>>>(in_, kN);
    TB_CUDA_CHECK(cudaGetLastError());
    first_ = true;
  }
  void enqueue(cudaStream_t s) override {
    if (first_) {
      for (int i = 0; i < 50; ++i) trap_cold_scale_kernel<<<grid_, kThreads, 0, s>>>(in_, out_, kN, kA);
      first_ = false;
    }
    trap_cold_scale_kernel<<<grid_, kThreads, 0, s>>>(in_, out_, kN, kA);
  }
  std::uint64_t bytes_moved() const override { return 2ull * kN * sizeof(float); }
  std::uint64_t flops() const override { return kN; }
  Checksum consume_output(cudaStream_t s) override {
    std::vector<float> got(kN);
    TB_CUDA_CHECK(cudaMemcpyAsync(got.data(), out_, kN * sizeof(float), cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    for (std::size_t i = 0; i < kN; i += 997)
      if (got[i] != trap_input(i) * kA) return {true, false, 0, "out[" + std::to_string(i) + "] wrong"};
    return {true, true, 0, "exact at every 997th element"};
  }

 private:
  static constexpr std::size_t kN = std::size_t{4} << 20;  // 16 MiB per buffer
  static constexpr float kA = 1.1f;
  float* in_ = nullptr;
  float* out_ = nullptr;
  unsigned grid_ = 0;
  bool first_ = true;
};

TrapResult t3(DeviceContext& ctx) {
  ColdStart w;
  BenchConfig cfg = quick(Mode::Latency);
  cfg.warmup_iters = 0;
  cfg.warmup_min_ms = 0;
  const Result r = run(w, cfg, ctx);

  TrapResult t;
  t.spec = spec("T3");
  t.layer = kLayer;
  t.wrong = verdict(r);
  t.control = "none (first-call cost of a plain kernel depends on driver lazy loading)";
  const bool surfaced = r.has(Flag::FirstCallOutlier);
  const bool excluded = r.device_ms.max < r.first_call_device_ms &&
                        r.device_ms.n == r.device_samples_ms.size();
  const bool warned = r.has(Flag::NoWarmup);
  double sum = r.first_call_device_ms;
  for (double d : r.device_samples_ms) sum += d;
  const double mean_all = sum / static_cast<double>(r.device_samples_ms.size() + 1);
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "first call %.3f ms reported separately; p50 %.4f ms; a mean over every call "
                "would have said %.4f ms",
                r.first_call_device_ms, r.device_ms.p50, mean_all);
  t.detail = buf;
  t.outcome = surfaced && excluded && warned ? TrapOutcome::Pass : TrapOutcome::Fail;
  if (t.outcome == TrapOutcome::Fail)
    t.detail += std::string(" | surfaced=") + (surfaced ? "yes" : "no") +
                " excluded=" + (excluded ? "yes" : "no") + " no_warmup=" + (warned ? "yes" : "no");
  return t;
}

// ===========================================================================
// T4: clocks drop during the run
// ===========================================================================

// Real timing, scripted clock readings: proves the runner consults the clock
// source during steady state and that a drop reaches the verdict.
class ScriptedClocks final : public ClockSource {
 public:
  ScriptedClocks(unsigned sm, int drop_from_call) : sm_(sm), drop_from_(drop_from_call) {}
  bool available() const override { return true; }
  std::string name() const override { return "scripted (real GPU timing, scripted clock readings)"; }
  ClockSnapshot sample() override {
    ++calls_;
    ClockSnapshot s;
    s.sm_mhz = (drop_from_ > 0 && calls_ >= drop_from_) ? sm_ * 7 / 10 : sm_;
    s.mem_mhz = 6251;
    return s;
  }

 private:
  unsigned sm_;
  int drop_from_;
  int calls_ = 0;
};

TrapResult t4(DeviceContext& ctx) {
  const unsigned sm = ctx.env().max_sm_clock_mhz > 0 ? unsigned(ctx.env().max_sm_clock_mhz) : 1500;
  auto copy = make_copy(std::size_t{64} << 20);
  BenchConfig cfg = quick(Mode::Latency);
  cfg.clock_sample_every = 10;
  // Calls: 1 = before warmup, 2 = after warmup, 3+ = steady state.
  ScriptedClocks dropping(sm, 4), steady(sm, 0);
  const Result w = run(*copy, cfg, ctx, dropping);
  const Result c = run(*copy, cfg, ctx, steady);
  return judge_flag(spec("T4"), kLayer, Flag::ClockDrift, w, &c);
}

// Real clocks: halve the SM clock with nvidia-smi -lgc partway through a run.
class InducedThrottle final : public ClockSource {
 public:
  InducedThrottle(ClockSource& inner, int index, unsigned mhz)
      : inner_(inner), index_(index), mhz_(mhz) {}
  bool available() const override { return true; }
  std::string name() const override { return inner_.name() + " (+ induced -lgc drop)"; }
  ClockSnapshot sample() override {
    if (++calls_ == 3) {
      const std::string m = std::to_string(mhz_);
      auto [code, out] =
          run_command("nvidia-smi -i " + std::to_string(index_) + " -lgc " + m + "," + m);
      triggered_ = code == 0;
      message_ = "exit " + std::to_string(code) + ": " + out;
    }
    return inner_.sample();
  }
  bool triggered() const { return triggered_; }
  const std::string& message() const { return message_; }

 private:
  ClockSource& inner_;
  int index_;
  unsigned mhz_;
  int calls_ = 0;
  bool triggered_ = false;
  std::string message_ = "trigger never reached";
};

TrapResult t4r(DeviceContext& ctx) {
  const TrapSpec& sp = spec("T4r");
  if (!ctx.clocks().available() || ctx.nvml_index() < 0)
    return skipped_traps({sp}, kLayer, "NVML unavailable: cannot observe a real clock change")[0];
  if (ctx.env().max_sm_clock_mhz <= 0)
    return skipped_traps({sp}, kLayer, "max SM clock unknown: no target for the induced drop")[0];

  const unsigned target = std::max(300u, unsigned(ctx.env().max_sm_clock_mhz) / 2);
  auto copy = make_copy(std::size_t{64} << 20);
  BenchConfig cfg = quick(Mode::Latency);
  cfg.clock_sample_every = 10;
  InducedThrottle src(ctx.clocks(), ctx.nvml_index(), target);
  const Result w = run(*copy, cfg, ctx, src);

  // Restore whatever clock policy the context started with.
  const std::string idx = std::to_string(ctx.nvml_index());
  if (ctx.locked_mhz() > 0) {
    const std::string m = std::to_string(ctx.locked_mhz());
    run_command("nvidia-smi -i " + idx + " -lgc " + m + "," + m);
  } else if (src.triggered()) {
    run_command("nvidia-smi -i " + idx + " -rgc");
  }

  if (!src.triggered())
    return skipped_traps({sp}, kLayer, "nvidia-smi -lgc not permitted (" + src.message() + ")")[0];
  TrapResult t = judge_flag(sp, kLayer, Flag::ClockDrift, w, nullptr);
  t.detail += " [locked to " + std::to_string(target) + " MHz mid-run]";
  return t;
}

// ===========================================================================
// T5: FP16 vs FP32 without an accuracy delta
// ===========================================================================

constexpr float kAlpha = 1.1f;

__global__ void trap_fill_f16(__half* p, std::size_t n) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = __float2half_rn(trap_input(i));
}

__global__ void scale_f32_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                 std::size_t n4, float a) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = in[i];
    out[i] = make_float4(v.x * a, v.y * a, v.z * a, v.w * a);
  }
}

__global__ void scale_f16_kernel(const __half2* __restrict__ in, __half2* __restrict__ out,
                                 std::size_t n2, __half2 a) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n2; i += stride)
    out[i] = __hmul2(in[i], a);
}

// y = 1.1 * x over n elements, FP32 or FP16. Memory-bound: FP16 moves half the
// bytes, so it is ~2x faster for reasons that have nothing to do with compute.
class Scale final : public Workload {
 public:
  Scale(std::size_t n, bool half) : n_(n - n % 8), half_(half) {}
  ~Scale() override {
    cudaFree(in_);
    cudaFree(out_);
  }
  std::string name() const override { return half_ ? "trap/scale_fp16" : "trap/scale_fp32"; }
  Precision precision() const override { return half_ ? Precision::FP16 : Precision::FP32; }
  void setup(cudaStream_t s) override {
    if (in_ == nullptr) {
      TB_CUDA_CHECK(cudaMalloc(&in_, n_ * elem()));
      TB_CUDA_CHECK(cudaMalloc(&out_, n_ * elem()));
    }
    grid_ = grid_for(half_ ? n_ / 2 : n_ / 4);
    if (half_)
      trap_fill_f16<<<grid_, kThreads, 0, s>>>(static_cast<__half*>(in_), n_);
    else
      trap_fill_f32<<<grid_, kThreads, 0, s>>>(static_cast<float*>(in_), n_);
    TB_CUDA_CHECK(cudaGetLastError());
  }
  void enqueue(cudaStream_t s) override {
    if (half_)
      scale_f16_kernel<<<grid_, kThreads, 0, s>>>(static_cast<const __half2*>(in_),
                                                  static_cast<__half2*>(out_), n_ / 2,
                                                  __float2half2_rn(kAlpha));
    else
      scale_f32_kernel<<<grid_, kThreads, 0, s>>>(static_cast<const float4*>(in_),
                                                  static_cast<float4*>(out_), n_ / 4, kAlpha);
  }
  std::uint64_t bytes_moved() const override { return 2ull * n_ * elem(); }
  std::uint64_t flops() const override { return n_; }

  Checksum consume_output(cudaStream_t s) override {
    const std::size_t rows = 4096, stride = n_ / rows;
    std::vector<unsigned char> raw(rows * elem());
    TB_CUDA_CHECK(cudaMemcpy2DAsync(raw.data(), elem(), out_, stride * elem(), elem(), rows,
                                    cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    samples_.clear();
    Checksum c{true, true, 0, ""};
    const __half a_h = __float2half_rn(kAlpha);
    for (std::size_t r = 0; r < rows; ++r) {
      const std::size_t i = r * stride;
      float got, want;
      if (half_) {
        __half h;
        std::memcpy(&h, raw.data() + r * elem(), sizeof h);
        got = __half2float(h);
        // Product of two 11-bit significands is exact in float; rounding it
        // to half reproduces __hmul2's correctly rounded result.
        want = __half2float(__float2half_rn(__half2float(__float2half_rn(trap_input(i))) *
                                            __half2float(a_h)));
      } else {
        std::memcpy(&got, raw.data() + r * elem(), sizeof got);
        want = trap_input(i) * kAlpha;
      }
      samples_.push_back({i, got});
      if (got != want && c.passed) {
        c.passed = false;
        c.note = "out[" + std::to_string(i) + "] = " + std::to_string(got) + ", expected " +
                 std::to_string(want);
      }
    }
    if (c.passed) c.note = "bit-exact at 4096 sampled positions";
    return c;
  }

  // Max relative error of the sampled outputs against y = 1.1 * x in double.
  double max_rel_err_vs_fp64() const {
    double worst = 0;
    for (const auto& [i, got] : samples_) {
      const double exact = static_cast<double>(trap_input(i)) * 1.1;
      worst = std::max(worst, std::abs(got - exact) / exact);
    }
    return worst;
  }

 private:
  std::size_t elem() const { return half_ ? 2 : 4; }
  std::size_t n_;
  bool half_;
  void* in_ = nullptr;
  void* out_ = nullptr;
  unsigned grid_ = 0;
  std::vector<std::pair<std::size_t, double>> samples_;
};

TrapResult t5(DeviceContext& ctx) {
  constexpr std::size_t kN = std::size_t{64} << 20;
  Scale f32(kN, false), f16(kN, true);
  BenchConfig cfg = quick(Mode::Throughput);
  const Result a = run(f32, cfg, ctx);
  const Result b = run(f16, cfg, ctx);
  const AccuracyDelta acc{"max_rel_err_vs_fp64", f32.max_rel_err_vs_fp64(), f16.max_rel_err_vs_fp64()};
  const Comparison without = compare(a, b);
  const Comparison with = compare(a, b, acc);

  TrapResult t;
  t.spec = spec("T5");
  t.layer = kLayer;
  t.wrong = std::string("compare(fp32, fp16): ") + to_string(without.status);
  char buf[256];
  if (with.status == ComparisonStatus::Ok)
    std::snprintf(buf, sizeof buf,
                  "compare(fp32, fp16, accuracy): ok, p50 speedup %.2fx, max rel err %.1e -> %.1e",
                  with.speedup_p50, acc.baseline, acc.candidate);
  else
    std::snprintf(buf, sizeof buf, "compare(fp32, fp16, accuracy): %s (max rel err %.1e -> %.1e)",
                  to_string(with.status), acc.baseline, acc.candidate);
  t.control = buf;
  const bool refused = without.status != ComparisonStatus::Ok && without.speedup_p50 == 0;
  t.detail = without.reason;
  if (!refused) {
    t.outcome = TrapOutcome::Fail;
    t.detail = "a speedup was reported without an accuracy delta";
  } else if (with.status == ComparisonStatus::Refused) {
    t.outcome = TrapOutcome::Fail;
    t.detail = "control refused even with an accuracy delta: " + with.reason;
  } else {
    t.outcome = TrapOutcome::Pass;
    if (with.status == ComparisonStatus::Untrustworthy)
      t.detail += " | control inconclusive: " + with.reason;
  }
  return t;
}

using TrapFn = TrapResult (*)(DeviceContext&);

}  // namespace

std::vector<TrapResult> run_gpu_traps(DeviceContext& ctx) {
  const std::pair<const char*, TrapFn> traps[] = {
      {"T1", t1}, {"T2", t2}, {"T3", t3}, {"T4", t4}, {"T4r", t4r}, {"T5", t5}};
  std::vector<TrapResult> out;
  for (const auto& [id, fn] : traps) {
    try {
      out.push_back(fn(ctx));
    } catch (const std::exception& e) {
      // A trap that crashes is a failed self-test, not a skipped one.
      TrapResult t;
      t.spec = spec(id);
      t.layer = kLayer;
      t.outcome = TrapOutcome::Fail;
      t.wrong = t.control = "-";
      t.detail = std::string("exception: ") + e.what();
      out.push_back(t);
    }
  }
  return out;
}

}  // namespace trtbench::cuda
