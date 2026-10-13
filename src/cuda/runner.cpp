#include "trtbench/cuda/runner.hpp"

#include <chrono>
#include <stdexcept>

#include "trtbench/cuda/check.hpp"

namespace trtbench::cuda {

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Sample {
  double device_ms;  // per launch
  double host_ms;    // per launch
  double wall_ms;    // per launch
};

// One timed sample.
//
//   t0 = now
//   record(start)
//   h0 = now; enqueue x N; h1 = now        <- host_ms: cost of the enqueue calls only
//   record(stop)
//   sync(stop); t2 = now                   <- wall_ms: t0..t2, encloses the device interval
//   device_ms = elapsed(start, stop)
//
// t0 is taken before record(start), so the device interval is strictly inside
// [t0, t2]. wall_ms < device_ms is therefore impossible for a correct
// measurement, and the auditor treats it as one.
Sample measure(Workload& w, DeviceContext& ctx, int launches, bool flush, int iter) {
  cudaStream_t s = ctx.stream();
  if (flush && ctx.flush_bytes() > 0) {
    TB_CUDA_CHECK(cudaMemsetAsync(ctx.flush_buffer(), iter & 0xff, ctx.flush_bytes(), s));
    // DELIBERATE SYNC: the flush must finish before the timed window opens,
    // or the start event would wait behind it and wall_ms would include it.
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
  }

  const auto t0 = Clock::now();
  TB_CUDA_CHECK(cudaEventRecord(ctx.start_event(), s));
  const auto h0 = Clock::now();
  for (int k = 0; k < launches; ++k) w.enqueue(s);  // NO SYNC: enqueue must be async
  const auto h1 = Clock::now();
  TB_CUDA_CHECK(cudaEventRecord(ctx.stop_event(), s));
  // DELIBERATE SYNC: elapsed time is only readable once stop has completed,
  // and it keeps samples from overlapping.
  TB_CUDA_CHECK(cudaEventSynchronize(ctx.stop_event()));
  const auto t2 = Clock::now();
  // Launch errors from enqueue() surface here, once per sample, outside the
  // timed region.
  TB_CUDA_CHECK(cudaGetLastError());

  float ev_ms = 0;
  TB_CUDA_CHECK(cudaEventElapsedTime(&ev_ms, ctx.start_event(), ctx.stop_event()));
  const double n = launches;
  return {ev_ms / n, ms_between(h0, h1) / n, ms_between(t0, t2) / n};
}

}  // namespace

Result run(Workload& w, const BenchConfig& cfg, DeviceContext& ctx) {
  return run(w, cfg, ctx, ctx.clocks());
}

Result run(Workload& w, const BenchConfig& cfg, DeviceContext& ctx, ClockSource& clocks) {
  if (const std::string err = validate(cfg); !err.empty()) throw std::invalid_argument(err);
  const int launches = cfg.mode == Mode::Latency ? 1 : cfg.launches_per_sample;
  cudaStream_t s = ctx.stream();

  w.setup(s);
  // DELIBERATE SYNC: setup may enqueue allocation/initialization work; none
  // of it may land inside the first timed sample.
  TB_CUDA_CHECK(cudaStreamSynchronize(s));
  TB_CUDA_CHECK(cudaGetLastError());

  const auto origin = Clock::now();
  auto snap = [&] {
    ClockSnapshot c = clocks.sample();
    c.t_ms = ms_between(origin, Clock::now());
    return c;
  };

  const ClockSnapshot pre = clocks.available() ? snap() : ClockSnapshot{};

  RawSamples raw;

  // First call: always one launch, always recorded on its own and never
  // folded into the percentiles. It carries lazy module loading and any
  // first-touch cost.
  {
    const Sample f = measure(w, ctx, 1, cfg.flush_l2, 0);
    raw.first_device_ms = f.device_ms;
    raw.first_host_ms = f.host_ms;
    raw.first_wall_ms = f.wall_ms;
  }

  // Warmup: until BOTH the iteration count and the time floor are met.
  {
    const auto w0 = Clock::now();
    for (int i = 0; i < cfg.warmup_iters || ms_between(w0, Clock::now()) < cfg.warmup_min_ms;
         ++i) {
      if (cfg.warmup_iters == 0) break;  // explicit no-warmup (trap suite) means none
      measure(w, ctx, launches, cfg.flush_l2, i);
    }
  }

  // Steady state.
  std::vector<ClockSnapshot> steady;
  if (clocks.available()) steady.push_back(snap());
  const auto s0 = Clock::now();
  for (int i = 0; i < cfg.max_iters; ++i) {
    if (i >= cfg.min_iters && ms_between(s0, Clock::now()) >= cfg.min_time_ms) break;
    const Sample m = measure(w, ctx, launches, cfg.flush_l2, i);
    raw.device_ms.push_back(m.device_ms);
    raw.host_enqueue_ms.push_back(m.host_ms);
    raw.wall_ms.push_back(m.wall_ms);
    if (clocks.available() && cfg.clock_sample_every > 0 && (i + 1) % cfg.clock_sample_every == 0)
      steady.push_back(snap());  // between samples, outside any timed window
  }
  if (clocks.available()) steady.push_back(snap());

  Checksum ck = w.consume_output(s);
  TB_CUDA_CHECK(cudaStreamSynchronize(s));  // DELIBERATE: consume_output may be async

  ClockReport cr = summarize_clocks(clocks.available(), clocks.name(), pre, std::move(steady));
  cr.locked = ctx.clocks_locked();
  cr.lock_status = ctx.lock_status();

  FinalizeInput in;
  in.name = w.name();
  in.precision = w.precision();
  in.config = cfg;
  in.bytes = w.bytes_moved();
  in.flops = w.flops();
  in.raw = std::move(raw);
  in.clocks = std::move(cr);
  in.peaks = ctx.peaks();
  in.launch_floor_ms = ctx.launch_floor_ms(cfg.mode);
  in.checksum = std::move(ck);
  return finalize(std::move(in));
}

}  // namespace trtbench::cuda
