#pragma once

#include <string>

namespace trtbench {

// Latency: one launch per sample, then a synchronize, so every sample starts
//          with an empty queue. Answers "how long does one call take?".
// Throughput: launches_per_sample launches back to back between one event
//          pair. The queue stays full, so per-launch host overhead overlaps
//          with device work. Answers "how many calls per second?".
// These are different questions; results from different modes are never
// compared (see compare.hpp).
enum class Mode { Latency, Throughput };

enum class Precision { FP32, FP16, BF16, INT8, Mixed };

const char* to_string(Mode m);
const char* to_string(Precision p);

struct BenchConfig {
  Mode mode = Mode::Latency;

  // Warmup runs until BOTH limits are met. The time floor matters: 50
  // iterations of a 5 us kernel is 250 us, far too short for the GPU to leave
  // its idle clock state, so an iteration count alone does not warm anything.
  int warmup_iters = 50;
  double warmup_min_ms = 250.0;

  // Steady state also runs until BOTH limits are met, capped by max_iters.
  int min_iters = 200;
  double min_time_ms = 1000.0;
  int max_iters = 100000;

  int launches_per_sample = 100;  // Throughput mode only; Latency is always 1.

  // Overwrite a buffer of 2x the L2 size before each sample (outside the
  // timed window) so the op starts with a cold L2. Off by default; when off,
  // the auditor still catches results that only make sense if data was
  // L2-resident (bandwidth above DRAM peak).
  bool flush_l2 = false;

  int clock_sample_every = 25;   // NVML clock snapshot every N steady-state samples.
  double clock_drift_tol = 0.05; // (max - min) / max over steady state.
};

// Returns an empty string if the config is usable, else the reason it is not.
std::string validate(const BenchConfig& cfg);

}  // namespace trtbench
