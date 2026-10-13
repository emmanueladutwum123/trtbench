#pragma once

// Plain device options with no CUDA dependency, so the CLI can parse them in
// CPU-only builds too.

namespace trtbench::cuda {

struct ContextOptions {
  int device = 0;
  bool lock_clocks = false;
  // Required when lock_clocks is set. There is no safe default: NVML's
  // "default application clock" is the base clock (585 MHz on a T4, a third
  // of what it actually runs at), and the boost clock is not sustainable
  // under a power cap. Pick a clock at or below the minimum an unlocked run
  // observed under load (clocks.sm_mhz.min in its JSON).
  unsigned lock_mhz = 0;
  // The clock was locked by the caller (a script that also runs other
  // processes under the same lock). Recorded in every result; the drift check
  // still verifies the clocks actually held.
  unsigned external_lock_mhz = 0;
  bool calibrate = true;   // measure launch floors and the copy-bandwidth peak
};

}  // namespace trtbench::cuda
