#pragma once

// Plain device options with no CUDA dependency, so the CLI can parse them in
// CPU-only builds too.

namespace trtbench::cuda {

struct ContextOptions {
  int device = 0;
  bool lock_clocks = false;
  unsigned lock_mhz = 0;   // 0 = NVML default application clock (the sustainable base clock)
  bool calibrate = true;   // measure launch floors and the copy-bandwidth peak
};

}  // namespace trtbench::cuda
