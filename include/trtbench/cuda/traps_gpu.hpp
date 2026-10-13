#pragma once

#include <vector>

#include "trtbench/cuda/device_context.hpp"
#include "trtbench/traps.hpp"

namespace trtbench::cuda {

// GPU layer of the trap suite: each wrong benchmark really runs on the
// device. T4r needs permission to change clocks and skips without it.
std::vector<TrapResult> run_gpu_traps(DeviceContext& ctx);

}  // namespace trtbench::cuda
