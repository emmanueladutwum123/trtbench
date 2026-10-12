#pragma once

#include <cstddef>
#include <memory>

#include "trtbench/cuda/workload.hpp"

namespace trtbench::cuda {

// An empty kernel: the launch-overhead floor. No declared work.
std::unique_ptr<Workload> make_empty_kernel();

// Device-to-device float4 copy of `bytes` (rounded down to 16). Declares
// 2 x bytes of DRAM traffic. Used for the measured bandwidth peak; it must be
// much larger than L2 to measure DRAM rather than cache.
std::unique_ptr<Workload> make_copy(std::size_t bytes);

// Register-only FP32 FMA chains (8 independent chains per thread), one store
// per thread. Compute-bound by construction; output checked bit-exactly
// against a host replica.
std::unique_ptr<Workload> make_fma(int iters_per_thread);

}  // namespace trtbench::cuda
