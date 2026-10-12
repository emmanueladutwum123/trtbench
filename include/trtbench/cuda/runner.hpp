#pragma once

#include "trtbench/config.hpp"
#include "trtbench/cuda/device_context.hpp"
#include "trtbench/cuda/workload.hpp"
#include "trtbench/result.hpp"

namespace trtbench::cuda {

// setup -> first call (recorded separately) -> warmup -> steady state ->
// consume output -> finalize (percentiles, roofline, audit).
//
// Every synchronization point is documented in runner.cpp. Throws CudaError
// on any CUDA failure, std::invalid_argument on an invalid config.
Result run(Workload& w, const BenchConfig& cfg, DeviceContext& ctx);

}  // namespace trtbench::cuda
