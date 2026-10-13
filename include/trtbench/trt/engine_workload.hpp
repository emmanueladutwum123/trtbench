#pragma once

#include <memory>
#include <string>

#include "trtbench/cuda/workload.hpp"

namespace trtbench::trt {

// A serialized TensorRT engine as a harness workload.
//
// It is described by a meta JSON file written by python/trtbench/engines.py:
// the engine path, the precision it was built at, the analytic FLOPs and
// lower-bound DRAM bytes of one inference, the input tensors (raw binary
// files of real preprocessed data), and the FP32 PyTorch reference outputs for
// those inputs.
//
// With cuda_graph = true, setup() captures one enqueueV3 into a CUDA graph and
// enqueue() replays it with a single cudaGraphLaunch. Comparing the two
// isolates per-launch host overhead: the device work is identical.
//
// consume_output() is a sanity check, not an accuracy measurement: outputs
// must be finite and have cosine similarity >= 0.99 with the FP32 reference.
// That catches garbage (unbound buffers, wrong bindings, a broken engine).
// Accuracy deltas are measured separately in python/trtbench/accuracy.py.
std::unique_ptr<cuda::Workload> make_engine_workload(const std::string& meta_path,
                                                     bool cuda_graph);

// Accuracy support: runs every batch of the evaluation set named in the meta
// ("eval_inputs": per-input files of N samples, "eval_count": N) through the
// engine and writes all outputs, FP32, concatenated, to out_path. Untimed.
// Returns the number of samples written.
std::size_t evaluate_engine(const std::string& meta_path, const std::string& out_path);

}  // namespace trtbench::trt
