#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "trtbench/clocks.hpp"
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

// Phase 3 support: per-layer attribution of one engine. Untimed by the
// harness runner; it reports
//   - per-layer times from TensorRT's IProfiler over `runs` inferences,
//   - end-to-end device time of the same inferences (profiler attached) and of
//     `runs` inferences without the profiler, so the profiler's own cost is
//     stated instead of hidden in the layer times,
//   - SM/memory clocks over the run (same drift summary as the runner),
//   - the engine inspector's JSON description of every layer (fused ONNX
//     nodes, formats, tactics; needs an engine built with kDETAILED).
nlohmann::json profile_engine(const std::string& meta_path, int warmup, int runs,
                              ClockSource& clocks);

}  // namespace trtbench::trt
