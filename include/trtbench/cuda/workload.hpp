#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "trtbench/audit.hpp"
#include "trtbench/config.hpp"

namespace trtbench::cuda {

// What a measured op must declare. Bytes and FLOPs are analytic (what the op
// must do at minimum), not measured; the auditor uses them to catch timings
// that imply more work per second than the hardware can do.
class Workload {
 public:
  virtual ~Workload() = default;

  virtual std::string name() const = 0;
  virtual Precision precision() const { return Precision::FP32; }

  // Allocate and initialize. May enqueue work on `s`; the runner synchronizes
  // after setup so none of it leaks into a timed sample.
  virtual void setup(cudaStream_t s) = 0;

  // Enqueue one invocation. Must be asynchronous: no synchronize, no blocking
  // copy, no allocation. The runner times this call on the host and the
  // device work it produces with events.
  virtual void enqueue(cudaStream_t s) = 0;

  // Minimum DRAM bytes read + written per invocation.
  virtual std::uint64_t bytes_moved() const = 0;
  virtual std::uint64_t flops() const = 0;

  // Read the output after all timing is done and check it against a
  // reference. Reading it is what makes the work observable; checking it is
  // what proves the work happened.
  virtual Checksum consume_output(cudaStream_t s) = 0;
};

}  // namespace trtbench::cuda
