#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "trtbench/clocks.hpp"
#include "trtbench/context_options.hpp"
#include "trtbench/cuda/clock_lock.hpp"
#include "trtbench/result.hpp"

namespace trtbench::cuda {

// Number of usable CUDA devices; 0 (with `why` set) when there is no device
// or no driver. Never throws.
int cuda_device_count(std::string* why = nullptr);

// Owns everything that must exist before any timing starts: the CUDA context,
// a stream, the timing events, the L2 flush buffer, clock observation and
// locking, and the calibrated peaks every result is judged against.
class DeviceContext {
 public:
  static std::unique_ptr<DeviceContext> create(const ContextOptions& opts);
  ~DeviceContext();
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  cudaEvent_t start_event() const { return start_; }
  cudaEvent_t stop_event() const { return stop_; }
  void* flush_buffer() const { return flush_buf_; }
  std::size_t flush_bytes() const { return flush_bytes_; }

  ClockSource& clocks() { return *clocks_; }
  bool clocks_locked() const { return (lock_ && lock_->locked()) || external_lock_mhz_ > 0; }
  const std::string& lock_status() const { return lock_status_; }
  unsigned locked_mhz() const {
    if (lock_ && lock_->locked()) return lock_->mhz();
    return external_lock_mhz_;
  }
  int nvml_index() const { return nvml_index_; }  // -1 without NVML

  const Environment& env() const { return env_; }
  const DevicePeaks& peaks() const { return peaks_; }
  double launch_floor_ms(Mode m) const {
    return m == Mode::Latency ? floor_latency_ms_ : floor_throughput_ms_;
  }
  const std::vector<Result>& calibration() const { return calibration_; }

  RunReport make_report(std::vector<Result> results) const;

 private:
  DeviceContext() = default;
  void query_environment();
  void calibrate();

  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  void* flush_buf_ = nullptr;
  std::size_t flush_bytes_ = 0;

  std::unique_ptr<ClockSource> clocks_;
  std::unique_ptr<ClockLock> lock_;
  std::string lock_status_ = "not requested";
  int nvml_index_ = -1;
  unsigned external_lock_mhz_ = 0;

  Environment env_;
  DevicePeaks peaks_;
  double floor_latency_ms_ = 0;
  double floor_throughput_ms_ = 0;
  std::vector<Result> calibration_;
};

}  // namespace trtbench::cuda
