#include "trtbench/cuda/device_context.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>

#include "trtbench/cuda/builtin_workloads.hpp"
#include "trtbench/cuda/check.hpp"
#include "trtbench/cuda/nvml.hpp"
#include "trtbench/cuda/runner.hpp"

namespace trtbench::cuda {

int cuda_device_count(std::string* why) {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    if (why) *why = std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
    cudaGetLastError();  // clear, so later calls do not inherit it
    return 0;
  }
  if (n == 0 && why) *why = "cudaGetDeviceCount returned 0";
  return n;
}

namespace {
int attr(cudaDeviceAttr a, int dev) {
  int v = -1;
  if (cudaDeviceGetAttribute(&v, a, dev) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return v;
}

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}
}  // namespace

std::unique_ptr<DeviceContext> DeviceContext::create(const ContextOptions& o) {
  std::string why;
  const int n = cuda_device_count(&why);
  if (n == 0) throw CudaError("no usable CUDA device: " + why);
  if (o.device < 0 || o.device >= n)
    throw std::invalid_argument("device " + std::to_string(o.device) + " out of range (have " +
                                std::to_string(n) + ")");

  std::unique_ptr<DeviceContext> c(new DeviceContext());
  c->device_ = o.device;
  TB_CUDA_CHECK(cudaSetDevice(o.device));
  // Deliberate: force context creation here. Lazy context creation inside the
  // first timed call is the classic first-iteration outlier.
  TB_CUDA_CHECK(cudaFree(nullptr));

  c->query_environment();

  // Clock observation (and with it, the device's NVML index for locking).
  std::string nvml_why;
  if (auto src = NvmlClockSource::open(c->env_.pci_bus_id, &nvml_why)) {
    const NvmlDeviceInfo& info = src->info();
    c->nvml_index_ = info.index;
    c->env_.driver_version = info.driver_version;
    c->env_.max_sm_clock_mhz = info.max_sm_clock_mhz;
    c->env_.default_app_sm_clock_mhz = info.default_app_sm_clock_mhz;
    c->env_.persistence_mode = info.persistence_mode;
    c->env_.ecc_enabled = info.ecc_enabled;
    c->env_.power_limit_w = info.power_limit_w;
    c->env_.nvml_status = "ok";
    c->clocks_ = std::move(src);
  } else {
    c->env_.nvml_status = nvml_why;
    c->clocks_ = std::make_unique<NullClockSource>(nvml_why);
  }
  if (c->env_.max_sm_clock_mhz < 0) {
    const int khz = attr(cudaDevAttrClockRate, o.device);
    if (khz > 0) c->env_.max_sm_clock_mhz = khz / 1000;
  }

  if (o.external_lock_mhz > 0) {
    if (o.lock_clocks) throw std::invalid_argument("--lock-clocks and an external lock are exclusive");
    c->external_lock_mhz_ = o.external_lock_mhz;
    c->lock_status_ = "locked externally at " + std::to_string(o.external_lock_mhz) +
                      " MHz (by the calling script, not this process)";
  }
  if (o.lock_clocks) {
    c->lock_ = std::make_unique<ClockLock>(c->nvml_index_, o.lock_mhz);
    c->lock_status_ = c->lock_->status();
  }

  TB_CUDA_CHECK(cudaStreamCreateWithFlags(&c->stream_, cudaStreamNonBlocking));
  // Default event flags: timing enabled, spin-wait synchronization. Spin-wait
  // keeps the host wake-up after cudaEventSynchronize short, which keeps the
  // wall-clock cross-check tight.
  TB_CUDA_CHECK(cudaEventCreate(&c->start_));
  TB_CUDA_CHECK(cudaEventCreate(&c->stop_));

  if (c->env_.l2_bytes > 0) {
    c->flush_bytes_ = static_cast<std::size_t>(c->env_.l2_bytes) * 2;
    TB_CUDA_CHECK(cudaMalloc(&c->flush_buf_, c->flush_bytes_));
  }

  c->peaks_ = theoretical_peaks(c->env_.cc_major, c->env_.cc_minor, c->env_.sm_count,
                                static_cast<unsigned>(std::max(0, c->env_.max_sm_clock_mhz)),
                                c->env_.mem_clock_khz, c->env_.mem_bus_width_bits);
  if (o.calibrate) c->calibrate();
  return c;
}

DeviceContext::~DeviceContext() {
  // Best effort: destructors must not throw, and a failed free at exit
  // changes nothing about the measurements already taken.
  if (flush_buf_) cudaFree(flush_buf_);
  if (start_) cudaEventDestroy(start_);
  if (stop_) cudaEventDestroy(stop_);
  if (stream_) cudaStreamDestroy(stream_);
  lock_.reset();  // resets clocks via nvidia-smi -rgc if we locked them
}

void DeviceContext::query_environment() {
  const int d = device_;
  cudaDeviceProp prop{};
  TB_CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
  env_.gpu_name = prop.name;
  char pci[32] = {};
  TB_CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof pci, d));
  env_.pci_bus_id = pci;
  env_.cc_major = attr(cudaDevAttrComputeCapabilityMajor, d);
  env_.cc_minor = attr(cudaDevAttrComputeCapabilityMinor, d);
  env_.sm_count = attr(cudaDevAttrMultiProcessorCount, d);
  env_.max_threads_per_sm = attr(cudaDevAttrMaxThreadsPerMultiProcessor, d);
  env_.l2_bytes = attr(cudaDevAttrL2CacheSize, d);
  env_.mem_bus_width_bits = attr(cudaDevAttrGlobalMemoryBusWidth, d);
  env_.mem_clock_khz = attr(cudaDevAttrMemoryClockRate, d);

  int v = 0;
  if (cudaDriverGetVersion(&v) == cudaSuccess) env_.cuda_driver_api_version = v;
  if (cudaRuntimeGetVersion(&v) == cudaSuccess) env_.cuda_runtime_version = v;

  const char* ml = std::getenv("CUDA_MODULE_LOADING");
  env_.cuda_module_loading = ml ? ml : "(unset: driver default)";
  char host[256] = {};
  if (gethostname(host, sizeof host - 1) == 0) env_.hostname = host;
  env_.timestamp_utc = utc_now();
  env_.trtbench_version = build_version();
  env_.git_sha = build_git_sha();
  env_.compiler = build_compiler();
}

void DeviceContext::calibrate() {
  // 1. Launch floors: an empty kernel in each mode. A "real" op whose time is
  //    within 10% of this is measuring launch overhead.
  BenchConfig lat;
  lat.mode = Mode::Latency;
  lat.min_iters = 1000;
  lat.min_time_ms = 500;
  auto empty = make_empty_kernel();
  Result r = run(*empty, lat, *this);
  r.name = "calibration/empty_kernel_latency";
  floor_latency_ms_ = r.device_ms.p50;
  calibration_.push_back(std::move(r));

  BenchConfig thr = lat;
  thr.mode = Mode::Throughput;
  thr.launches_per_sample = 100;
  thr.min_iters = 200;
  r = run(*empty, thr, *this);
  r.name = "calibration/empty_kernel_throughput";
  floor_throughput_ms_ = r.device_ms.p50;
  calibration_.push_back(std::move(r));

  // 2. Achievable DRAM bandwidth: a copy far larger than L2, capped at a
  //    quarter of free memory.
  std::size_t free_b = 0, total_b = 0;
  TB_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  std::size_t bytes = std::max<std::size_t>(std::size_t{256} << 20,
                                            static_cast<std::size_t>(std::max(0LL, env_.l2_bytes)) * 16);
  bytes = std::min(bytes, free_b / 4);
  BenchConfig bw;
  bw.mode = Mode::Throughput;
  bw.launches_per_sample = 10;
  bw.min_iters = 100;
  bw.min_time_ms = 1000;
  auto copy = make_copy(bytes);
  r = run(*copy, bw, *this);
  r.name = "calibration/" + r.name;
  if (r.trustworthy()) {
    peaks_.dram_gbps_measured = r.roofline.achieved_gbps;
    peaks_.source += "; measured DRAM = p50 of " + r.name;
  } else {
    peaks_.source += "; measured DRAM peak REJECTED by audit, % of peak uses theoretical";
  }
  calibration_.push_back(std::move(r));
}

RunReport DeviceContext::make_report(std::vector<Result> results) const {
  RunReport rep;
  rep.env = env_;
  rep.peaks = peaks_;
  rep.launch_floor_latency_ms = floor_latency_ms_;
  rep.launch_floor_throughput_ms = floor_throughput_ms_;
  rep.calibration = calibration_;
  rep.results = std::move(results);
  return rep;
}

}  // namespace trtbench::cuda
