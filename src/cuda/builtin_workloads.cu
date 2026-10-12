#include "trtbench/cuda/builtin_workloads.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "trtbench/cuda/check.hpp"

namespace trtbench::cuda {

namespace {

int device_attr(cudaDeviceAttr a) {
  int dev = 0, v = 0;
  TB_CUDA_CHECK(cudaGetDevice(&dev));
  TB_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

// ---------------------------------------------------------------------------
// Empty kernel
// ---------------------------------------------------------------------------
__global__ void empty_kernel() {}

class EmptyKernel final : public Workload {
 public:
  std::string name() const override { return "empty_kernel"; }
  void setup(cudaStream_t) override {}
  void enqueue(cudaStream_t s) override { empty_kernel<<<1, 1, 0, s>>>(); }
  std::uint64_t bytes_moved() const override { return 0; }
  std::uint64_t flops() const override { return 0; }
  Checksum consume_output(cudaStream_t) override {
    return {false, false, 0, "empty kernel has no output by design"};
  }
};

// ---------------------------------------------------------------------------
// Copy: the bandwidth reference
// ---------------------------------------------------------------------------
__device__ __host__ inline float copy_pattern(std::size_t i) {
  return static_cast<float>(i & 0xFFFFFu);  // exact in float (< 2^24)
}

__global__ void fill_kernel(float* p, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride)
    p[i] = copy_pattern(i);
}

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            std::size_t n4) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride)
    out[i] = in[i];
}

class Copy final : public Workload {
 public:
  explicit Copy(std::size_t bytes) : n4_(bytes / sizeof(float4)) {
    if (n4_ == 0) throw std::invalid_argument("copy size must be at least 16 bytes");
  }
  ~Copy() override {
    cudaFree(in_);
    cudaFree(out_);
  }
  std::string name() const override {
    return "copy_" + std::to_string(n4_ * sizeof(float4) >> 20) + "MiB";
  }
  void setup(cudaStream_t s) override {
    if (in_ == nullptr) {
      TB_CUDA_CHECK(cudaMalloc(&in_, n4_ * sizeof(float4)));
      TB_CUDA_CHECK(cudaMalloc(&out_, n4_ * sizeof(float4)));
    }
    const int sms = device_attr(cudaDevAttrMultiProcessorCount);
    const int per_sm = device_attr(cudaDevAttrMaxThreadsPerMultiProcessor) / kThreads;
    // Grid-stride: enough blocks to fill every SM several times over, no more.
    const std::size_t want = (n4_ + kThreads - 1) / kThreads;
    blocks_ = static_cast<unsigned>(std::min<std::size_t>(want, std::size_t(sms) * per_sm * 4));
    fill_kernel<<<blocks_, kThreads, 0, s>>>(reinterpret_cast<float*>(in_), n4_ * 4);
    TB_CUDA_CHECK(cudaGetLastError());
    TB_CUDA_CHECK(cudaMemsetAsync(out_, 0, n4_ * sizeof(float4), s));
  }
  void enqueue(cudaStream_t s) override {
    copy_kernel<<<blocks_, kThreads, 0, s>>>(in_, out_, n4_);
  }
  std::uint64_t bytes_moved() const override { return 2ull * n4_ * sizeof(float4); }
  std::uint64_t flops() const override { return 0; }

  Checksum consume_output(cudaStream_t s) override {
    // Gather kSamples floats spread across the whole output with one strided
    // 2D copy, and check each against the input pattern.
    const std::size_t n = n4_ * 4;
    const std::size_t rows = std::min<std::size_t>(kSamples, n);
    const std::size_t stride = n / rows;
    std::vector<float> got(rows);
    TB_CUDA_CHECK(cudaMemcpy2DAsync(got.data(), sizeof(float), out_, stride * sizeof(float),
                                    sizeof(float), rows, cudaMemcpyDeviceToHost, s));
    float last = 0;
    TB_CUDA_CHECK(cudaMemcpyAsync(&last, reinterpret_cast<float*>(out_) + (n - 1), sizeof(float),
                                  cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));

    Checksum c{true, true, 0, ""};
    for (std::size_t r = 0; r < rows; ++r) {
      c.value += got[r];
      if (got[r] != copy_pattern(r * stride) && c.passed) {
        c.passed = false;
        c.note = "out[" + std::to_string(r * stride) + "] = " + std::to_string(got[r]) +
                 ", expected " + std::to_string(copy_pattern(r * stride));
      }
    }
    if (last != copy_pattern(n - 1) && c.passed) {
      c.passed = false;
      c.note = "last element not copied";
    }
    if (c.passed) c.note = "exact match at " + std::to_string(rows + 1) + " sampled positions";
    return c;
  }

 private:
  static constexpr unsigned kThreads = 256;
  static constexpr std::size_t kSamples = 4096;
  std::size_t n4_;
  float4* in_ = nullptr;
  float4* out_ = nullptr;
  unsigned blocks_ = 0;
};

// ---------------------------------------------------------------------------
// FMA: the compute reference
// ---------------------------------------------------------------------------
constexpr int kChains = 8;
constexpr float kA = 0.9999f;
constexpr float kB = 0.0001f;

// Initial values are exact in float: (t & 1023) * 2^-10 + k involves no
// rounding, so it does not matter whether a compiler contracts it into an FMA.
__device__ __host__ inline float fma_init(unsigned t, int k) {
  return static_cast<float>(t & 1023u) * 0.0009765625f + static_cast<float>(k);
}

// Explicit summation order, adds only: identical on host and device.
__device__ __host__ inline float fma_reduce(const float* x) {
  return ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
}

__global__ void fma_kernel(float* out, int iters, float a, float b) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  float x[kChains];
#pragma unroll
  for (int k = 0; k < kChains; ++k) x[k] = fma_init(t, k);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < kChains; ++k) x[k] = fmaf(x[k], a, b);
  }
  // The single store is what keeps the compiler from deleting the loop.
  out[t] = fma_reduce(x);
}

float fma_reference(unsigned t, int iters) {
  float x[kChains];
  for (int k = 0; k < kChains; ++k) x[k] = fma_init(t, k);
  for (int i = 0; i < iters; ++i)
    for (int k = 0; k < kChains; ++k) x[k] = std::fma(x[k], kA, kB);  // correctly rounded, like fmaf
  return fma_reduce(x);
}

class Fma final : public Workload {
 public:
  explicit Fma(int iters) : iters_(iters) {
    if (iters < 1) throw std::invalid_argument("fma iters must be >= 1");
  }
  ~Fma() override { cudaFree(out_); }
  std::string name() const override { return "fma_fp32_x" + std::to_string(iters_); }
  void setup(cudaStream_t s) override {
    const int sms = device_attr(cudaDevAttrMultiProcessorCount);
    const int per_sm = device_attr(cudaDevAttrMaxThreadsPerMultiProcessor) / kThreads;
    // Exactly one full wave: every SM fully occupied, no partial tail wave
    // to drag the average down.
    blocks_ = static_cast<unsigned>(sms * per_sm);
    threads_total_ = static_cast<std::size_t>(blocks_) * kThreads;
    if (out_ == nullptr) TB_CUDA_CHECK(cudaMalloc(&out_, threads_total_ * sizeof(float)));
    TB_CUDA_CHECK(cudaMemsetAsync(out_, 0, threads_total_ * sizeof(float), s));
  }
  void enqueue(cudaStream_t s) override {
    fma_kernel<<<blocks_, kThreads, 0, s>>>(out_, iters_, kA, kB);
  }
  std::uint64_t bytes_moved() const override { return threads_total_ * sizeof(float); }
  // FMAs only; the 7 adds of the final reduction per thread are not counted.
  std::uint64_t flops() const override {
    return static_cast<std::uint64_t>(threads_total_) * static_cast<std::uint64_t>(iters_) *
           kChains * 2;
  }
  Checksum consume_output(cudaStream_t s) override {
    std::vector<float> got(threads_total_);
    TB_CUDA_CHECK(cudaMemcpyAsync(got.data(), out_, got.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    // Host replica is iters x 8 software FMAs per thread; check 128 threads
    // spread across the grid rather than all of them.
    const std::size_t step = std::max<std::size_t>(1, threads_total_ / 128);
    Checksum c{true, true, 0, ""};
    std::size_t checked = 0;
    for (std::size_t t = 0; t < threads_total_; t += step, ++checked) {
      const float want = fma_reference(static_cast<unsigned>(t), iters_);
      c.value += got[t];
      if (got[t] != want && c.passed) {
        c.passed = false;
        char buf[160];
        std::snprintf(buf, sizeof buf, "thread %zu: got %.9g, expected %.9g (bit-exact required)",
                      t, static_cast<double>(got[t]), static_cast<double>(want));
        c.note = buf;
      }
    }
    if (c.passed) c.note = "bit-exact at " + std::to_string(checked) + " sampled threads";
    return c;
  }

 private:
  static constexpr unsigned kThreads = 256;
  int iters_;
  float* out_ = nullptr;
  unsigned blocks_ = 0;
  std::size_t threads_total_ = 0;
};

}  // namespace

std::unique_ptr<Workload> make_empty_kernel() { return std::make_unique<EmptyKernel>(); }
std::unique_ptr<Workload> make_copy(std::size_t bytes) { return std::make_unique<Copy>(bytes); }
std::unique_ptr<Workload> make_fma(int iters) { return std::make_unique<Fma>(iters); }

}  // namespace trtbench::cuda
