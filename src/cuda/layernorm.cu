#include "trtbench/cuda/layernorm.hpp"

#include <cuda_fp16.h>

#include <stdexcept>

#include "trtbench/cuda/check.hpp"

namespace trtbench::cuda {

const char* to_string(LnVariant v) {
  switch (v) {
    case LnVariant::Naive: return "naive";
    case LnVariant::Block: return "block";
    case LnVariant::Warp: return "warp";
  }
  return "?";
}

namespace {

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }

// ---------------------------------------------------------------------------
// naive: one thread per row
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_ln_naive(const T* __restrict__ x, const T* __restrict__ r,
                             const T* __restrict__ gamma, const T* __restrict__ beta,
                             T* __restrict__ s_out, T* __restrict__ y, int rows, int hidden,
                             float eps) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  const std::size_t base = static_cast<std::size_t>(row) * hidden;
  float sum = 0;
  for (int j = 0; j < hidden; ++j) {
    const T s = from_f<T>(to_f(x[base + j]) + to_f(r[base + j]));
    s_out[base + j] = s;
    sum += to_f(s);
  }
  const float mean = sum / hidden;
  float var = 0;
  for (int j = 0; j < hidden; ++j) {
    const float d = to_f(s_out[base + j]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(var / hidden + eps);
  for (int j = 0; j < hidden; ++j)
    y[base + j] = from_f<T>((to_f(s_out[base + j]) - mean) * rstd * to_f(gamma[j]) + to_f(beta[j]));
}

// ---------------------------------------------------------------------------
// block: one block per row, shared-memory tree reduction
// ---------------------------------------------------------------------------
constexpr int kBlockThreads = 256;

__device__ float block_sum(float v, float* smem) {
  smem[threadIdx.x] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = smem[0];
  __syncthreads();  // smem is reused by the next reduction
  return total;
}

template <typename T>
__global__ void add_ln_block(const T* __restrict__ x, const T* __restrict__ r,
                             const T* __restrict__ gamma, const T* __restrict__ beta,
                             T* __restrict__ s_out, T* __restrict__ y, int hidden, float eps) {
  __shared__ float smem[kBlockThreads];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float part = 0;
  for (int j = static_cast<int>(threadIdx.x); j < hidden; j += static_cast<int>(blockDim.x)) {
    const T s = from_f<T>(to_f(x[base + j]) + to_f(r[base + j]));
    s_out[base + j] = s;
    part += to_f(s);
  }
  const float mean = block_sum(part, smem) / hidden;
  part = 0;
  for (int j = static_cast<int>(threadIdx.x); j < hidden; j += static_cast<int>(blockDim.x)) {
    const float d = to_f(s_out[base + j]) - mean;  // this thread wrote it
    part += d * d;
  }
  const float rstd = rsqrtf(block_sum(part, smem) / hidden + eps);
  for (int j = static_cast<int>(threadIdx.x); j < hidden; j += static_cast<int>(blockDim.x))
    y[base + j] = from_f<T>((to_f(s_out[base + j]) - mean) * rstd * to_f(gamma[j]) + to_f(beta[j]));
}

// ---------------------------------------------------------------------------
// warp: one warp per row, 16-byte vectors, row in registers, shuffles
// ---------------------------------------------------------------------------
constexpr int kWarpRowsPerBlock = 4;

template <typename T>
struct alignas(16) Vec {
  static constexpr int N = 16 / sizeof(T);
  T v[N];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
  return v;  // every lane holds the total
}

template <typename T, int VECS>
__global__ void add_ln_warp(const T* __restrict__ x, const T* __restrict__ r,
                            const T* __restrict__ gamma, const T* __restrict__ beta,
                            T* __restrict__ s_out, T* __restrict__ y, int rows, float eps) {
  using V = Vec<T>;
  constexpr int N = V::N;
  constexpr int H = 32 * VECS * N;
  const int lane = static_cast<int>(threadIdx.x & 31u);
  const int row = static_cast<int>(blockIdx.x) * kWarpRowsPerBlock + static_cast<int>(threadIdx.x >> 5);
  if (row >= rows) return;  // whole warp exits together: shuffles stay full-mask
  const std::size_t base = static_cast<std::size_t>(row) * H;
  const V* xv = reinterpret_cast<const V*>(x + base);
  const V* rv = reinterpret_cast<const V*>(r + base);
  V* sv = reinterpret_cast<V*>(s_out + base);

  // Vector k of lane l is element block k*32 + l: consecutive lanes read
  // consecutive 16-byte chunks, so each warp load is 512 contiguous bytes.
  float v[VECS][N];
  float sum = 0;
#pragma unroll
  for (int k = 0; k < VECS; ++k) {
    const V a = xv[k * 32 + lane];
    const V b = rv[k * 32 + lane];
    V s;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      s.v[i] = from_f<T>(to_f(a.v[i]) + to_f(b.v[i]));
      v[k][i] = to_f(s.v[i]);
      sum += v[k][i];
    }
    sv[k * 32 + lane] = s;
  }
  const float mean = warp_sum(sum) / H;
  float var = 0;
#pragma unroll
  for (int k = 0; k < VECS; ++k)
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const float d = v[k][i] - mean;
      var += d * d;
    }
  const float rstd = rsqrtf(warp_sum(var) / H + eps);

  const V* gv = reinterpret_cast<const V*>(gamma);
  const V* bv = reinterpret_cast<const V*>(beta);
  V* yv = reinterpret_cast<V*>(y + base);
#pragma unroll
  for (int k = 0; k < VECS; ++k) {
    const V g = gv[k * 32 + lane];
    const V b = bv[k * 32 + lane];
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i)
      o.v[i] = from_f<T>((v[k][i] - mean) * rstd * to_f(g.v[i]) + to_f(b.v[i]));
    yv[k * 32 + lane] = o;
  }
}

template <typename T>
int warp_vecs(int hidden) {
  constexpr int per_vec_row = 32 * Vec<T>::N;
  if (hidden % per_vec_row != 0) return 0;
  const int vecs = hidden / per_vec_row;
  return vecs >= 1 && vecs <= 8 ? vecs : 0;
}

template <typename T>
void launch(LnVariant v, const T* x, const T* r, const T* g, const T* b, T* s, T* y, int rows,
            int hidden, cudaStream_t st) {
  const float eps = ln::kEps;
  switch (v) {
    case LnVariant::Naive: {
      constexpr int threads = 128;
      add_ln_naive<T><<<(rows + threads - 1) / threads, threads, 0, st>>>(x, r, g, b, s, y, rows,
                                                                         hidden, eps);
      return;
    }
    case LnVariant::Block:
      add_ln_block<T><<<rows, kBlockThreads, 0, st>>>(x, r, g, b, s, y, hidden, eps);
      return;
    case LnVariant::Warp: {
      const unsigned blocks = (rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock;
      const unsigned threads = 32 * kWarpRowsPerBlock;
      switch (warp_vecs<T>(hidden)) {
#define TB_LN_CASE(K)                                                                      \
  case K:                                                                                  \
    add_ln_warp<T, K><<<blocks, threads, 0, st>>>(x, r, g, b, s, y, rows, eps);            \
    return;
        TB_LN_CASE(1) TB_LN_CASE(2) TB_LN_CASE(3) TB_LN_CASE(4)
        TB_LN_CASE(5) TB_LN_CASE(6) TB_LN_CASE(7) TB_LN_CASE(8)
#undef TB_LN_CASE
        default: throw std::invalid_argument("warp kernel: unsupported hidden size");
      }
    }
  }
}

// Storage conversion between the reference's floats and device bits.
template <typename T>
std::vector<T> to_storage(const std::vector<float>& f);
template <>
std::vector<float> to_storage<float>(const std::vector<float>& f) { return f; }
template <>
std::vector<__half> to_storage<__half>(const std::vector<float>& f) {
  std::vector<__half> out(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) {
    __half_raw raw;
    raw.x = ln::float_to_half(f[i]);
    out[i] = __half(raw);
  }
  return out;
}
template <typename T>
std::vector<float> from_storage(const std::vector<T>& d);
template <>
std::vector<float> from_storage<float>(const std::vector<float>& d) { return d; }
template <>
std::vector<float> from_storage<__half>(const std::vector<__half>& d) {
  std::vector<float> out(d.size());
  for (std::size_t i = 0; i < d.size(); ++i) {
    const __half_raw raw = static_cast<__half_raw>(d[i]);
    out[i] = ln::half_to_float(raw.x);
  }
  return out;
}

template <typename T>
class AddLayerNorm final : public Workload {
 public:
  AddLayerNorm(LnVariant v, ln::DType t, int rows, int hidden)
      : v_(v), t_(t), rows_(rows), hidden_(hidden),
        n_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(hidden)) {
    if (rows < 1 || hidden < 1) throw std::invalid_argument("add_layernorm: empty shape");
    if (!ln_variant_supported(v, hidden, t))
      throw std::invalid_argument(std::string("add_layernorm: ") + to_string(v) +
                                  " kernel does not support hidden=" + std::to_string(hidden));
  }
  ~AddLayerNorm() override {
    for (T* p : {x_, r_, g_, b_, s_, y_}) cudaFree(p);
  }
  std::string name() const override {
    return std::string("add_layernorm/") + to_string(v_) + "/" + ln::to_string(t_) + "/" +
           std::to_string(rows_) + "x" + std::to_string(hidden_);
  }
  Precision precision() const override {
    return t_ == ln::DType::F16 ? Precision::FP16 : Precision::FP32;
  }
  std::map<std::string, std::string> labels() const override {
    return {{"backend", std::string("trtbench-") + to_string(v_)},
            {"kernel", to_string(v_)},
            {"dtype", ln::to_string(t_)},
            {"rows", std::to_string(rows_)},
            {"hidden", std::to_string(hidden_)}};
  }
  void setup(cudaStream_t s) override {
    if (x_ != nullptr) return;
    in_ = ln::make_inputs(rows_, hidden_, t_);
    auto upload = [&](T** dst, const std::vector<float>& src) {
      const std::vector<T> h = to_storage<T>(src);
      TB_CUDA_CHECK(cudaMalloc(dst, h.size() * sizeof(T)));
      TB_CUDA_CHECK(cudaMemcpyAsync(*dst, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, s));
      TB_CUDA_CHECK(cudaStreamSynchronize(s));  // h dies at scope end
    };
    upload(&x_, in_.x);
    upload(&r_, in_.r);
    upload(&g_, in_.gamma);
    upload(&b_, in_.beta);
    TB_CUDA_CHECK(cudaMalloc(&s_, n_ * sizeof(T)));
    TB_CUDA_CHECK(cudaMalloc(&y_, n_ * sizeof(T)));
    TB_CUDA_CHECK(cudaMemsetAsync(s_, 0xFF, n_ * sizeof(T), s));  // NaN until written
    TB_CUDA_CHECK(cudaMemsetAsync(y_, 0xFF, n_ * sizeof(T), s));
  }
  void enqueue(cudaStream_t s) override {
    launch<T>(v_, x_, r_, g_, b_, s_, y_, rows_, hidden_, s);
  }
  std::uint64_t bytes_moved() const override {
    return (4ull * n_ + 2ull * static_cast<std::uint64_t>(hidden_)) * sizeof(T);
  }
  std::uint64_t flops() const override { return 0; }
  Checksum consume_output(cudaStream_t s) override {
    TB_CUDA_CHECK(cudaGetLastError());
    std::vector<T> sd(n_), yd(n_);
    TB_CUDA_CHECK(cudaMemcpyAsync(sd.data(), s_, n_ * sizeof(T), cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaMemcpyAsync(yd.data(), y_, n_ * sizeof(T), cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    const std::vector<float> sf = from_storage<T>(sd), yf = from_storage<T>(yd);
    const ln::Check c = ln::check(in_, sf, yf);
    double total = 0;
    for (float v : yf) total += v;
    return {true, c.passed, total, c.note};
  }

 private:
  LnVariant v_;
  ln::DType t_;
  int rows_, hidden_;
  std::size_t n_;
  ln::Inputs in_;
  T *x_ = nullptr, *r_ = nullptr, *g_ = nullptr, *b_ = nullptr, *s_ = nullptr, *y_ = nullptr;
};

}  // namespace

bool ln_variant_supported(LnVariant v, int hidden, ln::DType t) {
  if (v != LnVariant::Warp) return hidden >= 1;
  return t == ln::DType::F16 ? warp_vecs<__half>(hidden) > 0 : warp_vecs<float>(hidden) > 0;
}

std::unique_ptr<Workload> make_add_layernorm(LnVariant v, ln::DType t, int rows, int hidden) {
  if (t == ln::DType::F16) return std::make_unique<AddLayerNorm<__half>>(v, t, rows, hidden);
  return std::make_unique<AddLayerNorm<float>>(v, t, rows, hidden);
}

}  // namespace trtbench::cuda
