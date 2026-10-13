#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtbench::ln {

// Phase 4's operation, as in a pre-LN transformer block (GPT-2):
//   s = x + r                          (the new residual stream, output 1)
//   y = (s - mean(s)) / sqrt(var(s) + eps) * gamma + beta    (output 2)
// over rows of `hidden` elements. Storage type T is FP32 or FP16; arithmetic
// is FP32 on the GPU. s is rounded to T before it is normalized, exactly as
// an unfused framework would (add kernel writes s in T, LayerNorm reads it).
//
// Pure CPU: the reference, the tolerance and the input generator are tested
// without a GPU and shared by the CUDA kernels' checks.

enum class DType { F32, F16 };
const char* to_string(DType t);
std::size_t size_of(DType t);

// IEEE binary16 <-> binary32, round to nearest even (what __float2half_rn does).
std::uint16_t float_to_half(float f);
float half_to_float(std::uint16_t h);
// Round a float to the nearest value representable in t.
float round_to(DType t, float f);

// Deterministic inputs, identical in C++ and Python (python/trtbench/phase4.py):
// a 32-bit integer hash of (index, salt) mapped to [lo, hi), then rounded to t.
float input_value(std::uint32_t index, std::uint32_t salt, float lo, float hi);

struct Inputs {
  int rows = 0, hidden = 0;
  DType dtype = DType::F32;
  std::vector<float> x, r, gamma, beta;  // values already representable in dtype
};
Inputs make_inputs(int rows, int hidden, DType t);

constexpr float kEps = 1e-5f;

// One row: s_out in T (as float; the GPU must match it bit for bit, because
// it is one rounded add on both sides), y_out exact enough to judge against
// (double, two-pass mean/variance).
void reference_row(const Inputs& in, int row, float* s_out, double* y_out);

// Largest acceptable |y_gpu - y_ref| at y_ref, for hidden size H:
//   ulp_T(|y_ref| + d) / 2 + d,   d = 2 H u32 (1 + |y_ref|)
// d bounds the FP32 arithmetic error: recursive summation of H terms has
// relative error <= H u (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., sec. 4.2); mean and variance each contribute one such
// term, and it scales with |y - beta| <= |y| + 1 for |beta| <= 1. rsqrtf's
// 2-ulp error is far below it. The half ulp is the final rounding to T.
double tolerance(double y_ref, int hidden, DType t);

struct Check {
  bool passed = true;
  std::size_t sum_mismatches = 0;  // s must be bit-exact
  double max_err = 0;              // max |y - y_ref|
  double max_err_over_tol = 0;     // worst ratio; < 1 passes
  int worst_row = -1, worst_col = -1;
  std::string note;
};
// s and y are the GPU outputs converted to float (row-major rows x hidden).
Check check(const Inputs& in, const std::vector<float>& s, const std::vector<float>& y);

}  // namespace trtbench::ln
