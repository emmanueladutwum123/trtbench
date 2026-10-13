#include "trtbench/ln_reference.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace trtbench::ln {

const char* to_string(DType t) { return t == DType::F16 ? "fp16" : "fp32"; }
std::size_t size_of(DType t) { return t == DType::F16 ? 2 : 4; }

std::uint16_t float_to_half(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  x &= 0x7FFFFFFFu;
  if (x >= 0x7F800000u)  // inf or nan
    return static_cast<std::uint16_t>(sign | 0x7C00u | (x > 0x7F800000u ? 0x200u : 0u));
  if (x >= 0x477FF000u)  // rounds to >= 65520: overflow to inf
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  if (x < 0x38800000u) {  // below the smallest normal half (2^-14): subnormal or zero
    if (x < 0x33000000u) return static_cast<std::uint16_t>(sign);  // < 2^-25 rounds to 0
    const std::uint32_t exp = x >> 23;
    const std::uint32_t mant = (x & 0x7FFFFFu) | 0x800000u;
    // value = mant * 2^(exp - 150) = h * 2^-24, so h = mant >> (126 - exp)
    const std::uint32_t shift = 126u - exp;
    std::uint32_t h = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = ((x - 0x38000000u) >> 13);  // rebias exponent 127 -> 15
  const std::uint32_t rem = x & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // carry may bump the exponent: correct
  return static_cast<std::uint16_t>(sign | h);
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  std::uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal: value = mant * 2^-24, exact in float
      const float v = static_cast<float>(mant) * 5.9604644775390625e-8f;
      std::memcpy(&x, &v, 4);
      x |= sign;
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000u | (mant << 13);
  } else {
    x = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

float round_to(DType t, float f) { return t == DType::F16 ? half_to_float(float_to_half(f)) : f; }

float input_value(std::uint32_t index, std::uint32_t salt, float lo, float hi) {
  // murmur3 finalizer of (index * golden + salt); all arithmetic mod 2^32.
  std::uint32_t h = index * 0x9E3779B1u + salt;
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  const float u = static_cast<float>(h >> 8) * (1.0f / 16777216.0f);  // [0, 1), exact
  return lo + (hi - lo) * u;
}

Inputs make_inputs(int rows, int hidden, DType t) {
  Inputs in;
  in.rows = rows;
  in.hidden = hidden;
  in.dtype = t;
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(hidden);
  in.x.resize(n);
  in.r.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto k = static_cast<std::uint32_t>(i);
    in.x[i] = round_to(t, input_value(k, 1, -2.0f, 2.0f));
    in.r[i] = round_to(t, input_value(k, 2, -2.0f, 2.0f));
  }
  in.gamma.resize(static_cast<std::size_t>(hidden));
  in.beta.resize(static_cast<std::size_t>(hidden));
  for (int j = 0; j < hidden; ++j) {
    const auto k = static_cast<std::uint32_t>(j);
    in.gamma[static_cast<std::size_t>(j)] = round_to(t, input_value(k, 3, 0.5f, 1.5f));
    in.beta[static_cast<std::size_t>(j)] = round_to(t, input_value(k, 4, -0.5f, 0.5f));
  }
  return in;
}

void reference_row(const Inputs& in, int row, float* s_out, double* y_out) {
  const std::size_t H = static_cast<std::size_t>(in.hidden);
  const std::size_t base = static_cast<std::size_t>(row) * H;
  double mean = 0;
  for (std::size_t j = 0; j < H; ++j) {
    const float sum = in.x[base + j] + in.r[base + j];  // one FP32 add, as on the GPU
    s_out[j] = round_to(in.dtype, sum);
    mean += s_out[j];
  }
  mean /= static_cast<double>(H);
  double var = 0;
  for (std::size_t j = 0; j < H; ++j) var += (s_out[j] - mean) * (s_out[j] - mean);
  var /= static_cast<double>(H);
  const double rstd = 1.0 / std::sqrt(var + static_cast<double>(kEps));
  for (std::size_t j = 0; j < H; ++j)
    y_out[j] = (s_out[j] - mean) * rstd * in.gamma[j] + in.beta[j];
}

namespace {
double ulp(double a, DType t) {
  const int mant = t == DType::F16 ? 10 : 23;
  const int min_exp = t == DType::F16 ? -14 : -126;
  int e = a > 0 ? static_cast<int>(std::floor(std::log2(a))) : min_exp;
  if (e < min_exp) e = min_exp;  // subnormal spacing is constant
  return std::ldexp(1.0, e - mant);
}
}  // namespace

double tolerance(double y_ref, int hidden, DType t) {
  const double u32 = std::ldexp(1.0, -24);
  const double a = std::fabs(y_ref);
  const double d = 2.0 * hidden * u32 * (1.0 + a);
  return ulp(a + d, t) / 2.0 + d;
}

Check check(const Inputs& in, const std::vector<float>& s, const std::vector<float>& y) {
  Check c;
  const std::size_t H = static_cast<std::size_t>(in.hidden);
  std::vector<float> s_ref(H);
  std::vector<double> y_ref(H);
  for (int row = 0; row < in.rows; ++row) {
    reference_row(in, row, s_ref.data(), y_ref.data());
    const std::size_t base = static_cast<std::size_t>(row) * H;
    for (std::size_t j = 0; j < H; ++j) {
      if (std::memcmp(&s[base + j], &s_ref[j], 4) != 0) ++c.sum_mismatches;
      const double err = std::fabs(static_cast<double>(y[base + j]) - y_ref[j]);
      const double ratio = std::isfinite(err) ? err / tolerance(y_ref[j], in.hidden, in.dtype)
                                              : std::numeric_limits<double>::infinity();
      if (ratio > c.max_err_over_tol || !std::isfinite(ratio)) {
        c.max_err_over_tol = ratio;
        c.worst_row = row;
        c.worst_col = static_cast<int>(j);
      }
      if (err > c.max_err || !std::isfinite(err)) c.max_err = err;
    }
  }
  c.passed = c.sum_mismatches == 0 && c.max_err_over_tol < 1.0;
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "s: %zu of %zu not bit-exact; y: max |err| %.3g, worst %.2f of tolerance "
                "(row %d, col %d), all %d rows checked",
                c.sum_mismatches, s.size(), c.max_err, c.max_err_over_tol, c.worst_row,
                c.worst_col, in.rows);
  c.note = buf;
  return c;
}

}  // namespace trtbench::ln
