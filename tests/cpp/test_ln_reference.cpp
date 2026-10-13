#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>

#include "trtbench/ln_reference.hpp"

using namespace trtbench::ln;

namespace {
bool is_nan_half(std::uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }
}  // namespace

TEST(Half, KnownValues) {
  EXPECT_EQ(float_to_half(1.0f), 0x3C00);
  EXPECT_EQ(float_to_half(-2.0f), 0xC000);
  EXPECT_EQ(float_to_half(65504.0f), 0x7BFF);
  EXPECT_EQ(float_to_half(65520.0f), 0x7C00);  // first value that rounds to inf
  EXPECT_EQ(float_to_half(0.1f), 0x2E66);
  EXPECT_EQ(float_to_half(5.9604644775390625e-8f), 0x0001);  // smallest subnormal
  EXPECT_EQ(float_to_half(2.98023223876953125e-8f), 0x0000);  // half of it: tie to even (0)
  EXPECT_EQ(float_to_half(1.0f + std::ldexp(1.0f, -11)), 0x3C00);      // tie -> even
  EXPECT_EQ(float_to_half(1.0f + 3 * std::ldexp(1.0f, -11)), 0x3C02);  // tie -> even (up)
}

TEST(Half, EveryHalfRoundTripsExactly) {
  for (std::uint32_t h = 0; h < 0x10000u; ++h) {
    const auto hh = static_cast<std::uint16_t>(h);
    if (is_nan_half(hh)) continue;
    ASSERT_EQ(float_to_half(half_to_float(hh)), hh) << std::hex << h;
  }
}

TEST(Half, EveryMidpointRoundsToEven) {
  // Positive finite halves, normal and subnormal: the float halfway between h
  // and h+1 is exact in float and must round to whichever of them is even.
  for (std::uint32_t h = 0; h < 0x7BFFu; ++h) {
    const float lo = half_to_float(static_cast<std::uint16_t>(h));
    const float hi = half_to_float(static_cast<std::uint16_t>(h + 1));
    const float mid = lo + (hi - lo) / 2;
    ASSERT_EQ(float_to_half(mid), (h & 1u) ? h + 1 : h) << std::hex << h;
    ASSERT_EQ(float_to_half(std::nextafter(mid, 0.0f)), h);
    ASSERT_EQ(float_to_half(std::nextafter(mid, 1e9f)), h + 1);
  }
}

#ifdef __FLT16_MAX__
TEST(Half, MatchesTheCompilersFloat16) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::uint32_t> bits;
  for (int i = 0; i < 2000000; ++i) {
    std::uint32_t x = bits(rng);
    float f;
    std::memcpy(&f, &x, 4);
    if (std::isnan(f)) continue;
    const _Float16 g = static_cast<_Float16>(f);
    std::uint16_t want;
    std::memcpy(&want, &g, 2);
    ASSERT_EQ(float_to_half(f), want) << f;
  }
}
#endif

TEST(Inputs, GeneratorIsPinned) {
  // The 24-bit hash outputs below were computed by an independent Python
  // implementation; tests/python/test_phase4.py asserts the same constants,
  // so C++ and Python generate identical inputs.
  const float v[4] = {input_value(0, 1, 0, 1), input_value(1, 1, 0, 1), input_value(2, 2, 0, 1),
                      input_value(12345, 3, 0, 1)};
  for (float x : v) {
    EXPECT_GE(x, 0.0f);
    EXPECT_LT(x, 1.0f);
  }
  EXPECT_EQ(static_cast<std::uint32_t>(v[0] * 16777216.0f), 0x514E28u);
  EXPECT_EQ(static_cast<std::uint32_t>(v[1] * 16777216.0f), 0xA64B3Du);
  EXPECT_EQ(static_cast<std::uint32_t>(v[2] * 16777216.0f), 0x45E53Au);
  EXPECT_EQ(static_cast<std::uint32_t>(v[3] * 16777216.0f), 0x8848F4u);
}

TEST(Inputs, ValuesAreRepresentableInTheStorageType) {
  const Inputs in = make_inputs(4, 64, DType::F16);
  for (float x : in.x) EXPECT_EQ(round_to(DType::F16, x), x);
  for (float g : in.gamma) {
    EXPECT_GE(g, 0.5f);
    EXPECT_LE(g, 1.5f);
  }
}

TEST(Reference, NormalizesEachRow) {
  Inputs in = make_inputs(3, 256, DType::F32);
  std::fill(in.gamma.begin(), in.gamma.end(), 1.0f);
  std::fill(in.beta.begin(), in.beta.end(), 0.0f);
  std::vector<float> s(256);
  std::vector<double> y(256);
  reference_row(in, 2, s.data(), y.data());
  double m = 0, v = 0;
  for (double a : y) m += a;
  m /= 256;
  for (double a : y) v += (a - m) * (a - m);
  v /= 256;
  EXPECT_NEAR(m, 0.0, 1e-12);
  EXPECT_NEAR(v, 1.0, 1e-4);  // eps = 1e-5 in the denominator
  EXPECT_EQ(s[5], in.x[2 * 256 + 5] + in.r[2 * 256 + 5]);
}

TEST(Reference, SumIsRoundedToStorageTypeBeforeNormalizing) {
  const Inputs in = make_inputs(1, 512, DType::F16);
  std::vector<float> s(512);
  std::vector<double> y(512);
  reference_row(in, 0, s.data(), y.data());
  for (int j = 0; j < 512; ++j) EXPECT_EQ(round_to(DType::F16, s[j]), s[j]);
}

TEST(Tolerance, HalfUlpOfOutputPlusArithmeticBound) {
  // y = 1, H = 768, fp16: ulp(1) = 2^-10, d = 2 * 768 * 2^-24 * 2.
  const double d = 2.0 * 768 * std::ldexp(1.0, -24) * 2.0;
  EXPECT_DOUBLE_EQ(tolerance(1.0, 768, DType::F16), std::ldexp(1.0, -11) + d);
  EXPECT_DOUBLE_EQ(tolerance(1.0, 768, DType::F32), std::ldexp(1.0, -24) + d);
  EXPECT_GT(tolerance(4.0, 768, DType::F16), tolerance(1.0, 768, DType::F16));
}

namespace {
void reference_outputs(const Inputs& in, std::vector<float>& s, std::vector<float>& y) {
  const std::size_t H = static_cast<std::size_t>(in.hidden);
  s.assign(in.x.size(), 0);
  y.assign(in.x.size(), 0);
  std::vector<double> yr(H);
  for (int r = 0; r < in.rows; ++r) {
    reference_row(in, r, s.data() + r * H, yr.data());
    for (std::size_t j = 0; j < H; ++j) y[r * H + j] = round_to(in.dtype, static_cast<float>(yr[j]));
  }
}
}  // namespace

TEST(Check, CorrectlyRoundedOutputPasses) {
  for (DType t : {DType::F32, DType::F16}) {
    const Inputs in = make_inputs(8, 768, t);
    std::vector<float> s, y;
    reference_outputs(in, s, y);
    const Check c = check(in, s, y);
    // A correctly rounded output is off by at most half an ulp, which is
    // below the tolerance (half an ulp + d) by the margin d provides.
    EXPECT_TRUE(c.passed) << c.note;
    EXPECT_LT(c.max_err_over_tol, 1.0) << to_string(t);
  }
}

TEST(Check, TwoUlpErrorInOneFp16OutputFails) {
  const Inputs in = make_inputs(8, 768, DType::F16);
  std::vector<float> s, y;
  reference_outputs(in, s, y);
  const std::size_t k = 5 * 768 + 100;
  std::uint16_t h = float_to_half(y[k]);
  h = static_cast<std::uint16_t>(h + 2);  // two ulps away (same sign, same binade almost always)
  y[k] = half_to_float(h);
  const Check c = check(in, s, y);
  EXPECT_FALSE(c.passed);
  EXPECT_EQ(c.worst_row, 5);
  EXPECT_EQ(c.worst_col, 100);
}

TEST(Check, SumMustBeBitExact) {
  const Inputs in = make_inputs(2, 256, DType::F32);
  std::vector<float> s, y;
  reference_outputs(in, s, y);
  s[3] = std::nextafter(s[3], 100.0f);
  const Check c = check(in, s, y);
  EXPECT_FALSE(c.passed);
  EXPECT_EQ(c.sum_mismatches, 1u);
}

TEST(Check, NonFiniteOutputFails) {
  const Inputs in = make_inputs(2, 256, DType::F32);
  std::vector<float> s, y;
  reference_outputs(in, s, y);
  y[7] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(check(in, s, y).passed);
}
