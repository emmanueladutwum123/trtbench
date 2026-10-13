// Phase 4 kernels against the CPU reference on a real GPU; skips without one.
// Shapes include a row count that is not a multiple of the warp kernel's 4
// rows per block, and a hidden size only the naive and block kernels accept.

#include <gtest/gtest.h>

#include "trtbench/cuda/device_context.hpp"
#include "trtbench/cuda/layernorm.hpp"

using namespace trtbench;
using namespace trtbench::cuda;

namespace {

bool have_gpu(std::string* why) { return cuda_device_count(why) > 0; }

void expect_correct(LnVariant v, ln::DType t, int rows, int hidden) {
  cudaStream_t s = nullptr;
  ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
  auto w = make_add_layernorm(v, t, rows, hidden);
  w->setup(s);
  w->enqueue(s);
  w->enqueue(s);  // idempotent: a second launch must produce the same outputs
  const Checksum c = w->consume_output(s);
  cudaStreamDestroy(s);
  EXPECT_TRUE(c.passed) << w->name() << ": " << c.note;
}

}  // namespace

TEST(LayerNormGpu, EveryKernelMatchesTheReference) {
  std::string why;
  if (!have_gpu(&why)) GTEST_SKIP() << "no CUDA device: " << why;
  for (ln::DType t : {ln::DType::F32, ln::DType::F16})
    for (LnVariant v : {LnVariant::Naive, LnVariant::Block, LnVariant::Warp})
      for (int rows : {1, 7, 128, 1031}) expect_correct(v, t, rows, 768);
}

TEST(LayerNormGpu, OddHiddenSizeOnTheGeneralKernels) {
  std::string why;
  if (!have_gpu(&why)) GTEST_SKIP() << "no CUDA device: " << why;
  EXPECT_FALSE(ln_variant_supported(LnVariant::Warp, 1000, ln::DType::F16));
  for (ln::DType t : {ln::DType::F32, ln::DType::F16})
    for (LnVariant v : {LnVariant::Naive, LnVariant::Block}) expect_correct(v, t, 33, 1000);
}

TEST(LayerNormGpu, WarpKernelCoversEveryVectorCount) {
  std::string why;
  if (!have_gpu(&why)) GTEST_SKIP() << "no CUDA device: " << why;
  for (int vecs = 1; vecs <= 8; ++vecs) {
    expect_correct(LnVariant::Warp, ln::DType::F32, 9, 128 * vecs);
    expect_correct(LnVariant::Warp, ln::DType::F16, 9, 256 * vecs);
  }
}
