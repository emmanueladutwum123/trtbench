#pragma once

#include <memory>

#include "trtbench/cuda/workload.hpp"
#include "trtbench/ln_reference.hpp"

namespace trtbench::cuda {

// Phase 4: fused residual add + LayerNorm (see ln_reference.hpp for the exact
// operation). Three kernels, each adding one idea to the one before:
//
//   naive  one thread per row. Lanes of a warp read addresses `hidden`
//          elements apart, so every load is uncoalesced, and a 128-row
//          batch launches 128 threads on a 40-SM GPU. Three passes over
//          the row (sum, variance, normalize), re-reading global memory.
//   block  one 256-thread block per row: coalesced scalar loads, block-wide
//          tree reductions in shared memory with __syncthreads. Still three
//          passes, the second and third served from cache.
//   warp   one warp per row, 4 rows per block: 16-byte vector loads
//          (float4 / 8 halves), the whole row held in registers so DRAM is
//          read once and written once, mean and variance by __shfl_xor_sync
//          butterflies, no shared memory and no block barrier. Needs
//          hidden to be a multiple of 32 lanes x one vector (128 for FP32,
//          256 for FP16) and at most 8 vectors per lane.
//
// All three compute in FP32, round s = x + r to the storage type before
// normalizing, and write both s and y.
enum class LnVariant { Naive, Block, Warp };
const char* to_string(LnVariant v);
bool ln_variant_supported(LnVariant v, int hidden, ln::DType t);

// Declares 4 x rows x hidden x sizeof(T) + 2 x hidden x sizeof(T) bytes
// (read x, r, gamma, beta once; write s, y once) and no FLOPs: the op is
// memory-bound by an order of magnitude, and elementwise FLOPs are not
// counted anywhere in this harness. Output checked on every row against
// ln::check (s bit-exact, y within the derived tolerance).
std::unique_ptr<Workload> make_add_layernorm(LnVariant v, ln::DType t, int rows, int hidden);

}  // namespace trtbench::cuda
