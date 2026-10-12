#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace trtbench::cuda {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void throw_cuda(cudaError_t e, const char* expr, const char* file, int line) {
  throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr + " -> " +
                  cudaGetErrorName(e) + ": " + cudaGetErrorString(e));
}

}  // namespace trtbench::cuda

#define TB_CUDA_CHECK(expr)                                                  \
  do {                                                                       \
    const cudaError_t tb_err_ = (expr);                                      \
    if (tb_err_ != cudaSuccess)                                              \
      ::trtbench::cuda::throw_cuda(tb_err_, #expr, __FILE__, __LINE__);      \
  } while (0)
