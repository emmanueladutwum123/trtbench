#include "trtbench/cuda/clock_lock.hpp"

#include <sys/wait.h>

#include <array>
#include <cstdio>

namespace trtbench::cuda {

std::pair<int, std::string> run_command(const std::string& cmd) {
  std::string out;
  FILE* p = popen((cmd + " 2>&1").c_str(), "r");
  if (p == nullptr) return {-1, "popen failed"};
  std::array<char, 256> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), p) != nullptr) out += buf.data();
  const int status = pclose(p);
  const int code = (status != -1 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return {code, out};
}

ClockLock::ClockLock(int nvml_index, unsigned mhz) : index_(nvml_index) {
  if (nvml_index < 0) {
    status_ = "not attempted: NVML unavailable, so the CUDA device cannot be mapped to an "
              "nvidia-smi index";
    return;
  }
  if (mhz == 0) {
    status_ = "not attempted: no target clock (NVML reported no default application clock); "
              "pass --lock-clocks=<MHz>";
    return;
  }
  const std::string m = std::to_string(mhz);
  const auto [code, out] =
      run_command("nvidia-smi -i " + std::to_string(nvml_index) + " -lgc " + m + "," + m);
  if (code == 0) {
    locked_ = true;
    status_ = "locked SM clock at " + m + " MHz via nvidia-smi -lgc";
  } else {
    status_ = "nvidia-smi -lgc failed (exit " + std::to_string(code) + "): " + out;
  }
}

ClockLock::~ClockLock() {
  if (locked_) run_command("nvidia-smi -i " + std::to_string(index_) + " -rgc");
}

}  // namespace trtbench::cuda
