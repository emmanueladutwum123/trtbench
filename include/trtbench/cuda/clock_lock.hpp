#pragma once

#include <string>
#include <utility>

namespace trtbench::cuda {

// Locks the SM clock with `nvidia-smi -lgc <mhz>,<mhz>` for the lifetime of
// the object and resets it with `nvidia-smi -rgc` on destruction.
//
// Never throws. Locking usually needs root and is often not permitted on
// cloud VMs or containers; in that case locked() is false, status() says why,
// and the run continues with drift detection as the only defence.
class ClockLock {
 public:
  ClockLock(int nvml_index, unsigned mhz);
  ~ClockLock();
  ClockLock(const ClockLock&) = delete;
  ClockLock& operator=(const ClockLock&) = delete;

  bool locked() const { return locked_; }
  const std::string& status() const { return status_; }
  unsigned mhz() const { return mhz_; }

 private:
  int index_;
  unsigned mhz_;
  bool locked_ = false;
  std::string status_;
};

// Runs a shell command, returns {exit status, combined stdout+stderr}.
std::pair<int, std::string> run_command(const std::string& cmd);

}  // namespace trtbench::cuda
