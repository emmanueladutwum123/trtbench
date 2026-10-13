#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trtbench/stats.hpp"

namespace trtbench {

// Collects TensorRT IProfiler callbacks (one reportLayerTime per layer per
// inference) into per-layer distributions. Pure CPU, so the bookkeeping that
// turns callbacks into the Phase 3 numbers is tested without a GPU.
//
// Every run must report the same layers in the same order: an engine executes
// a fixed layer sequence, so a run that differs means a callback was lost or
// two runs were merged, and the profile is marked inconsistent rather than
// silently averaged. A name reported twice within one run is kept as two
// entries (suffix "#2"), never summed into one.
class LayerProfile {
 public:
  void begin_run();
  void add(const std::string& layer, double ms);
  void end_run();

  struct Layer {
    std::string name;
    Percentiles ms;  // over runs
    double mean_ms = 0;
  };
  std::vector<Layer> layers() const;

  std::size_t runs() const { return runs_.size(); }
  bool consistent() const { return inconsistency_.empty(); }
  const std::string& inconsistency() const { return inconsistency_; }
  // Sum of all layer times within each run.
  std::vector<double> run_totals_ms() const;

 private:
  std::vector<std::vector<std::pair<std::string, double>>> runs_;
  std::vector<std::pair<std::string, double>> current_;
  bool in_run_ = false;
  std::string inconsistency_;
};

nlohmann::json to_json(const LayerProfile& p);

}  // namespace trtbench
