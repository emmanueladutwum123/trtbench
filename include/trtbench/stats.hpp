#pragma once

#include <cstddef>
#include <vector>

namespace trtbench {

struct Percentiles {
  double p50 = 0, p90 = 0, p99 = 0, min = 0, max = 0;
  std::size_t n = 0;
};

// Nearest-rank percentile on an already sorted vector: the value at rank
// ceil(q * n). Every reported percentile is a latency that was actually
// observed; no interpolation invents a value between two samples.
// q must be in (0, 1]. Returns 0 for an empty vector.
double nearest_rank(const std::vector<double>& sorted, double q);

Percentiles summarize(std::vector<double> samples);

}  // namespace trtbench
