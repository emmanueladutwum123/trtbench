#include "trtbench/stats.hpp"

#include <algorithm>
#include <cmath>

namespace trtbench {

double nearest_rank(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0;
  const auto n = static_cast<double>(sorted.size());
  auto rank = static_cast<std::size_t>(std::ceil(q * n));
  rank = std::clamp<std::size_t>(rank, 1, sorted.size());
  return sorted[rank - 1];
}

Percentiles summarize(std::vector<double> s) {
  Percentiles p;
  p.n = s.size();
  if (s.empty()) return p;
  std::sort(s.begin(), s.end());
  p.min = s.front();
  p.max = s.back();
  p.p50 = nearest_rank(s, 0.50);
  p.p90 = nearest_rank(s, 0.90);
  p.p99 = nearest_rank(s, 0.99);
  return p;
}

}  // namespace trtbench
