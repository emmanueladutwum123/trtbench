#include "trtbench/layer_profile.hpp"

#include <map>
#include <stdexcept>

namespace trtbench {

void LayerProfile::begin_run() {
  if (in_run_) throw std::logic_error("LayerProfile: begin_run inside a run");
  in_run_ = true;
  current_.clear();
}

void LayerProfile::add(const std::string& layer, double ms) {
  if (!in_run_) throw std::logic_error("LayerProfile: layer time outside a run");
  int seen = 0;
  for (const auto& e : current_)
    if (e.first == layer || e.first.rfind(layer + "#", 0) == 0) ++seen;
  current_.emplace_back(seen ? layer + "#" + std::to_string(seen + 1) : layer, ms);
}

void LayerProfile::end_run() {
  if (!in_run_) throw std::logic_error("LayerProfile: end_run outside a run");
  in_run_ = false;
  if (!runs_.empty() && inconsistency_.empty()) {
    const auto& first = runs_.front();
    if (current_.size() != first.size()) {
      inconsistency_ = "run " + std::to_string(runs_.size()) + " reported " +
                       std::to_string(current_.size()) + " layers, run 0 reported " +
                       std::to_string(first.size());
    } else {
      for (std::size_t i = 0; i < first.size(); ++i) {
        if (current_[i].first != first[i].first) {
          inconsistency_ = "run " + std::to_string(runs_.size()) + " layer " + std::to_string(i) +
                           " is '" + current_[i].first + "', run 0 had '" + first[i].first + "'";
          break;
        }
      }
    }
  }
  runs_.push_back(std::move(current_));
  current_.clear();
}

std::vector<LayerProfile::Layer> LayerProfile::layers() const {
  std::vector<Layer> out;
  if (runs_.empty()) return out;
  // Order of run 0; a consistent profile has the same order in every run.
  std::map<std::string, std::vector<double>> times;
  for (const auto& run : runs_)
    for (const auto& e : run) times[e.first].push_back(e.second);
  for (const auto& e : runs_.front()) {
    const std::vector<double>& t = times[e.first];
    Layer l;
    l.name = e.first;
    double sum = 0;
    for (double x : t) sum += x;
    l.mean_ms = sum / static_cast<double>(t.size());
    l.ms = summarize(t);
    out.push_back(std::move(l));
  }
  return out;
}

std::vector<double> LayerProfile::run_totals_ms() const {
  std::vector<double> out;
  for (const auto& run : runs_) {
    double s = 0;
    for (const auto& e : run) s += e.second;
    out.push_back(s);
  }
  return out;
}

nlohmann::json to_json(const LayerProfile& p) {
  nlohmann::json layers = nlohmann::json::array();
  for (const auto& l : p.layers()) {
    layers.push_back({{"name", l.name},
                      {"median_ms", l.ms.p50},
                      {"p90_ms", l.ms.p90},
                      {"min_ms", l.ms.min},
                      {"max_ms", l.ms.max},
                      {"mean_ms", l.mean_ms},
                      {"n", l.ms.n}});
  }
  const Percentiles tot = summarize(p.run_totals_ms());
  return {{"runs", p.runs()},
          {"consistent", p.consistent()},
          {"inconsistency", p.inconsistency()},
          {"sum_of_layers_ms", {{"p50", tot.p50}, {"p90", tot.p90}, {"min", tot.min}, {"max", tot.max}}},
          {"layers", layers}};
}

}  // namespace trtbench
