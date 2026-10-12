#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "trtbench/compare.hpp"
#include "trtbench/result.hpp"

namespace trtbench {

nlohmann::json to_json(const Percentiles& p);
nlohmann::json to_json(const ClockReport& c);
nlohmann::json to_json(const DevicePeaks& p);
nlohmann::json to_json(const RooflinePoint& r);
nlohmann::json to_json(const Environment& e);
nlohmann::json to_json(const Result& r, bool include_samples = true);
nlohmann::json to_json(const RunReport& r, bool include_samples = true);
nlohmann::json to_json(const Comparison& c);

}  // namespace trtbench
