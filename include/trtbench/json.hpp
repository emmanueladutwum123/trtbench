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

// Samples collected outside the C++ runner (the Python baselines) enter the
// same audit through this. Schema "trtbench-samples-1"; see
// python/trtbench/pyrunner.py for the writer. Peaks and the launch floor are
// not part of the file: the caller fills them from the live DeviceContext so
// every backend is judged against the same calibration. Throws
// std::invalid_argument on a malformed document.
FinalizeInput samples_from_json(const nlohmann::json& j);
Precision precision_from_string(const std::string& s);
Mode mode_from_string(const std::string& s);

}  // namespace trtbench
