#include "trtbench/json.hpp"

namespace trtbench {

using nlohmann::json;

namespace {
json snapshot(const ClockSnapshot& s) {
  return {{"t_ms", s.t_ms},
          {"sm_mhz", s.sm_mhz},
          {"mem_mhz", s.mem_mhz},
          {"throttle_reasons", throttle::describe(s.throttle_reasons)},
          {"temp_c", s.temp_c},
          {"power_w", s.power_w}};
}

json config(const BenchConfig& c, int launches) {
  return {{"mode", to_string(c.mode)},
          {"warmup_iters", c.warmup_iters},
          {"warmup_min_ms", c.warmup_min_ms},
          {"min_iters", c.min_iters},
          {"min_time_ms", c.min_time_ms},
          {"max_iters", c.max_iters},
          {"launches_per_sample", launches},
          {"flush_l2", c.flush_l2},
          {"clock_sample_every", c.clock_sample_every},
          {"clock_drift_tol", c.clock_drift_tol}};
}
}  // namespace

json to_json(const Percentiles& p) {
  return {{"p50", p.p50}, {"p90", p.p90}, {"p99", p.p99},
          {"min", p.min}, {"max", p.max}, {"n", p.n}};
}

json to_json(const ClockReport& c) {
  json steady = json::array();
  for (const auto& s : c.steady) steady.push_back(snapshot(s));
  return {{"available", c.available},
          {"source", c.source},
          {"locked", c.locked},
          {"lock_status", c.lock_status},
          {"pre_warmup", snapshot(c.pre_warmup)},
          {"sm_mhz", {{"min", c.sm_min}, {"median", c.sm_median}, {"max", c.sm_max}}},
          {"mem_mhz", {{"min", c.mem_min}, {"max", c.mem_max}}},
          {"sm_drift", c.sm_drift},
          {"mem_drift", c.mem_drift},
          {"throttle_reasons_seen", throttle::describe(c.throttle_union)},
          {"steady", steady}};
}

json to_json(const DevicePeaks& p) {
  return {{"dram_gbps_theoretical", p.dram_gbps_theoretical},
          {"dram_gbps_measured", p.dram_gbps_measured},
          {"fp32_gflops", p.fp32_gflops},
          {"fp16_tc_gflops", p.fp16_tc_gflops},
          {"peak_sm_clock_mhz", p.peak_sm_clock_mhz},
          {"source", p.source}};
}

json to_json(const RooflinePoint& r) {
  return {{"time_ms", r.time_ms},
          {"bytes", r.bytes},
          {"flops", r.flops},
          {"achieved_gbps", r.achieved_gbps},
          {"achieved_gflops", r.achieved_gflops},
          {"arithmetic_intensity", r.intensity},
          {"ridge_point", r.ridge},
          {"peak_gbps_reference", r.peak_gbps},
          {"peak_gflops_at_run_clock", r.peak_gflops},
          {"pct_bw_peak", r.pct_bw_peak},
          {"pct_bw_theoretical", r.pct_bw_theoretical},
          {"pct_compute_peak", r.pct_compute_peak},
          {"bound", to_string(r.bound)}};
}

json to_json(const Environment& e) {
  return {{"gpu_name", e.gpu_name},
          {"pci_bus_id", e.pci_bus_id},
          {"compute_capability", std::to_string(e.cc_major) + "." + std::to_string(e.cc_minor)},
          {"sm_count", e.sm_count},
          {"max_threads_per_sm", e.max_threads_per_sm},
          {"l2_bytes", e.l2_bytes},
          {"mem_bus_width_bits", e.mem_bus_width_bits},
          {"mem_clock_khz", e.mem_clock_khz},
          {"max_sm_clock_mhz", e.max_sm_clock_mhz},
          {"default_app_sm_clock_mhz", e.default_app_sm_clock_mhz},
          {"driver_version", e.driver_version},
          {"cuda_driver_api_version", e.cuda_driver_api_version},
          {"cuda_runtime_version", e.cuda_runtime_version},
          {"persistence_mode", e.persistence_mode},
          {"ecc_enabled", e.ecc_enabled},
          {"power_limit_w", e.power_limit_w},
          {"cuda_module_loading", e.cuda_module_loading},
          {"nvml_status", e.nvml_status},
          {"hostname", e.hostname},
          {"timestamp_utc", e.timestamp_utc},
          {"trtbench_version", e.trtbench_version},
          {"git_sha", e.git_sha},
          {"compiler", e.compiler}};
}

json to_json(const Result& r, bool include_samples) {
  json findings = json::array();
  for (const auto& f : r.findings) {
    findings.push_back(
        {{"flag", to_string(f.flag)}, {"severity", to_string(f.severity)}, {"detail", f.detail}});
  }
  json j = {
      {"name", r.name},
      {"precision", to_string(r.precision)},
      {"config", config(r.config, r.launches_per_sample)},
      {"bytes_per_launch", r.bytes_per_launch},
      {"flops_per_launch", r.flops_per_launch},
      {"device_ms", to_json(r.device_ms)},
      {"host_enqueue_ms", to_json(r.host_enqueue_ms)},
      {"wall_ms", to_json(r.wall_ms)},
      {"first_call_device_ms", r.first_call_device_ms},
      {"first_call_host_ms", r.first_call_host_ms},
      {"launches_per_second_p50", r.device_ms.p50 > 0 ? 1e3 / r.device_ms.p50 : 0.0},
      {"launch_floor_ms", r.launch_floor_ms},
      {"clocks", to_json(r.clocks)},
      {"roofline", to_json(r.roofline)},
      {"checksum",
       {{"verifiable", r.checksum.verifiable},
        {"passed", r.checksum.passed},
        {"value", r.checksum.value},
        {"note", r.checksum.note}}},
      {"findings", findings},
      {"trustworthy", r.trustworthy()},
  };
  if (include_samples) j["device_samples_ms"] = r.device_samples_ms;
  return j;
}

json to_json(const RunReport& r, bool include_samples) {
  json cal = json::array();
  for (const auto& x : r.calibration) cal.push_back(to_json(x, include_samples));
  json res = json::array();
  for (const auto& x : r.results) res.push_back(to_json(x, include_samples));
  return {{"schema_version", r.schema_version},
          {"environment", to_json(r.env)},
          {"peaks", to_json(r.peaks)},
          {"launch_floor_latency_ms", r.launch_floor_latency_ms},
          {"launch_floor_throughput_ms", r.launch_floor_throughput_ms},
          {"calibration", cal},
          {"results", res},
          {"all_trustworthy", r.all_trustworthy()}};
}

json to_json(const Comparison& c) {
  json j = {{"status", to_string(c.status)},
            {"speedup_p50", c.speedup_p50},
            {"speedup_p99", c.speedup_p99},
            {"reason", c.reason}};
  if (c.accuracy) {
    j["accuracy"] = {{"metric", c.accuracy->metric},
                     {"baseline", c.accuracy->baseline},
                     {"candidate", c.accuracy->candidate},
                     {"delta", c.accuracy->delta()}};
  } else {
    j["accuracy"] = nullptr;
  }
  return j;
}

}  // namespace trtbench
