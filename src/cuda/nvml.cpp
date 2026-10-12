#include "trtbench/cuda/nvml.hpp"

#include <dlfcn.h>

#include <cstring>

namespace trtbench::cuda {

namespace {
// Minimal NVML ABI, declared here so the build does not depend on nvml.h.
using nvmlReturn_t = int;
struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st*;
constexpr nvmlReturn_t kSuccess = 0;
constexpr int kClockGraphics = 0;
constexpr int kClockSm = 1;
constexpr int kClockMem = 2;
constexpr int kTemperatureGpu = 0;

template <typename Fn>
bool load(void* lib, const char* sym, Fn* out) {
  void* p = dlsym(lib, sym);
  if (p == nullptr) return false;
  static_assert(sizeof(Fn) == sizeof(void*), "function pointer size");
  std::memcpy(out, &p, sizeof p);  // avoids the object->function pointer cast
  return true;
}
}  // namespace

struct NvmlClockSource::Api {
  void* lib = nullptr;
  nvmlDevice_t dev = nullptr;
  nvmlReturn_t (*init)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  const char* (*error_string)(nvmlReturn_t) = nullptr;
  nvmlReturn_t (*by_pci)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*get_index)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*clock)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*max_clock)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*default_app_clock)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*reasons)(nvmlDevice_t, unsigned long long*) = nullptr;
  nvmlReturn_t (*temperature)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*power)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*power_limit)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*persistence)(nvmlDevice_t, int*) = nullptr;
  nvmlReturn_t (*ecc)(nvmlDevice_t, int*, int*) = nullptr;
  nvmlReturn_t (*driver_version)(char*, unsigned) = nullptr;
  bool initialized = false;
};

std::unique_ptr<NvmlClockSource> NvmlClockSource::open(const std::string& pci, std::string* why) {
  auto fail = [&](std::string msg) -> std::unique_ptr<NvmlClockSource> {
    if (why) *why = std::move(msg);
    return nullptr;
  };

  auto api = std::make_unique<Api>();
  api->lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (api->lib == nullptr) return fail(std::string("dlopen libnvidia-ml.so.1 failed: ") + dlerror());

  void* lib = api->lib;
  const bool required = load(lib, "nvmlInit_v2", &api->init) &&
                        load(lib, "nvmlShutdown", &api->shutdown) &&
                        load(lib, "nvmlErrorString", &api->error_string) &&
                        load(lib, "nvmlDeviceGetHandleByPciBusId_v2", &api->by_pci) &&
                        load(lib, "nvmlDeviceGetClockInfo", &api->clock);
  if (!required) {
    dlclose(lib);
    return fail("libnvidia-ml.so.1 is missing a required symbol");
  }
  load(lib, "nvmlDeviceGetIndex", &api->get_index);
  load(lib, "nvmlDeviceGetMaxClockInfo", &api->max_clock);
  load(lib, "nvmlDeviceGetDefaultApplicationsClock", &api->default_app_clock);
  if (!load(lib, "nvmlDeviceGetCurrentClocksEventReasons", &api->reasons))
    load(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", &api->reasons);
  load(lib, "nvmlDeviceGetTemperature", &api->temperature);
  load(lib, "nvmlDeviceGetPowerUsage", &api->power);
  load(lib, "nvmlDeviceGetEnforcedPowerLimit", &api->power_limit);
  load(lib, "nvmlDeviceGetPersistenceMode", &api->persistence);
  load(lib, "nvmlDeviceGetEccMode", &api->ecc);
  load(lib, "nvmlSystemGetDriverVersion", &api->driver_version);

  if (nvmlReturn_t rc = api->init(); rc != kSuccess) {
    dlclose(lib);
    return fail(std::string("nvmlInit failed: ") + api->error_string(rc));
  }
  api->initialized = true;
  if (nvmlReturn_t rc = api->by_pci(pci.c_str(), &api->dev); rc != kSuccess) {
    const std::string msg = std::string("no NVML device at PCI ") + pci + ": " + api->error_string(rc);
    api->shutdown();
    dlclose(lib);
    return fail(msg);
  }

  std::unique_ptr<NvmlClockSource> src(new NvmlClockSource());
  NvmlDeviceInfo& info = src->info_;
  unsigned u = 0;
  int a = 0, b = 0;
  if (api->get_index && api->get_index(api->dev, &u) == kSuccess) info.index = static_cast<int>(u);
  if (api->max_clock && api->max_clock(api->dev, kClockSm, &u) == kSuccess)
    info.max_sm_clock_mhz = static_cast<int>(u);
  if (api->default_app_clock && api->default_app_clock(api->dev, kClockGraphics, &u) == kSuccess)
    info.default_app_sm_clock_mhz = static_cast<int>(u);
  if (api->persistence && api->persistence(api->dev, &a) == kSuccess) info.persistence_mode = a;
  if (api->ecc && api->ecc(api->dev, &a, &b) == kSuccess) info.ecc_enabled = a;
  if (api->power_limit && api->power_limit(api->dev, &u) == kSuccess) info.power_limit_w = u / 1e3;
  if (api->driver_version) {
    char buf[96] = {};
    if (api->driver_version(buf, sizeof buf) == kSuccess) info.driver_version = buf;
  }
  src->api_ = std::move(api);
  return src;
}

NvmlClockSource::~NvmlClockSource() {
  if (api_ && api_->initialized) api_->shutdown();
  if (api_ && api_->lib) dlclose(api_->lib);
}

ClockSnapshot NvmlClockSource::sample() {
  ClockSnapshot s;
  unsigned u = 0;
  if (api_->clock(api_->dev, kClockSm, &u) == kSuccess) s.sm_mhz = u;
  if (api_->clock(api_->dev, kClockMem, &u) == kSuccess) s.mem_mhz = u;
  unsigned long long r = 0;
  if (api_->reasons && api_->reasons(api_->dev, &r) == kSuccess) s.throttle_reasons = r;
  if (api_->temperature && api_->temperature(api_->dev, kTemperatureGpu, &u) == kSuccess)
    s.temp_c = u;
  if (api_->power && api_->power(api_->dev, &u) == kSuccess) s.power_w = u / 1e3;
  return s;
}

}  // namespace trtbench::cuda
