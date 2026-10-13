#include "trtbench/trt/engine_workload.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "trtbench/cuda/check.hpp"
#include "trtbench/json.hpp"
#include "trtbench/layer_profile.hpp"
#include "trtbench/stats.hpp"

namespace trtbench::trt {

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    default: throw std::runtime_error("unsupported TensorRT I/O data type");
  }
}

std::string dir_of(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string resolve(const std::string& base_dir, const std::string& p) {
  return (!p.empty() && p[0] == '/') ? p : base_dir + "/" + p;
}

struct Tensor {
  std::string name;
  bool is_input = false;
  nvinfer1::DataType dtype{};
  std::size_t bytes = 0;
  void* device = nullptr;
  std::string data_file;  // input data, or FP32 reference output
};

// Forwards IProfiler callbacks into a LayerProfile. reportLayerTime is
// noexcept, so a failure is remembered and raised after the run.
class Collector final : public nvinfer1::IProfiler {
 public:
  explicit Collector(LayerProfile& p) : p_(p) {}
  void reportLayerTime(char const* layer, float ms) noexcept override {
    try {
      p_.add(layer, ms);
    } catch (...) {
      failed_ = true;
    }
  }
  bool failed() const { return failed_; }

 private:
  LayerProfile& p_;
  bool failed_ = false;
};

template <typename T>
struct TrtDeleter {
  void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

class EngineWorkload final : public cuda::Workload {
 public:
  EngineWorkload(const std::string& meta_path, bool graph) : graph_(graph) {
    std::ifstream f(meta_path);
    if (!f) throw std::runtime_error("cannot open engine meta " + meta_path);
    meta_ = nlohmann::json::parse(f);
    base_ = dir_of(meta_path);
    precision_ = precision_from_string(meta_.at("precision").get<std::string>());
    // Written by engines.py per precision; the check catches a wrong
    // computation, the accuracy column judges quantization quality.
    min_cosine_ = meta_.value("sanity_min_cosine", 0.99);
    name_ = "tensorrt/" + meta_.at("model").get<std::string>() + "_b" +
            std::to_string(meta_.at("batch").get<int>()) + "/" +
            meta_.at("precision_label").get<std::string>() + (graph_ ? "+graph" : "");
  }

  ~EngineWorkload() override {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    for (auto& t : tensors_) cudaFree(t.device);
  }

  std::string name() const override { return name_; }
  Precision precision() const override { return precision_; }
  std::map<std::string, std::string> labels() const override {
    return {{"backend", graph_ ? "tensorrt+graph" : "tensorrt"},
            {"model", meta_.at("model").get<std::string>()},
            {"batch", std::to_string(meta_.at("batch").get<int>())},
            {"precision", meta_.at("precision_label").get<std::string>()},
            {"trt_version", meta_.value("trt_version", std::string("?"))}};
  }
  std::uint64_t bytes_moved() const override { return meta_.at("bytes_lower_bound").get<std::uint64_t>(); }
  std::uint64_t flops() const override { return meta_.at("flops").get<std::uint64_t>(); }

  void setup(cudaStream_t s) override {
    if (!engine_) load(s);
    if (graph_ && !graph_exec_) capture(s);
  }

  void enqueue(cudaStream_t s) override {
    if (graph_exec_) {
      TB_CUDA_CHECK(cudaGraphLaunch(graph_exec_, s));
    } else if (!context_->enqueueV3(s)) {
      throw std::runtime_error(name_ + ": enqueueV3 failed");
    }
  }

  Checksum consume_output(cudaStream_t s) override {
    Checksum c{true, true, 0, ""};
    std::string notes;
    for (const auto& t : tensors_) {
      if (t.is_input) continue;
      if (t.data_file.empty()) {
        c.verifiable = false;
        c.passed = false;
        notes += t.name + ": no reference; ";
        continue;
      }
      const std::vector<float> got = to_float(t, s);
      const std::vector<char> ref_raw = read_file(resolve(base_, t.data_file));
      if (ref_raw.size() != got.size() * sizeof(float))
        throw std::runtime_error(t.name + ": reference size mismatch");
      std::vector<float> ref(got.size());
      std::memcpy(ref.data(), ref_raw.data(), ref_raw.size());

      double dot = 0, ng = 0, nr = 0, max_abs = 0;
      bool finite = true;
      for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) finite = false;
        dot += double(got[i]) * ref[i];
        ng += double(got[i]) * got[i];
        nr += double(ref[i]) * ref[i];
        max_abs = std::max(max_abs, std::abs(double(got[i]) - ref[i]));
      }
      const double cosine = (ng > 0 && nr > 0) ? dot / std::sqrt(ng * nr) : 0;
      c.value += cosine;
      char buf[200];
      std::snprintf(buf, sizeof buf, "%s: cosine %.6f vs FP32 reference (min %.2f), max |diff| %.3g%s; ",
                    t.name.c_str(), cosine, min_cosine_, max_abs, finite ? "" : ", NON-FINITE");
      notes += buf;
      if (!finite || cosine < min_cosine_) c.passed = false;
    }
    c.note = notes + "(sanity check, not an accuracy measurement)";
    return c;
  }

  std::size_t evaluate(const std::string& out_path, cudaStream_t s) {
    setup(s);
    const std::size_t n = meta_.at("eval_count").get<std::size_t>();
    const std::size_t batch = meta_.at("batch").get<std::size_t>();
    if (n % batch != 0) throw std::runtime_error("eval_count must be a multiple of the batch");
    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + out_path);

    std::vector<std::ifstream> files;
    for (const auto& t : tensors_) {
      if (!t.is_input) continue;
      const std::string f = meta_.at("eval_inputs").at(t.name).get<std::string>();
      files.emplace_back(resolve(base_, f), std::ios::binary);
      if (!files.back()) throw std::runtime_error("cannot open eval input " + f);
    }
    std::vector<char> host;
    for (std::size_t b = 0; b < n / batch; ++b) {
      std::size_t fi = 0;
      for (const auto& t : tensors_) {
        if (!t.is_input) continue;
        host.resize(t.bytes);
        if (!files[fi++].read(host.data(), static_cast<std::streamsize>(t.bytes)))
          throw std::runtime_error(t.name + ": eval input file too short");
        TB_CUDA_CHECK(cudaMemcpyAsync(t.device, host.data(), t.bytes, cudaMemcpyHostToDevice, s));
      }
      enqueue(s);
      TB_CUDA_CHECK(cudaStreamSynchronize(s));  // host buffer reuse + output read
      for (const auto& t : tensors_) {
        if (t.is_input) continue;
        const std::vector<float> o = to_float(t, s);
        out.write(reinterpret_cast<const char*>(o.data()),
                  static_cast<std::streamsize>(o.size() * sizeof(float)));
      }
    }
    if (!out) throw std::runtime_error("write failed: " + out_path);
    return n;
  }

  nlohmann::json profile(int warmup, int runs, ClockSource& clocks, cudaStream_t s) {
    setup(s);
    cudaEvent_t a = nullptr, b = nullptr;
    TB_CUDA_CHECK(cudaEventCreate(&a));
    TB_CUDA_CHECK(cudaEventCreate(&b));
    auto timed = [&]() {
      TB_CUDA_CHECK(cudaEventRecord(a, s));
      enqueue(s);
      TB_CUDA_CHECK(cudaEventRecord(b, s));
      TB_CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0;
      TB_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      return static_cast<double>(ms);
    };

    const ClockSnapshot pre = clocks.sample();
    for (int i = 0; i < warmup; ++i) timed();
    std::vector<ClockSnapshot> steady{clocks.sample()};
    const int every = std::max(1, runs / 20);

    std::vector<double> plain;
    for (int i = 0; i < runs; ++i) {
      plain.push_back(timed());
      if ((i + 1) % every == 0) steady.push_back(clocks.sample());
    }

    // enqueueEmitsProfile(false) keeps enqueueV3 asynchronous; the layer
    // times of each inference are collected after it completes.
    LayerProfile prof;
    Collector collector(prof);
    context_->setProfiler(&collector);
    context_->setEnqueueEmitsProfile(false);
    std::vector<double> profiled;
    std::string error;
    for (int i = 0; i < runs && error.empty(); ++i) {
      prof.begin_run();
      profiled.push_back(timed());
      if (!context_->reportToProfiler()) error = "reportToProfiler failed";
      prof.end_run();
      if (collector.failed()) error = "profiler callback failed";
      if ((i + 1) % every == 0) steady.push_back(clocks.sample());
    }
    context_->setProfiler(nullptr);
    steady.push_back(clocks.sample());
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (!error.empty()) throw std::runtime_error(name_ + ": " + error);

    TrtPtr<nvinfer1::IEngineInspector> insp(engine_->createEngineInspector());
    nlohmann::json info = nullptr;
    if (insp) {
      insp->setExecutionContext(context_.get());
      const char* text = insp->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
      if (text) {
        info = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
        if (info.is_discarded()) info = std::string(text);  // keep it, unparsed
      }
    }

    const Percentiles pp = summarize(plain), pf = summarize(profiled);
    return {{"name", name_},
            {"meta", meta_},
            {"warmup", warmup},
            {"layer_profile", to_json(prof)},
            {"end_to_end_ms",
             {{"without_profiler", to_json(pp)}, {"with_profiler", to_json(pf)}}},
            {"clocks", to_json(summarize_clocks(clocks.available(), clocks.name(), pre, steady))},
            {"engine_information", info}};
  }

 private:
  void load(cudaStream_t s) {
    const std::vector<char> blob = read_file(resolve(base_, meta_.at("engine").get<std::string>()));
    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error(name_ + ": deserializeCudaEngine failed");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error(name_ + ": createExecutionContext failed");
    // Engines are built with kDETAILED verbosity for Phase 3's inspector. Its
    // NVTX markers add enqueue latency, so every context times with the
    // default verbosity's markers (a no-op for engines built without it).
    context_->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY);

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      Tensor t;
      t.name = engine_->getIOTensorName(i);
      t.is_input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
      t.dtype = engine_->getTensorDataType(t.name.c_str());
      const nvinfer1::Dims d = engine_->getTensorShape(t.name.c_str());
      std::size_t n = 1;
      for (int k = 0; k < d.nbDims; ++k) {
        if (d.d[k] < 0) throw std::runtime_error(t.name + ": dynamic shape; engines must be static");
        n *= static_cast<std::size_t>(d.d[k]);
      }
      t.bytes = n * dtype_size(t.dtype);
      TB_CUDA_CHECK(cudaMalloc(&t.device, t.bytes));
      const char* key = t.is_input ? "inputs" : "reference_outputs";
      if (meta_.contains(key) && meta_[key].contains(t.name))
        t.data_file = meta_[key][t.name].get<std::string>();
      if (t.is_input) {
        if (t.data_file.empty()) throw std::runtime_error(t.name + ": no input data in meta");
        const std::vector<char> data = read_file(resolve(base_, t.data_file));
        if (data.size() != t.bytes)
          throw std::runtime_error(t.name + ": input file has " + std::to_string(data.size()) +
                                   " bytes, engine expects " + std::to_string(t.bytes));
        TB_CUDA_CHECK(cudaMemcpyAsync(t.device, data.data(), t.bytes, cudaMemcpyHostToDevice, s));
        TB_CUDA_CHECK(cudaStreamSynchronize(s));  // `data` dies at scope end
      }
      if (!context_->setTensorAddress(t.name.c_str(), t.device))
        throw std::runtime_error(t.name + ": setTensorAddress failed");
      tensors_.push_back(std::move(t));
    }
  }

  void capture(cudaStream_t s) {
    // TensorRT recommends one enqueue before capture so lazy initialization
    // happens outside the graph.
    if (!context_->enqueueV3(s)) throw std::runtime_error(name_ + ": pre-capture enqueue failed");
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    TB_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    const bool ok = context_->enqueueV3(s);
    TB_CUDA_CHECK(cudaStreamEndCapture(s, &g));
    if (!ok) {
      cudaGraphDestroy(g);
      throw std::runtime_error(name_ + ": enqueueV3 failed during graph capture");
    }
    TB_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, g, 0));
    TB_CUDA_CHECK(cudaGraphDestroy(g));
  }

  std::vector<float> to_float(const Tensor& t, cudaStream_t s) const {
    std::vector<char> raw(t.bytes);
    TB_CUDA_CHECK(cudaMemcpyAsync(raw.data(), t.device, t.bytes, cudaMemcpyDeviceToHost, s));
    TB_CUDA_CHECK(cudaStreamSynchronize(s));
    if (t.dtype != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error(t.name + ": only FP32 outputs are checked; engines keep FP32 I/O");
    std::vector<float> out(t.bytes / sizeof(float));
    std::memcpy(out.data(), raw.data(), t.bytes);
    return out;
  }

  bool graph_;
  nlohmann::json meta_;
  std::string base_;
  std::string name_;
  Precision precision_ = Precision::FP32;
  double min_cosine_ = 0.99;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
  std::vector<Tensor> tensors_;
  cudaGraphExec_t graph_exec_ = nullptr;
};

}  // namespace

std::unique_ptr<cuda::Workload> make_engine_workload(const std::string& meta_path, bool graph) {
  return std::make_unique<EngineWorkload>(meta_path, graph);
}

std::size_t evaluate_engine(const std::string& meta_path, const std::string& out_path) {
  cudaStream_t s = nullptr;
  TB_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  try {
    EngineWorkload w(meta_path, /*graph=*/false);
    const std::size_t n = w.evaluate(out_path, s);
    cudaStreamDestroy(s);
    return n;
  } catch (...) {
    cudaStreamDestroy(s);
    throw;
  }
}

nlohmann::json profile_engine(const std::string& meta_path, int warmup, int runs,
                              ClockSource& clocks) {
  if (runs < 1) throw std::runtime_error("profile_engine: runs must be >= 1");
  cudaStream_t s = nullptr;
  TB_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  try {
    EngineWorkload w(meta_path, /*graph=*/false);
    nlohmann::json j = w.profile(warmup, runs, clocks, s);
    cudaStreamDestroy(s);
    return j;
  } catch (...) {
    cudaStreamDestroy(s);
    throw;
  }
}

}  // namespace trtbench::trt
