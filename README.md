# trtbench

[![CI](https://github.com/emmanueladutwum123/trtbench/actions/workflows/ci.yml/badge.svg)](https://github.com/emmanueladutwum123/trtbench/actions/workflows/ci.yml)

A GPU inference performance analysis harness for TensorRT, built so that its
numbers can be defended: it measures with CUDA events, reports distributions
rather than means, watches the clocks, places every op on a roofline, and
**rejects measurements that are physically impossible or internally
inconsistent** instead of printing them.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Measurement core: event timing, warmup/steady state, clocks, roofline, JSON | **Code complete, CPU-tested. Not yet run on a GPU.** |
| 5 | Trap suite: deliberately wrong benchmarks the harness must reject (`--self-test`) | **Synthetic layer passing in CI; SASS check passing in CI. GPU layer not yet run.** |
| 2 | TensorRT FP32/FP16/INT8 vs PyTorch eager vs ONNX Runtime; CUDA Graphs | Planned |
| 3 | Per-layer attribution (IProfiler), fusion diff vs ONNX, roofline classification | Planned |
| 4 | Fused LayerNorm + residual kernel, naive vs optimized, Nsight Compute | Planned |

## Results

**There are no GPU results yet.** This repository was developed on a machine
without an NVIDIA GPU. The CUDA code is compiled by `nvcc` in CI, and the
logic that decides whether a measurement is believable is unit-tested on CPU,
but nothing has been timed on a GPU. This table will only be filled from JSON
in `results/`, produced by `scripts/run_phase1.sh` on hardware stated next to
the numbers.

## Methodology

### What one sample is

```
t0 = now
record(start)
h0 = now; enqueue x N; h1 = now        host_enqueue_ms = (h1 - h0) / N
record(stop)
sync(stop); t2 = now                   wall_ms         = (t2 - t0) / N
device_ms = elapsed(start, stop) / N
```

- **Device time** comes from CUDA events. **Host time** is `std::chrono`
  around the enqueue calls only. They are reported as two separate
  distributions and never summed.
- **Wall time** is a cross-check. `t0` is taken before `record(start)` and
  `t2` after `sync(stop)`, so the device interval must lie inside it. A sample
  where `wall_ms < device_ms` means the host timer stopped before the work
  finished, and the result is rejected.
- **Latency mode** uses N = 1 and synchronizes after every sample, so each
  sample starts with an empty queue. Event time then includes any time the
  GPU spent waiting for the host to submit the kernel. When host enqueue time
  reaches 80% of device time the result is flagged `launch_bound` rather than
  presented as kernel time.
- **Throughput mode** issues N launches (default 100) back to back between one
  pair of events, so per-launch host overhead overlaps with device work.
- Results from different modes are never compared.

### Phases of a run

1. **Setup**, then synchronize. Context creation is forced before any timing
   with `cudaFree(nullptr)`.
2. **First call**: always one launch, recorded on its own
   (`first_call_device_ms`) and excluded from the percentiles. If it is more
   than 2x the steady-state p50 it is flagged `first_call_outlier`, so it is
   surfaced rather than averaged away.
3. **Warmup**: continues until **both** 50 iterations and 250 ms have passed.
   An iteration count alone is not enough: 50 iterations of a 5 µs kernel is
   250 µs, too short for the GPU to leave its idle clocks.
4. **Steady state**: continues until **both** 200 samples and 1 s have
   passed, capped at 100 000 samples. p50, p90 and p99 use the nearest-rank
   method, so every reported percentile is a latency that was actually
   observed. Fewer than 100 samples is flagged, because p99 of a handful of
   points is just the maximum.
5. **Consume output**: the workload reads its result back and checks it
   against a reference. This is outside the timed region.

### Synchronization points

Every synchronize in `src/cuda/runner.cpp` is marked `DELIBERATE` and has a
comment explaining why it is there:

- after setup;
- after the optional L2 flush, so the flush is not inside the timed window;
- `cudaEventSynchronize(stop)` once per sample;
- after consuming the output.

`enqueue()` must never synchronize.

### Clocks

- **Observation.** SM clock, memory clock, temperature, power and NVML's
  clock-event ("throttle") reason bits are sampled before warmup, after
  warmup, every 25 steady-state samples and at the end. Sampling happens
  between samples, never inside a timed window. NVML is loaded with `dlopen`
  and the GPU is found by PCI bus ID, because CUDA and NVML can number devices
  differently.
- **Drift** is `(max - min) / max` over the steady-state snapshots, for both
  SM and memory clocks. Above 5% the result is rejected. This is stricter
  than comparing only before and after, which misses a dip in the middle of a
  run. The snapshot taken before warmup is recorded but excluded from drift:
  rising from idle clocks to boost is not throttling.
- **Throttle reasons.**
  - Thermal or hardware slowdown at any point rejects the run.
  - A software power cap is a warning, not a rejection. Steady power-capped
    clocks are normal for a 72 W L4 under load, and the drift check catches
    them if they move.
- **Locking.** `--lock-clocks[=MHz]` runs `nvidia-smi -lgc` with the NVML
  default application clock as the default target, and resets with `-rgc` on
  exit. This usually needs root. When it is not permitted the reason is
  recorded in the report and the run continues, with drift detection as the
  only defence.

### Peaks and the roofline

- **DRAM bandwidth** is reported two ways:
  - **theoretical**: 2 x memory clock x bus width, from the CUDA device
    attributes;
  - **measured**: the p50 of a device-to-device `float4` copy of
    max(256 MiB, 16 x L2), run during calibration and itself audited.

  "% of peak" uses the measured value; "% of theoretical" is reported
  alongside it.
- **Compute peak**: CUDA has no API for peak FLOP/s, so it comes from a
  per-architecture table (FP32 lanes per SM and FP16 tensor-core FLOP/clk per
  SM). Tests check the table against NVIDIA datasheet TFLOPS for the L4, A10
  and A100 to within 1%. % of compute peak is scaled to the **SM clock
  observed during the run**, because a GPU running at 1500 MHz cannot reach a
  2040 MHz datasheet peak.
- **Placement**: each op declares its minimum DRAM bytes and its FLOPs.
  Arithmetic intensity = FLOPs / bytes, and the op is classified as memory- or
  compute-bound against the ridge point (peak FLOP/s divided by peak
  bandwidth).

### The auditor

`src/core/audit.cpp` is pure C++ with no CUDA, so every rule is unit-tested on
a CPU-only CI runner. It produces findings of three severities. Any `invalid`
finding makes the result untrustworthy: the CLI then exits with status 3, and
`compare()` refuses to compute a speedup from it.

| Flag | Severity | Rule |
|---|---|---|
| `impossible_bandwidth` | invalid | fastest sample implies DRAM traffic above 1.05x theoretical peak (work eliminated, or data L2-resident) |
| `impossible_flops` | invalid | fastest sample implies FLOP/s above 1.05x theoretical peak |
| `wallclock_below_device_time` | invalid | the enclosing host interval is shorter than the device interval |
| `clock_drift` | invalid | SM or memory clock range above 5% in steady state |
| `thermal_or_hw_throttle` | invalid | NVML thermal, hardware-slowdown or power-brake reason seen |
| `checksum_failed` | invalid | output does not match the reference |
| `no_samples` | invalid | — |
| `power_cap_throttle` | warning | clocks were set by the power limit |
| `clocks_unavailable` | warning | no NVML, so throttling cannot be ruled out |
| `at_launch_floor` | warning | p50 within 10% of an empty kernel: this measures launch overhead |
| `launch_bound` | warning | latency mode, host enqueue ≥ 80% of device time |
| `too_few_samples` / `high_variance` / `no_warmup` / `output_unverified` | warning | — |
| `first_call_outlier` | info | first call more than 2x the steady-state p50 |

The 5% tolerance in the two impossibility checks covers rounding of the
nominal clocks; it is not there to make results pass. Physical impossibility
is judged against the peak at **maximum** clock, which nothing can beat.

When a comparison involves two precisions (for example FP32 against FP16), it
is refused unless an accuracy delta is supplied with it. A faster
lower-precision engine is only reported next to what it cost in accuracy.

## The trap suite (`--self-test`)

A harness that never sees a wrong measurement has not shown it can catch
one. `trtbench --self-test` runs benchmarks that are wrong on purpose and
checks that the harness rejects each one. Each trap also has a **control**: a
correct version of the same measurement, which must *not* be flagged for the
same reason. A harness that flags everything therefore fails its own
self-test.

| ID | Wrong benchmark | Harness must report | Control |
|---|---|---|---|
| T1 | FMA loop whose result is never stored, so the compiler deletes it, while the benchmark still claims the FLOPs | `impossible_flops` | same loop, result stored |
| T2 | `t0 = now; launch; t1 = now` with no synchronize, so it times the launch, not the work | `wallclock_below_device_time` | the harness's own synchronized measurement |
| T3 | no warmup, with a first call that runs the kernel 50 extra times (modelling autotuning) on top of a lazy module load | `first_call_outlier`; first call excluded from the percentiles; `no_warmup` | none (see below) |
| T4 | SM clock drops 30% partway through steady state | `clock_drift` | same run, steady clocks |
| T4r | the SM clock is really halved with `nvidia-smi -lgc` mid-run | `clock_drift` | (T4's) |
| T4t | clocks steady, but NVML reports a thermal slowdown | `thermal_or_hw_throttle` | no reason bits |
| T5 | FP16 vs FP32 `y = 1.1x` over 64M elements, comparing speed only | comparison refused | the same comparison with a measured accuracy delta must be reported |

**Two layers.**
- The **synthetic** layer (T1–T5, plus T4t) is pure CPU. It feeds the
  auditor the exact data each wrong benchmark produces on an L4-class GPU. It
  runs in every CI job and as a ctest test, so a regression in any audit rule
  fails the build.
- The **GPU** layer (T1–T5, plus T4r) really executes the wrong benchmarks on
  the device. It reports SKIP, with the reason, when there is no device. T4r
  needs permission to change clocks and skips without it. In the GPU T4, the
  timing is real but the clock readings are scripted; this proves the runner
  consults the clocks during steady state and that a drop reaches the
  verdict.

**T1 is checked against the compiled code, not just the timing.**
`scripts/check_dce_sass.py` reads the SASS for every architecture in the
binary. CI runs it on each push; the latest run shows:

| | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|---|---|---|---|---|---|
| live `fma_kernel` (control): FFMA count | 48 | 48 | 48 | 48 | 48 |
| `trap_dead_fma_kernel`: FFMA count | 0 | 0 | 0 | 0 | 0 |

The dead kernel compiles to a check on the loop count followed by `EXIT`;
the loop body is gone. The GPU layer then has to catch the empty kernel from
its timing alone.

**Why T3 has no control.** Whether a plain kernel's first call is an outlier
depends on the driver's lazy module loading (`CUDA_MODULE_LOADING`), so "the
control must not be flagged" would test the driver, not the harness. T3
instead asserts three things:
- the outlier is surfaced;
- the first call lies outside the reported distribution;
- the missing warmup is warned about.

Its output also states what a mean over every call would have reported.

When any trap fails, `--self-test` exits with status 4, and `--json PATH`
writes the table as JSON.

## Build

```bash
# CPU-only: core library + unit tests (any machine)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build

# With CUDA (needs the CUDA toolkit; running needs a GPU)
cmake -S . -B build -DTB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j

# Python report tools
pip install -e '.[test]' && pytest
```

Requirements: CMake ≥ 3.24, a C++17 compiler, and the CUDA toolkit 12.x for
the GPU build (CI uses 12.8). Google Test and nlohmann/json are fetched by
CMake.

## Running on a GPU

```bash
./build-gpu/trtbench --self-test --json results/<gpu>/self_test.json
./scripts/run_phase1.sh                       # clocks unlocked, drift-checked
sudo -E ./scripts/run_phase1.sh --lock-clocks # clocks locked where permitted
python -m trtbench.diff results/A/phase1_latency.json results/B/phase1_latency.json
```

`trtbench --help` lists every option. The diff tool refuses to compare across
different GPUs, drivers or runtimes unless `--allow-env-mismatch` is passed.
It refuses untrustworthy results. When a change is smaller than either run's
own p50-to-p90 spread, it labels the change as within run-to-run spread.

## Repository layout

```
include/trtbench/       core interfaces (pure C++17)
include/trtbench/cuda/  Workload, DeviceContext, runner, NVML, clock lock
src/core/               stats, clocks, roofline, audit, compare, JSON
src/cuda/               event-timed runner, NVML via dlopen, calibration kernels
tools/                  trtbench CLI (CPU-only builds: --self-test only)
scripts/                run_phase1.sh, check_dce_sass.py
python/trtbench/        report loading, validation, diff
tests/cpp, tests/python unit tests; GPU tests skip without a device
results/                committed raw JSON from real runs
```

## What this does not show

- **No GPU numbers yet.** Everything above describes what the harness does.
  It has not yet produced a measurement.
- Results will come from a **single GPU model** (planned: one rented L4 or
  A10G). They say nothing about other architectures, and the arch table has
  peaks only for sm_70, 75, 80, 86, 87, 89 and 90.
- **Compute peaks come from a table, not a measurement.** The FP16
  tensor-core figure is the datacenter rate; GeForce parts run FP16 with FP32
  accumulation at half that rate. INT8 has no peak until Phase 2.
- **Event timing has about 0.5 µs resolution.** Sub-microsecond differences
  between kernels in latency mode are not resolvable; the `at_launch_floor`
  flag exists for this reason.
- **Clock observation is sampled.** Clocks are read every 25 samples, so a
  throttling excursion shorter than that interval can be missed. The NVML
  reason bits are a current state, not a history.
- The copy-based bandwidth peak is what one simple kernel achieves, **not the
  hardware maximum**. A better copy kernel would raise the reference and
  lower every "% of peak".
- The CPU-side wall-clock cross-check catches a host timer that stops early.
  It **cannot detect** a device timer that is wrong in a consistent way.
- **The GPU layer of the trap suite has not run yet.** CI shows that it
  compiles and skips correctly, and that T1's kernel really is eliminated.
  Whether each GPU trap is caught end to end is still unverified.
- **The synthetic traps test the auditor's rules, not the GPU.** They prove
  the rules fire on the data shapes they were designed for. They do not prove
  a real GPU produces those shapes; the GPU layer exists to check that.
- **Dead code is caught by rate, so the size of the declared work matters.**
  T1 is caught because the declared work (58 SMs × 1536 threads × 32768
  iterations × 16 = 4.7e10 FLOPs per launch on an L4) cannot finish in the few microseconds an empty kernel takes. An
  eliminated kernel that declares only a little work is indistinguishable
  from a fast one. The defences in that case are weaker: `at_launch_floor`,
  and output checking where the workload provides it.
- **The no-sync trap relies on the work outlasting the launch.** When a
  kernel is shorter than its own launch overhead, a no-sync timer reports a
  number at least as long as the device time and is not caught. That number
  is then dominated by launch overhead, which `launch_bound` reports in
  latency mode.
- **T4 on the GPU uses scripted clock readings.** Only T4r changes the real
  clock, and it needs root.
- Not tested: multi-GPU, MIG, MPS, distributed inference, or concurrent
  workloads sharing the GPU.

## License

MIT
