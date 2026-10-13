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
| 1 | Measurement core: event timing, warmup/steady state, clocks, roofline, JSON | **Done.** Run on a Tesla T4 (results below) |
| 5 | Trap suite: deliberately wrong benchmarks the harness must reject (`--self-test`) | **Done.** 12/12 traps pass on a Tesla T4; synthetic layer and SASS check run in CI |
| 2 | TensorRT FP32/FP16/INT8 vs PyTorch eager vs ONNX Runtime; CUDA Graphs | **First T4 run rejected by the harness itself (31 of 54 timings: clock drift, broken INT8). Both causes fixed; rerun pending.** [Details](#first-t4-run-rejected) |
| 3 | Per-layer attribution (IProfiler), fusion diff vs ONNX, roofline classification | **Built, not yet run.** Analysis unit-tested on CPU; GPU profiling waits for Colab GPU quota. [Method](#phase-3-where-the-time-goes) |
| 4 | Fused LayerNorm + residual kernel, naive vs optimized, Nsight Compute | Planned |

## Results

All numbers below come from one session on a **Google Colab Tesla T4**. They
are generated from the JSON in
[`results/Tesla_T4-20261013/`](results/Tesla_T4-20261013) by
`python -m trtbench.report`, and none were typed by hand. `nvidia-smi -q`
from the same session is committed alongside them.

> Tesla T4 (sm_75, 40 SMs), driver 580.82.07, CUDA runtime 13.0, ECC on,
> persistence off, power limit 70 W, **clocks unlocked**, trtbench
> `9b41bd386797`, 2026-10-13. One run per mode; sample counts are per row.

### Phase 1 reference workloads

Latency mode (one launch per sample, synchronized):

| Workload | Samples | Device p50 (µs) | Device p99 (µs) | Host enqueue p50 (µs) | Achieved | SM clock (MHz) | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| empty_kernel | 1325 | 4.10 | 19.62 | 2.74 | – | 1590–1590 | accepted (output_unverified, high_variance) |
| copy_256MiB | 377 | 2,334.11 | 2,351.97 | 3.76 | 230.0 GB/s (71.9% of theoretical) | 1365–1395 | accepted (power_cap_throttle) |
| fma_fp32_x16384 | 475 | – | – | – | – | 1170–1260 | **rejected**: clock_drift |

Throughput mode (100 launches per sample for the empty kernel, 10 otherwise):

| Workload | Samples | Device p50 (µs) | Device p99 (µs) | Host enqueue p50 (µs) | Achieved | SM clock (MHz) | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| empty_kernel | 1000 | 2.58 | 3.94 | 2.56 | – | 1590–1590 | accepted (output_unverified) |
| copy_256MiB | 200 | 2,322.23 | 2,324.85 | 2.88 | 231.2 GB/s (72.2% of theoretical) | 1320–1365 | accepted (power_cap_throttle) |
| fma_fp32_x16384 | 200 | 1,832.71 | 1,842.69 | 2.95 | 5,859 GFLOP/s (95.4% of peak at run clock) | 1185–1200 | accepted (power_cap_throttle) |

### What the run showed

- **The harness rejected one of its own measurements.** The latency-mode FMA
  run saw the SM clock wander between 1170 and 1260 MHz (7.1%) under the
  70 W power cap, above the 5% tolerance. The table therefore gives no number
  for that row. The throughput-mode run of the same kernel stayed within 1.3%
  and was accepted.
- **On this card the power cap sets the clock, not the clock policy.** The SM
  clock was 1590 MHz for the empty kernel, 1320–1395 MHz for the copy, and
  1170–1260 MHz for the FMA. Measured against the SM clock observed during
  the run, the FMA kernel reaches 95.4% of FP32 peak. Against the 1590 MHz
  datasheet peak the same run is only 72.0% (5,859 of 8,141 GFLOP/s). This is
  why % of peak is scaled to the observed clock: a single fixed peak would
  misstate either the kernel or the card.
- **Correctness held on hardware.** The copy output matched the input exactly
  at 4,097 sampled positions. The FMA output was bit-exact against the
  host's software FMA replica at 128 sampled threads.
- **The first call was faster than steady state.** In latency mode the FMA's
  first call took 1,517 µs against a steady-state p50 of 1,741 µs. It ran
  before the power cap pulled the clock down (1470 MHz before warmup), so
  warmup here exists to reach a steady power state, not just to absorb setup
  cost. Because the first call is always reported separately, this showed up
  instead of being averaged in.
- **Launch-scale timings moved between processes.** The empty-kernel latency
  floor was 4.06 µs in one process and 5.70 µs in the next, minutes later,
  on the same VM. Throughput-mode floors moved from 2.50 to 3.46 µs. On this
  shared host, differences of tens of percent in microsecond-scale numbers
  between separate runs are noise.
- **Copy bandwidth.** The copy reached 72% of the 320 GB/s theoretical peak
  with ECC enabled. ECC was not turned off for comparison, so this run cannot
  say how much of the gap ECC accounts for. "% of measured peak" is about
  100% for the copy by construction: the copy *is* the reference.

### Trap suite on the T4

All 12 traps passed: 6 synthetic and 6 on the GPU. Full table:
`python -m trtbench.report results/Tesla_T4-20261013/self_test.json`.

- **T1**: the dead kernel ran in 5.7 µs (p50) against 3,482 µs for the live
  control. That implies 5.2 PFLOP/s on an 8.1 TFLOP/s part, and it was
  rejected as `impossible_flops`.
- **T2**: the timer with no synchronize reported 6.4 µs while the events
  measured 2,332 µs, and it was rejected as `wallclock_below_device_time`.
- **T3**: the first call (7.6 ms) was surfaced and kept out of the reported
  distribution (p50 0.152 ms).
- **T4r ran rather than skipping**: `nvidia-smi -lgc` was permitted on Colab.
  The clock really was locked to 795 MHz mid-run, and the harness rejected
  the run for `clock_drift`.
- **T5's refusal held, but its control was inconclusive.** The FP16 run
  itself failed its audit during the session. A refused comparison does not
  report which flag caused it, so the cause was not recorded. Refusals now
  name their flags, and T5 will be re-run with locked clocks.

### First T4 run: rejected

The first complete run (Tesla T4, 2026-10-13, trtbench `66c02603c051`, clocks
locked at 1005 MHz) produced 54 timings. The auditor rejected 31 of them:

| Rejected for | Timings |
|---|---:|
| `clock_drift` only | 19 |
| `checksum_failed` only | 8 |
| both | 4 |

Everything the run wrote is committed in
[`results/Tesla_T4-20261013/phase2-lock1005/`](results/Tesla_T4-20261013/phase2-lock1005),
including the generated tables. None of its speedups are quoted here,
because most comparisons in it lost their baseline to a rejection.

**Clock drift.** `nvidia-smi -lgc` sets a ceiling, not a floor. Under the
T4's 70 W cap, heavy FP32 workloads ran far below the 1005 MHz lock, and
moved while they ran: ONNX Runtime GPT-2 at batch 4 sat at 570–630 MHz,
and TensorRT ResNet-50 FP32 at batch 16 at 690–885 MHz. The lock is now
495 MHz, below the lowest clock the cap imposed on any workload in that run.
Results for each lock setting go to their own directory, so timings at
different clocks never mix.

**Broken INT8 for the transformers.** With ModelOpt's default INT8 recipe:

| Model | Batch | FP32 | INT8 (default recipe) |
|---|---:|---:|---:|
| BERT-base, cosine vs FP32 | 1 / 8 | 1.000000 | 0.662524 / 0.653169 |
| GPT-2, perplexity | 1 / 4 | 59.49 | 345,425 / 65,407 |
| ResNet-50, top-1 % | 1 / 16 | 86.29 | 85.48 / 86.09 |

The same quantized graphs fail the same way in ONNX Runtime on CPU, so the
fault is in the quantized graph, not in TensorRT or the harness.
`scripts/diagnose_int8.py` reproduces the diagnosis on CPU; its output is in
[`results/int8_diagnosis_cpu.txt`](results/int8_diagnosis_cpu.txt). For
BERT, cosine vs FP32 on 16 test windows, with unquantized ops in FP32:

| What is INT8 | Cosine |
|---|---:|
| ModelOpt default (every linear, the attention BMMs, residual Adds) | 0.6712 |
| only the FFN up-projections (inputs come from LayerNorm) | 0.8961 |
| the same, after SmoothQuant | 0.9979 |
| only layer 2's FFN down-projection, after SmoothQuant | 0.9679 |
| the recipe above | 0.9960 |

- **LayerNorm-fed linears.** In every layer, the largest channel of the
  attention LayerNorm output spans 13–37× the median channel's range, and
  in 9 of 12 layers the two largest are channels 308 and 381. With one scale
  per tensor, every other channel quantizes to a handful of levels. SmoothQuant
  divides each input channel by s and multiplies the matching weight row by
  s, which is exact in FP32 (cosine 1.0000) and moves the range into the
  weights. LayerNorm's output also feeds the residual connection, so the
  division is an explicit `Mul` rather than folded into LayerNorm's affine.
- **FFN down-projections.** Their inputs come from GELU, and smoothing does
  not rescue them: one such layer in INT8, after SmoothQuant, costs more
  than all twelve up-projections together. I have not established why, so
  they stay FP16.

On GPT-2 (8 windows), the same recipe gives perplexity 56.07 against FP32's
55.14. ModelOpt's default gives 4,000.

These are diagnosis numbers on a small CPU set. The published INT8 accuracy
will come from the rerun on the T4, through TensorRT, on the full evaluation
sets. Keeping more of each transformer in FP16 also costs INT8 speed; the
rerun measures how much is left.

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
- **Locking.** `--lock-clocks=MHz` runs `nvidia-smi -lgc` and resets with
  `-rgc` on exit. It usually needs root; it was permitted on Colab. When it
  is not permitted, the reason is recorded in the report and the run
  continues, with drift detection as the only defence.

  There is deliberately **no default target**. An earlier version used NVML's
  "default application clock", which on the T4 is 585 MHz, about a third of
  the clock the card actually runs at. The boost clock is no good either: it
  is not sustainable under a power cap. Choose a clock at or below the
  minimum an unlocked run observed under load (`clocks.sm_mhz.min` in its
  JSON).

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

## Phase 2: TensorRT against PyTorch and ONNX Runtime

`scripts/run_phase2.sh` runs the whole comparison and can resume after an
interruption. The first run on a T4 was rejected (below); this section
describes the method as it now stands.

**Models and shapes**, identical for every backend:

| Model | Batch sizes | Input | Accuracy metric (evaluation set) |
|---|---|---|---|
| ResNet-50 (torchvision IMAGENET1K_V2) | 1, 16 | 3×224×224 | top-1 over all 1000 classes, 992 Imagenette val images |
| BERT-base-uncased encoder | 1, 8 | 128 tokens | cosine similarity of hidden states with FP32 PyTorch, 256 WikiText-2 test windows |
| GPT-2 124M, prefill | 1, 4 | 128 tokens | perplexity, 32 WikiText-2 test windows (4,064 predicted tokens) |

**Backends**:
- PyTorch eager at FP32 and FP16;
- ONNX Runtime with the CUDA execution provider at FP32;
- TensorRT 11.4 at FP32, FP16 and INT8, each also replayed from a captured
  CUDA graph.

**How precision is set.** In TensorRT 11, networks are always strongly
typed, so precision comes from the ONNX graph itself; builder flags and the
implicit INT8 calibrator are gone.
- **FP32**: the exported graph. TF32 is disabled, which changes nothing on
  Turing but keeps "FP32" honest on newer GPUs.
- **FP16**: ModelOpt AutoCast, with FP32 network inputs and outputs.
- **INT8**: ModelOpt post-training quantization with explicit Q/DQ and
  entropy calibration; ops left out of INT8 run in FP16. The recipe depends
  on the model family (`python/trtbench/int8_recipe.py`):
  - ResNet-50: ModelOpt's default, Q/DQ on every Conv, Gemm and MatMul.
  - BERT and GPT-2: SmoothQuant (α = 0.5) on every linear layer with a
    constant weight, then INT8 only on the up-projections. Down-projections,
    activation × activation MatMuls (attention, GPT-2's tied LM head) and
    residual Adds stay FP16. The rules are structural, so one recipe serves
    both models. Why: see [the first run](#first-t4-run-rejected).
- **Output sanity check**: every timed engine's output must have cosine
  ≥ 0.99 with the FP32 PyTorch output on the benchmark inputs, and ≥ 0.95
  for INT8. It catches engines that compute the wrong thing; the accuracy
  column, not this check, judges quantization quality. The INT8 threshold
  was set *after* the first run, where correct ResNet-50 INT8 engines scored
  0.971–0.985 with top-1 within 0.8 points of FP32.

**Engine build.** TF32 off, a shared timing cache, and `kDETAILED`
profiling verbosity, so the engine inspector can name the ONNX nodes behind
every layer (Phase 3 profiles the very engines timed here, since two builds
of one graph may pick different kernels). TensorRT warns that `kDETAILED`
adds NVTX markers that slow `enqueueV3`, so every execution context switches
its markers back to the default level (`setNvtxVerbosity(kLAYER_NAMES_ONLY)`)
before timing.

**INT8 calibration set**: 512 samples per model, taken from the *training*
split, so they never overlap the evaluation set.
- ResNet-50: 512 Imagenette-320 train images.
- BERT: 512 WikiText-2 train windows of `[CLS]` + 126 tokens + `[SEP]`.
- GPT-2: 512 WikiText-2 train windows of 128 tokens.

**One audit for every backend.**
- TensorRT engines are timed by the C++ runner.
- PyTorch and ONNX Runtime are timed by `python/trtbench/pyrunner.py`, a
  line-for-line port of the same per-sample protocol. Their raw samples go
  through `trtbench --ingest`, so they are audited against the same
  calibration, peaks and launch floor as the engines, in the same process.
- One clock lock covers every timing process.
- Each process loads only one CUDA runtime. Engine building and execution
  never share a process with PyTorch.

**What "bytes" means here.** It is a lower bound on DRAM traffic per
inference: weights at the precision they are stored in, plus inputs and
outputs.
- Embedding tables count only the rows a Gather actually reads.
- INT8-quantized weights count as one byte each.

Achieved bandwidth from this figure is therefore also a lower bound, which is
the safe direction for the impossibility check. FLOPs count Conv, Gemm and
MatMul. Checked against known figures: ResNet-50 8.18 GFLOPs and 25.5M
parameters; BERT-base at 128 tokens 22.3 GFLOPs; GPT-2 at 128 tokens 32.2
GFLOPs and 123.8M parameters.

**Speedups require accuracy.** Every speedup in the tables is relative to
PyTorch eager FP32 at the same model and batch size. A speedup at a
different precision is printed only next to its measured accuracy change.
Without one, the table says "refused".

**Correctness check versus accuracy.** Every timed run checks its own output.
Outputs must be finite and have cosine similarity of at least 0.99 with the
FP32 PyTorch output on the same input. That catches broken bindings or
engines; it is not an accuracy measurement.

## Phase 3: where the time goes

A total latency is a fact; an attribution is an argument. `scripts/run_phase3.sh`
profiles every Phase 2 engine at the same clock lock and writes, per engine,
which layers the time goes to, which ONNX nodes TensorRT fused into each
layer, and whether a faster kernel for that layer could help at all. **No
GPU results yet**: the code and its CPU tests are done; the run waits for the
GPU.

**Per-layer times.** `trtbench --engine-profile` attaches an `IProfiler` with
`setEnqueueEmitsProfile(false)` and collects each inference's layer times
with `reportToProfiler()` after it completes, over 200 inferences after
warmup. Each layer gets its median over those runs. Every run must report
the same layers in the same order, or the profile is rejected as
inconsistent rather than averaged. The same process also times 200
inferences *without* the profiler: profiling inserts events between layers,
so the sum of layer times is larger than the real latency, and both numbers
are printed. Shares are shares of the summed layer time. Clocks are sampled
throughout; drift over 5% rejects the engine's attribution, as in the runner.

**Fusion diff.** The engine inspector (JSON, `kDETAILED`) lists each engine
layer with the ONNX nodes it came from (`[ONNX Layer: …]` metadata written by
TensorRT's ONNX parser). Against the ONNX graph, `python/trtbench/attribution.py`
counts:
- ONNX compute nodes: every node except Constant, Q/DQ, and nodes that only
  transform constants (weight dequantization, transposes of weights);
- how many of them landed in a multi-node (fused) layer, how many got a layer
  of their own, and how many appear in no layer (removed at build);
- the fusion rate (nodes in fused layers ÷ nodes in any layer) and the
  largest unfused op by time;
- layers TensorRT inserted itself (reformats), which map to no ONNX node.

A profiled name the inspector does not list is reported as "not in
inspector" with its time, never matched by guesswork. A layer with no
metadata is mapped by its name (split on " + ") only when the parts are ONNX
node names.

**Roofline per layer.** At the SM clock the profile ran at:
- FLOPs: the Conv/Gemm/MatMul FLOPs of the layer's ONNX nodes.
- Bytes: each input and output activation once, at the dimensions and data
  type the inspector reports, plus the weights the layer's nodes read (1 byte
  for weights behind QuantizeLinear, only the gathered rows for Gather). A
  lower bound on what the layer must move.
- Compute peak by the layer's input type: INT8 or FP16 tensor cores, or FP32
  CUDA cores; DRAM peak is the measured copy.
- Memory-bound if bytes ÷ DRAM peak ≥ FLOPs ÷ compute peak; marked "near
  ridge" within 2× of the ridge point, where the lower-bound inputs make the
  call uncertain.
- A layer that finishes faster than its bytes allow at the DRAM peak is
  "above DRAM roofline": its data came from L2, and the DRAM roofline makes
  no claim about it.
- Headroom = measured − max(bytes ÷ DRAM peak, FLOPs ÷ compute peak): the
  most a perfect kernel *for the same fused layer* could save. Summed over an
  engine, it bounds what better kernels alone could buy; it does not bound
  what better fusion could buy.

Tables are generated from the committed `attribution.json` by
`python -m trtbench.attribution render`, never typed.

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
WORK=/path/to/work ./scripts/run_phase2.sh    # Phase 2, locked at 495 MHz
WORK=/path/to/work ./scripts/run_phase3.sh    # Phase 3 on the same engines
sudo -E ./scripts/run_phase1.sh --lock-clocks=1100   # locked where permitted
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
src/core/               stats, clocks, roofline, audit, compare, JSON,
                        per-layer profile bookkeeping
src/cuda/               event-timed runner, NVML via dlopen, calibration kernels
src/trt/                TensorRT engine workload (timing, CUDA graph,
                        evaluation, IProfiler + inspector profiling)
tools/                  trtbench CLI (CPU-only builds: --self-test only)
scripts/                run_phase1/2/3.sh, gpu_env.sh, setup_tensorrt.sh,
                        check_dce_sass.py, diagnose_int8.py
python/trtbench/        reports and diff; Phase 2 models, data, ONNX stats,
                        INT8 recipes, engine builds, PyTorch/ORT runner;
                        Phase 3 attribution
tests/cpp, tests/python unit tests; GPU tests skip without a device
results/                committed raw JSON from real runs
```

## What this does not show

- **One GPU, one session, shared hardware.** All results come from a single
  Colab T4 VM, which is shared infrastructure with unknown co-tenancy, and
  from one run per mode. They say nothing about other architectures, and the
  arch table has peaks only for sm_70, 75, 80, 86, 87, 89 and 90.
- **The Phase 1 numbers were taken with clocks unlocked.** They are
  power-cap-governed, and each row states the clock range it ran at.
- **T5 on the GPU has not yet shown a reported FP16 speedup.** It has shown
  only the refusal.
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
- **T4 on the GPU uses scripted clock readings.** T4r changes the real
  clock; it needs root, which Colab allowed.
- **Phase 2 scope.**
  - GPT-2 is measured as a single 128-token prefill. Token-by-token decoding
    with a KV cache is a different workload and is not measured.
  - BERT has no task head, so its accuracy figure is fidelity to FP32, not
    task accuracy.
  - GPT-2 perplexity uses 4,064 tokens. That is enough to compare variants
    scored on the same tokens, but not a benchmark perplexity.
  - ONNX Runtime is measured at FP32 only.
  - Timings are taken at a 495 MHz lock, well under the T4's 1590 MHz
    maximum, so that the power cap cannot move the clock mid-run. They
    compare backends at one clock; they are not what a T4 does at boost.
  - The transformer INT8 recipe was chosen by diagnosing BERT, then applied
    unchanged to GPT-2. It is partial INT8 (down-projections, attention and
    residuals stay FP16), so it says nothing about what full INT8 could reach
    with quantization-aware training or a better PTQ method.
  - Host-side numbers for PyTorch and ONNX Runtime include the Python
    interpreter. They are reported separately from device time.
- **Phase 3 scope.**
  - Attribution is per TensorRT layer, not per CUDA kernel. A fused layer's
    time cannot be split among the ONNX nodes inside it.
  - If TensorRT compiles a transformer into large opaque subgraphs, the
    attribution is only as fine as what the profiler and inspector expose;
    the tables report how much time could not be mapped to ONNX nodes.
  - Layer bytes are a lower bound on what the layer moves, not measured DRAM
    traffic; elementwise FLOPs are not counted. Near the ridge point, the
    memory/compute call can flip; such layers are marked.
  - The profiler perturbs what it measures. Shares are of the profiled sum,
    which is reported next to the unprofiled latency.
- Not tested: multi-GPU, MIG, MPS, distributed inference, or concurrent
  workloads sharing the GPU.

## License

MIT
