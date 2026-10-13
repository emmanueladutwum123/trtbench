#!/usr/bin/env bash
# Phase 2 on an NVIDIA GPU: export, convert and quantize three models, build
# TensorRT engines, time TensorRT (with and without CUDA graphs), PyTorch eager
# and ONNX Runtime, score accuracy, and write the tables.
#
#   ./scripts/run_phase2.sh                  # LOCK_MHZ=495 MODE=latency by default
#   LOCK_MHZ=0 ./scripts/run_phase2.sh       # do not lock clocks
#
# Resumable: every step skips outputs that already exist under $WORK, so an
# interrupted session can simply be re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

# The lock is a ceiling, not a floor: the 70 W power cap still pulls the clock
# down. In the first T4 run, locked at 1005 MHz, heavy FP32 workloads sagged to
# 570-765 MHz and 23 of 54 timings were rejected for clock drift. 495 MHz sits
# below the lowest clock the cap imposed on any workload in that run.
LOCK_MHZ=${LOCK_MHZ:-495}
MODE=${MODE:-latency}
WORK=${WORK:-work}
export PYTHONPATH=python

source scripts/gpu_env.sh
out="$run_dir/phase2-${tag}"
mkdir -p "$out"

echo "===== prepare (export, FP16, INT8, references) ====="
python3 -m trtbench.phase2 prepare "$WORK"
echo "===== engines ====="
python3 -m trtbench.phase2 engines "$WORK"

lock_clocks

echo "===== baselines (PyTorch eager, ONNX Runtime) ====="
python3 -m trtbench.phase2 baselines "$WORK" "$out" --mode "$MODE"

echo "===== TensorRT timing + audit of every backend ====="
args=()
for m in "$WORK"/*/b*/*.meta.json; do args+=(--engine "$m"); done
for s in "$out"/samples/*.json; do args+=(--ingest "$s"); done
status=0
./build-gpu/trtbench --mode "$MODE" $lockflag --cuda-graph "${args[@]}" \
  --json "$out/phase2_${MODE}.json" > "$out/phase2_${MODE}.log" || status=$?
tail -n 5 "$out/phase2_${MODE}.log"

echo "===== TensorRT accuracy ====="
for m in "$WORK"/*/b*/*.meta.json; do
  ./build-gpu/trtbench --engine-eval "$m" --eval-out "$WORK/eval_out.bin"
  python3 -m trtbench.phase2 score "$WORK" "$m" "$WORK/eval_out.bin" "$out/accuracy.json"
  rm -f "$WORK/eval_out.bin"
done

echo "===== tables ====="
python3 -m trtbench.report --phase2 "$out/phase2_${MODE}.json" "$out/accuracy.json" \
  | tee "$out/phase2_tables.md"
mkdir -p "$out/engines"
for m in "$WORK"/*/b*/*.meta.json; do   # work/<model>/b<B>/<prec>.meta.json
  rel=${m#"$WORK"/}; cp "$m" "$out/engines/$(echo "$rel" | tr '/' '_')"
done
echo
echo "results in $out (trtbench exit $status: 0 all trustworthy, 3 something was rejected)"
