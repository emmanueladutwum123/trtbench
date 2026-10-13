#!/usr/bin/env bash
# Phase 2 on an NVIDIA GPU: export, convert and quantize three models, build
# TensorRT engines, time TensorRT (with and without CUDA graphs), PyTorch eager
# and ONNX Runtime, score accuracy, and write the tables.
#
#   ./scripts/run_phase2.sh                  # LOCK_MHZ=1005 MODE=latency by default
#   LOCK_MHZ=0 ./scripts/run_phase2.sh       # do not lock clocks
#
# Resumable: every step skips outputs that already exist under $WORK, so an
# interrupted session can simply be re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_MHZ=${LOCK_MHZ:-1005}   # below the T4's observed 1170-1260 MHz under load
MODE=${MODE:-latency}
WORK=${WORK:-work}
export PYTHONPATH=python

echo "===== setup ====="
python3 -m pip install -q -r requirements-phase2.txt
eval "$(scripts/setup_tensorrt.sh)"
echo "TensorRT $TRT_VERSION (CUDA $TRT_CUDA_MAJOR): $TENSORRT_LIBRARY"
python3 - <<'PY'
import importlib
for m in ("torch", "torchvision", "transformers", "onnx", "onnxruntime", "modelopt", "tensorrt"):
    try:
        print(f"  {m} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"  {m} UNAVAILABLE: {e}")
PY
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DTB_ENABLE_CUDA=ON -DTB_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DTENSORRT_INCLUDE_DIR="$TENSORRT_INCLUDE_DIR" -DTENSORRT_LIBRARY="$TENSORRT_LIBRARY"
cmake --build build-gpu -j

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
out="results/${gpu}-$(date -u +%Y%m%d)/phase2"
mkdir -p "$out"

echo "===== prepare (export, FP16, INT8, references) ====="
python3 -m trtbench.phase2 prepare "$WORK"
echo "===== engines ====="
python3 -m trtbench.phase2 engines "$WORK"

# One clock lock for every timing process, so PyTorch, ONNX Runtime and
# TensorRT are measured under the same clock policy.
lockflag=""
if [ "$LOCK_MHZ" != 0 ]; then
  if nvidia-smi -lgc "$LOCK_MHZ,$LOCK_MHZ"; then
    trap 'nvidia-smi -rgc >/dev/null' EXIT
    export TRTBENCH_EXTERNAL_LOCK_MHZ=$LOCK_MHZ
    lockflag="--clocks-locked-externally=$LOCK_MHZ"
  else
    echo "clock lock not permitted; timing unlocked (drift is still audited)"
  fi
fi

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
