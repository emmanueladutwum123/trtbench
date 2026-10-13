#!/usr/bin/env bash
# Phase 3 on an NVIDIA GPU: per-layer attribution of every Phase 2 engine.
# For each engine: IProfiler layer times, end-to-end time with and without the
# profiler, clocks and the engine inspector's layer list; then the fusion diff
# against the ONNX graph and a roofline placement of every layer.
#
#   ./scripts/run_phase3.sh          # same WORK and LOCK_MHZ as run_phase2.sh
#
# Profiles the engines run_phase2.sh timed (it builds them if missing), so the
# attribution and the Phase 2 latencies describe the same engines.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_MHZ=${LOCK_MHZ:-495}
RUNS=${RUNS:-200}
WORK=${WORK:-work}
export PYTHONPATH=python

source scripts/gpu_env.sh
out="$run_dir/phase3-${tag}"
mkdir -p "$out/profiles"

echo "===== engines (no-op when run_phase2.sh built them) ====="
python3 -m trtbench.phase2 prepare "$WORK"
python3 -m trtbench.phase2 engines "$WORK"

lock_clocks

echo "===== per-layer profiles ====="
args=()
for m in "$WORK"/*/b*/*.meta.json; do args+=(--engine-profile "$m"); done
./build-gpu/trtbench $lockflag --profile-out "$out/profiles" --profile-runs "$RUNS" "${args[@]}" \
  | tee "$out/phase3_profile.log"

echo "===== attribution ====="
python3 -m trtbench.attribution analyze "$out"/profiles/*.profile.json --out "$out/attribution.json"
python3 -m trtbench.attribution render "$out/attribution.json" | tee "$out/phase3_tables.md"
echo
echo "results in $out"
