#!/usr/bin/env bash
# Phase 1 on a real NVIDIA GPU: build, run GPU tests, then run the reference
# workloads in both modes and write JSON under results/<gpu>-<date>/.
#
#   ./scripts/run_phase1.sh            # unlocked clocks (drift-checked)
#   sudo -E ./scripts/run_phase1.sh --lock-clocks
#
# Extra arguments are passed to every trtbench invocation.
set -euo pipefail

cd "$(dirname "$0")/.."
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DTB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gpu -j

ctest --test-dir build-gpu --output-on-failure

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
out="results/${gpu}-$(date -u +%Y%m%d)"
mkdir -p "$out"
nvidia-smi -q > "$out/nvidia-smi-q.txt"

status=0
./build-gpu/trtbench --mode latency "$@" --json "$out/phase1_latency.json" || status=$?
./build-gpu/trtbench --mode throughput "$@" --json "$out/phase1_throughput.json" || status=$?

echo
echo "results in $out (exit status $status: 0 = all trustworthy, 3 = something was rejected)"
exit "$status"
