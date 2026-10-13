#!/usr/bin/env bash
# Phase 1 on a real NVIDIA GPU: build, run GPU tests, then run the reference
# workloads in both modes and write JSON under results/<gpu>-<date>/.
#
#   ./scripts/run_phase1.sh            # unlocked clocks (drift-checked)
#   sudo -E ./scripts/run_phase1.sh --lock-clocks=1100
#
# Extra arguments are passed to every trtbench invocation.
set -euo pipefail

cd "$(dirname "$0")/.."
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DTB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gpu -j

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
out="results/${gpu}-$(date -u +%Y%m%d)"
mkdir -p "$out"
nvidia-smi -q > "$out/nvidia-smi-q.txt"

# Every step runs even if an earlier one fails, so one session yields as much
# evidence as possible. Each step's exit status is reported at the end.
declare -A rc
step() {
  local name=$1; shift
  echo; echo "===== $name ====="
  set +e; "$@"; rc[$name]=$?; set -e
}

step unit_tests  ctest --test-dir build-gpu --output-on-failure
step self_test   ./build-gpu/trtbench --self-test --json "$out/self_test.json"
step latency     ./build-gpu/trtbench --mode latency "$@" --json "$out/phase1_latency.json"
step throughput  ./build-gpu/trtbench --mode throughput "$@" --json "$out/phase1_throughput.json"

echo
echo "===== summary (results in $out) ====="
status=0
for name in unit_tests self_test latency throughput; do
  echo "  $name: exit ${rc[$name]}"
  [ "${rc[$name]}" -eq 0 ] || status=1
done
echo "  (trtbench: 0 ok, 1 error, 3 a result failed its audit, 4 a trap failed)"
exit "$status"
