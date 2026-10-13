#!/usr/bin/env bash
# Phase 4 on an NVIDIA GPU: time the three fused add + LayerNorm kernels and
# the unfused PyTorch baseline in one audited report, then profile each
# kernel once with Nsight Compute where the environment permits it.
#
#   ./scripts/run_phase4.sh          # LOCK_MHZ=495, same clock as Phases 2-3
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_MHZ=${LOCK_MHZ:-495}
MODE=${MODE:-latency}
ROWS=${ROWS:-128,1024,8192,65536}
HIDDEN=${HIDDEN:-768}
export PYTHONPATH=python

source scripts/gpu_env.sh
out="$run_dir/phase4-${tag}"
mkdir -p "$out/samples"

echo "===== GPU correctness tests ====="
ctest --test-dir build-gpu -R LayerNormGpu --output-on-failure | tee "$out/gpu_tests.log"

lock_clocks

echo "===== PyTorch unfused baseline ====="
python3 -m trtbench.phase4 baselines "$out/samples" --rows "$ROWS" --hidden "$HIDDEN" --mode "$MODE"

echo "===== kernels + audit of every result ====="
args=()
for s in "$out"/samples/*.json; do args+=(--ingest "$s"); done
status=0
./build-gpu/trtbench --mode "$MODE" $lockflag --op ln --ln-rows "$ROWS" --ln-hidden "$HIDDEN" \
  "${args[@]}" --json "$out/phase4_${MODE}.json" > "$out/phase4_${MODE}.log" || status=$?
tail -n 5 "$out/phase4_${MODE}.log"

echo "===== Nsight Compute ====="
# One profiled launch per kernel and dtype at the largest shape, saved as a
# report and exported to CSV (so the harness's own stdout never mixes into
# it). --clock-control none keeps the run's clock lock: by default ncu locks
# clocks to base itself. Without counter permission (ERR_NVGPUCTRPERM) the
# failure is recorded in ncu_status.txt and the tables say so.
ncu_bin=$(command -v ncu || ls /usr/local/cuda/bin/ncu 2>/dev/null || true)
largest=${ROWS##*,}
ncu_csv=""
if [ -z "$ncu_bin" ]; then
  echo "ncu not found in this environment" | tee "$out/ncu_status.txt"
else
  "$ncu_bin" --version | tee "$out/ncu_status.txt"
  ncu_ok=1
  for k in naive block warp; do
    for d in fp32 fp16; do
      if ! "$ncu_bin" --clock-control none -k "regex:add_ln_${k}" --launch-skip 5 --launch-count 1 \
          --section SpeedOfLight --section Occupancy --section LaunchStats \
          --section MemoryWorkloadAnalysis --section WarpStateStats \
          --metrics "regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio" \
          -f -o "$out/ncu_${k}_${d}" \
          ./build-gpu/trtbench --op ln --ln-kernels "$k" --ln-dtypes "$d" --ln-rows "$largest" \
            --ln-hidden "$HIDDEN" --no-calibrate --warmup 5 --warmup-ms 0 --min-iters 10 \
            --min-time-ms 0 > "$out/ncu_${k}_${d}.log" 2>&1 \
         || ! "$ncu_bin" --import "$out/ncu_${k}_${d}.ncu-rep" --csv --page raw \
              > "$out/ncu_${k}_${d}.csv" 2>> "$out/ncu_${k}_${d}.log"; then
        ncu_ok=0
        echo "ncu failed for $k $d:" | tee -a "$out/ncu_status.txt"
        tail -n 5 "$out/ncu_${k}_${d}.log" | tee -a "$out/ncu_status.txt"
      fi
    done
  done
  if [ "$ncu_ok" = 1 ]; then
    # One header + units row, then every kernel's row.
    python3 - "$out" <<'PY'
import csv, glob, sys
rows, head, units = [], None, None
for f in sorted(glob.glob(f"{sys.argv[1]}/ncu_*_*.csv")):
    data = [r for r in csv.reader(open(f)) if r]
    i = next(i for i, r in enumerate(data) if "Kernel Name" in r)
    if head is None:
        head, units = data[i], data[i + 1]
    for r in data[i + 2:]:
        d = dict(zip(data[i], r))
        rows.append([d.get(h, "") for h in head])
with open(f"{sys.argv[1]}/ncu_raw.csv", "w", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerows([head, units] + rows)
PY
    ncu_csv="--ncu $out/ncu_raw.csv"
  fi
fi

echo "===== tables ====="
python3 -m trtbench.phase4 tables "$out/phase4_${MODE}.json" $ncu_csv | tee "$out/phase4_tables.md"
if [ -z "$ncu_csv" ]; then
  { echo; echo "#### Nsight Compute: not captured"; echo; sed 's/^/    /' "$out/ncu_status.txt"; } \
    | tee -a "$out/phase4_tables.md"
fi
echo
echo "results in $out (trtbench exit $status: 0 all trustworthy, 3 something was rejected)"
