# Shared GPU setup for run_phase2.sh and run_phase3.sh; sourced, not run.
# Needs LOCK_MHZ. Checks for a GPU, installs the Python stack and TensorRT,
# builds the harness into build-gpu/, and sets $gpu, $tag, $run_dir and
# lock_clocks().
echo "===== setup ====="
# Fail here, with the reason, rather than several steps later. A Colab session
# that has run out of GPU quota comes back as a CPU runtime: no nvidia-smi.
nvidia-smi -L || { echo "${0##*/}: no NVIDIA GPU visible (on Colab: Runtime > Change runtime type > T4 GPU)" >&2; exit 1; }
[ -x /usr/local/cuda/bin/nvcc ] && export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "${0##*/}: nvcc not found (CUDA toolkit missing)" >&2; exit 1; }
python3 -m pip install -q -r requirements-phase2.txt
# Assign first: set -e does not see a failure inside eval "$(...)".
trt_env=$(scripts/setup_tensorrt.sh)
eval "$trt_env"
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
# One directory per lock setting: timings at different clocks never mix, and
# a rerun at a new clock does not reuse samples from the old one.
if [ "$LOCK_MHZ" != 0 ]; then tag="lock${LOCK_MHZ}"; else tag="unlocked"; fi
run_dir="results/${gpu}-$(date -u +%Y%m%d)"

# One clock lock for every timing process, so every backend and every
# profile is measured under the same clock policy. Sets $lockflag.
lock_clocks() {
  lockflag=""
  if [ "$LOCK_MHZ" != 0 ]; then
    if nvidia-smi -lgc "$LOCK_MHZ,$LOCK_MHZ"; then
      trap 'nvidia-smi -rgc >/dev/null' EXIT
      export TRTBENCH_EXTERNAL_LOCK_MHZ=$LOCK_MHZ
      lockflag="--clocks-locked-externally=$LOCK_MHZ"
    else
      echo "clock lock not permitted; running unlocked (drift is still audited)"
    fi
  fi
}
