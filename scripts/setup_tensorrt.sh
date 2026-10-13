#!/usr/bin/env bash
# Make TensorRT available to both the Python tools and the C++ build, from one
# pip install so engine builder and engine runner are the same TensorRT.
#
#   eval "$(scripts/setup_tensorrt.sh)"          # full: Python bindings + libs
#   eval "$(scripts/setup_tensorrt.sh --libs)"   # C++ only (CI)
#
# Prints `export TENSORRT_INCLUDE_DIR=... TENSORRT_LIBRARY=... TRT_VERSION=...`.
#
# The CUDA major is taken from nvcc, so the C++ harness, libnvinfer and the
# builder all use the toolkit the harness is compiled with. The pip wheels
# ship no C++ headers; the headers are fetched from NVIDIA's TensorRT GitHub
# repository at the tag matching the installed version.
set -euo pipefail

TRT_VERSION=${TRT_VERSION:-11.4.0.106}
mode=${1:-full}
cuda_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
[ -n "$cuda_major" ] || { echo "setup_tensorrt: cannot read CUDA version from nvcc" >&2; exit 1; }

pkg="tensorrt-cu${cuda_major}"
[ "$mode" = "--libs" ] && pkg="tensorrt-cu${cuda_major}-libs"
python3 -m pip install -q --extra-index-url https://pypi.nvidia.com "${pkg}==${TRT_VERSION}" >&2

lib=$(python3 - <<'EOF'
import glob, site, sys
roots = site.getsitepackages() + [site.getusersitepackages()]
hits = sorted({p for r in roots for p in glob.glob(f"{r}/**/libnvinfer.so.*", recursive=True)
               if "builder_resource" not in p and "lean" not in p and "dispatch" not in p})
if not hits:
    sys.exit("setup_tensorrt: libnvinfer.so.* not found in site-packages")
print(hits[0])
EOF
)

tag="v$(echo "$TRT_VERSION" | cut -d. -f1-2)"
inc="$(pwd)/third_party/tensorrt-${TRT_VERSION}/include"
mkdir -p "$inc"
for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPluginBase.h NvInferRuntime.h \
         NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
  [ -s "$inc/$h" ] || curl -sfL -o "$inc/$h" \
    "https://raw.githubusercontent.com/NVIDIA/TensorRT/${tag}/include/$h" \
    || { echo "setup_tensorrt: could not fetch $h at $tag" >&2; exit 1; }
done

# The headers must describe the library we link: compare version macros.
hv=$(awk '/#define TRT_(MAJOR|MINOR|PATCH)_ENTERPRISE / {print $3}' "$inc/NvInferVersion.h" | paste -sd. -)
want=$(echo "$TRT_VERSION" | cut -d. -f1-3)
[ "$hv" = "$want" ] || { echo "setup_tensorrt: headers are $hv, library is $want" >&2; exit 1; }

echo "export TENSORRT_INCLUDE_DIR='$inc' TENSORRT_LIBRARY='$lib' TRT_VERSION='$TRT_VERSION' TRT_CUDA_MAJOR='$cuda_major'"
