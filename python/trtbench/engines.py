"""Build TensorRT engines from the prepared ONNX variants and describe them.

Deliberately imports no PyTorch: this process loads exactly one CUDA runtime,
the one TensorRT was built for, which is also the one the C++ harness links.

TensorRT 11 networks are always strongly typed: an engine's precision is the
precision of its ONNX graph, not a builder flag. So
    fp32  the exported graph. TF32 is disabled (no effect on Turing; on Ampere
          and later it keeps "FP32" honest).
    fp16  the graph converted by ModelOpt AutoCast, FP32 network I/O kept.
    int8  ModelOpt post-training quantization: Q/DQ on Conv/Gemm/MatMul,
          calibrated (entropy) on 512 training samples, other ops in FP16.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

PRECISIONS = ("fp32", "fp16", "int8")

# Output sanity check (cosine vs the FP32 PyTorch output on the benchmark
# inputs). It exists to catch an engine that computes the wrong thing, not to
# grade quantization: that is the accuracy column. INT8 gets 0.95 because the
# first T4 run showed correct ResNet-50 INT8 engines at 0.971-0.985 (top-1
# within 0.8 points of FP32); the threshold was set after seeing that run.
SANITY_MIN_COSINE = {"fp32": 0.99, "fp16": 0.99, "int8": 0.95}


def build_engine(onnx_path: Path, plan_path: Path, timing_cache: Path) -> Dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    blob = timing_cache.read_bytes() if timing_cache.exists() else b""
    cache = config.create_timing_cache(blob)
    config.set_timing_cache(cache, ignore_mismatch=False)

    t = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    plan_path.write_bytes(bytes(plan))
    timing_cache.write_bytes(bytes(cache.serialize()))
    return {"trt_version": trt.__version__, "build_seconds": round(time.time() - t, 1),
            "layers_in_network": network.num_layers}


def build_all(work: Path, models, rebuild: bool = False) -> None:
    from .models import get

    for name in models:
        spec = get(name)
        mdir = work / name
        for batch in spec.batches:
            bdir = mdir / f"b{batch}"
            for prec in PRECISIONS:
                onnx_path = bdir / f"{prec}.onnx"
                meta_path = bdir / f"{prec}.meta.json"
                # A meta older than its ONNX describes an engine built from a
                # previous variant (prepare redid it), so rebuild. A fresh
                # engine keeps its build info but gets its meta rewritten, so
                # meta fields added later reach engines built earlier.
                fresh = (meta_path.exists() and onnx_path.exists()
                         and meta_path.stat().st_mtime >= onnx_path.stat().st_mtime)
                if not onnx_path.exists():
                    print(f"[engines] {name} b{batch} {prec}: NO ONNX ({onnx_path.name}); skipped")
                    continue
                if fresh and not rebuild:
                    old = json.loads(meta_path.read_text())
                    info = {k: old[k] for k in ("trt_version", "build_seconds", "layers_in_network")}
                    print(f"[engines] {name} b{batch} {prec}: exists")
                else:
                    print(f"[engines] {name} b{batch} {prec}: building", flush=True)
                    info = build_engine(onnx_path, bdir / f"{prec}.plan", work / "timing.cache")
                    print(f"[engines]   done in {info['build_seconds']} s "
                          f"(TensorRT {info['trt_version']})")
                stats = json.loads((bdir / f"{prec}.stats.json").read_text())
                prep = json.loads((bdir / "prepare.json").read_text())
                meta = {
                    "model": name,
                    "batch": batch,
                    "precision": prec,
                    "precision_label": prec,
                    "engine": f"{prec}.plan",
                    "onnx": onnx_path.name,
                    "flops": stats["flops"],
                    "bytes_lower_bound": stats["bytes_lower_bound"],
                    "inputs": prep["bench_inputs"],
                    "reference_outputs": prep["reference_outputs"],
                    "eval_inputs": {k: f"../{v}" for k, v in prep["eval_inputs"].items()},
                    "eval_count": prep["eval_count"],
                    "export": prep["export"],
                    "variant": prep["variants"][prec],
                    "sanity_min_cosine": SANITY_MIN_COSINE[prec],
                    **info,
                }
                meta_path.write_text(json.dumps(meta, indent=2))
