"""Phase 2 steps. Each runs in its own process (see scripts/run_phase2.sh), so
no process mixes PyTorch's CUDA runtime with TensorRT's.

    python -m trtbench.phase2 prepare   WORK [--models ...]   # torch + ModelOpt
    python -m trtbench.phase2 engines   WORK [--models ...]   # TensorRT only
    python -m trtbench.phase2 baselines WORK OUT [--models ...]  # torch + ORT
    python -m trtbench.phase2 score     WORK META OUTPUTS ACC.json  # numpy only

Every step skips work whose output already exists, so an interrupted Colab
session resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import accuracy, int8_recipe, models

# Bump when prepare's output changes meaning, so a resumed run redoes stale
# variants. The value is the version each variant was last changed in.
#   2: FP16 AutoCast ranges come from real inputs, not random ones.
#   3: transformer INT8 uses SmoothQuant and keeps sensitive ops FP16.
PREPARE_VERSION = 3
VARIANT_SINCE = {"fp32": 1, "fp16": 2, "int8": 3}

# --------------------------------------------------------------------------- io


def _save_bin(path: Path, a: np.ndarray) -> None:
    np.ascontiguousarray(a).tofile(path)


def _load_eval(mdir: Path, spec: models.ModelSpec) -> Dict[str, np.ndarray]:
    meta = json.loads((mdir / "eval.json").read_text())
    n = meta["count"]
    out = {}
    for name, shape_tail in meta["shapes"].items():
        dtype = np.float32 if spec.input_dtypes()[name] == "float32" else np.int64
        out[name] = np.fromfile(mdir / f"eval_{name}.bin", dtype=dtype).reshape([n] + shape_tail)
    return out


def _add_accuracy(path: Path, name: str, entry: Dict) -> None:
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[name] = entry
    path.write_text(json.dumps(doc, indent=2))


def _torch_no_tf32():
    import torch

    # FP32 means FP32 (no effect on Turing; matters on Ampere and later).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- prepare


def prepare(work: Path, names: List[str], cache: Path) -> None:
    import onnx
    import torch

    from . import data
    from .onnx_stats import graph_stats

    _torch_no_tf32()
    for name in names:
        spec = models.get(name)
        mdir = work / name
        mdir.mkdir(parents=True, exist_ok=True)

        # Evaluation set, shared by every batch size and backend.
        if not (mdir / "eval.json").exists():
            print(f"[prepare] {name}: evaluation data", flush=True)
            ev, labels = data.model_inputs(name, cache, "eval")
            for k, v in ev.items():
                _save_bin(mdir / f"eval_{k}.bin", v)
            np.save(mdir / "eval_labels.npy", labels)
            (mdir / "eval.json").write_text(json.dumps(
                {"count": len(next(iter(ev.values()))),
                 "shapes": {k: list(v.shape[1:]) for k, v in ev.items()}}))
        ev = _load_eval(mdir, spec)

        # Calibration set, saved once: the ModelOpt steps read it in a child process.
        calib_path = mdir / "calib.npz"
        if not calib_path.exists():
            print(f"[prepare] {name}: calibration data", flush=True)
            calib, _ = data.model_inputs(name, cache, "calib")
            np.savez(calib_path, **calib)
            del calib

        module = None
        for batch in spec.batches:
            bdir = mdir / f"b{batch}"
            bdir.mkdir(exist_ok=True)
            done = bdir / "prepare.json"
            prev = json.loads(done.read_text()) if done.exists() else {}
            if prev.get("version") == PREPARE_VERSION:
                print(f"[prepare] {name} b{batch}: exists")
                continue
            # Redo only the variants that changed since this directory was prepared.
            for prec, since in VARIANT_SINCE.items():
                if prev.get("version", 0) < since:
                    (bdir / f"{prec}.onnx").unlink(missing_ok=True)

            if (bdir / "fp32.onnx").exists() and "export" in prev:
                export = prev["export"]
            else:
                print(f"[prepare] {name} b{batch}: export", flush=True)
                export = models.export_onnx(spec, batch, bdir / "fp32.onnx")
            if not (bdir / "fp16.onnx").exists():
                print(f"[prepare] {name} b{batch}: FP16 (ModelOpt AutoCast)", flush=True)
                _modelopt_child("fp16", bdir, calib_path, batch)
            if not (bdir / "int8.onnx").exists():
                print(f"[prepare] {name} b{batch}: INT8 ({int8_recipe.RECIPES[name]}, "
                      f"{data.N_CALIB} calibration samples)", flush=True)
                _modelopt_child("int8", bdir, calib_path, batch)
            int8_info = json.loads((bdir / "int8.recipe.json").read_text())

            for prec in ("fp32", "fp16", "int8"):
                st = graph_stats(onnx.load(str(bdir / f"{prec}.onnx")))
                (bdir / f"{prec}.stats.json").write_text(json.dumps(st.to_dict(), indent=2))

            # Benchmark inputs: the first `batch` evaluation samples (real data).
            bench = {k: v[:batch] for k, v in ev.items()}
            for k, v in bench.items():
                _save_bin(bdir / f"input_{k}.bin", v)
            if module is None:
                module = models.build_module(spec).cuda()
            with torch.no_grad():
                ref = module(*[torch.from_numpy(bench[k]).cuda() for k in spec.input_names])
            _save_bin(bdir / f"ref_{spec.output_name}.bin", ref.float().cpu().numpy())

            (bdir / "prepare.json").write_text(json.dumps({
                "version": PREPARE_VERSION,
                "export": export,
                "bench_inputs": {k: f"input_{k}.bin" for k in spec.input_names},
                "reference_outputs": {spec.output_name: f"ref_{spec.output_name}.bin"},
                "eval_inputs": {k: f"eval_{k}.bin" for k in spec.input_names},
                "eval_count": int(len(next(iter(ev.values())))),
                "variants": {
                    "fp32": {"how": "exported graph, constant-folded"},
                    "fp16": {"how": "modelopt.onnx.autocast fp16, keep_io_types=True, "
                                    f"ranges from {batch} real calibration samples"},
                    "int8": {"how": int8_recipe.describe(name), **int8_info,
                             "calibration": data.CALIBRATION_SET[name],
                             "calibration_samples": data.N_CALIB},
                },
            }, indent=2))
        del module


def _modelopt_child(kind: str, bdir: Path, calib_path: Path, batch: int) -> None:
    """Run one ModelOpt conversion in a fresh process. AutoCast keeps every
    intermediate tensor of a reference run in host memory, and ModelOpt and
    ONNX Runtime do not hand memory back between models; in one long-lived
    process BERT b8 was OOM-killed on a 12.7 GB Colab VM. A child process
    returns everything to the OS when it exits."""
    import subprocess

    cmd = [sys.executable, "-m", "trtbench.phase2", "_modelopt", kind, str(bdir),
           str(calib_path), str(batch)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ModelOpt {kind} for {bdir} failed (exit {r.returncode}"
                           f"{', killed: out of memory?' if r.returncode == -9 else ''})")


def _modelopt(kind: str, bdir: Path, calib_path: Path, batch: int) -> None:
    calib = dict(np.load(calib_path))
    src = str(bdir / "fp32.onnx")
    if kind == "fp16":
        import onnx
        from modelopt.onnx.autocast import convert_to_mixed_precision

        # Exactly one batch of real data. Without it AutoCast feeds random
        # values in [-1, 1], which for token-id inputs means ids -1/0/1, and
        # its keep-in-FP32 decisions come from ranges no real input produces.
        # One batch, because it keeps every intermediate of every batch.
        one = bdir / "autocast_inputs.npz"
        np.savez(one, **{k: v[:batch] for k, v in calib.items()})
        model = convert_to_mixed_precision(onnx_path=src, low_precision_type="fp16",
                                           keep_io_types=True, calibration_data=str(one))
        onnx.save(model, str(bdir / "fp16.onnx"))
        one.unlink()
    else:
        model = bdir.parent.name
        info = int8_recipe.quantize_int8(model, src, str(bdir / "int8.onnx"), calib, batch,
                                         eps=["cuda:0", "cpu"])
        (bdir / "int8.recipe.json").write_text(json.dumps(info, indent=2))


# --------------------------------------------------------------------------- baselines


def baselines(work: Path, out: Path, names: List[str], cfg_overrides: Dict) -> None:
    """Time PyTorch eager (FP32, FP16) and ONNX Runtime CUDA EP (FP32), and
    score their accuracy on the same evaluation set as the engines."""
    import torch

    from .pyrunner import NvmlClocks, RunConfig, run, sanity_checksum, torch_pci_bus_id

    _torch_no_tf32()
    out.mkdir(parents=True, exist_ok=True)
    acc_path = out / "accuracy.json"
    cfg = RunConfig(**cfg_overrides)
    clocks = NvmlClocks(torch_pci_bus_id())
    stream = torch.cuda.Stream()

    for name in names:
        spec = models.get(name)
        mdir = work / name
        ev = _load_eval(mdir, spec)
        labels = np.load(mdir / "eval_labels.npy")
        fp32_ref_path = mdir / "eval_ref_fp32.npy"

        for precision in ("fp32", "fp16"):
            module = models.build_module(spec).cuda()
            if precision == "fp16":
                module = module.half()

            def to_dev(a: np.ndarray):
                t = torch.from_numpy(a).cuda()
                return t.half() if (precision == "fp16" and t.is_floating_point()) else t

            # Accuracy on the full evaluation set, at each benchmark batch size.
            # The FP32 batch-1 outputs are the reference for BERT's fidelity
            # metric (and for every other backend's).
            n_eval = len(next(iter(ev.values())))
            for batch in spec.batches:
                outs = []
                with torch.no_grad():
                    for i in range(0, n_eval, batch):
                        outs.append(module(*[to_dev(ev[k][i:i + batch])
                                             for k in spec.input_names]).float().cpu().numpy())
                outs = np.concatenate(outs)
                if precision == "fp32" and batch == spec.batches[0]:
                    np.save(fp32_ref_path, outs)
                ref = np.load(fp32_ref_path) if name == "bert_base" else None
                _add_accuracy(acc_path, f"torch_eager/{name}_b{batch}/{precision}",
                              accuracy.score(name, outs, ev, labels, ref))
                del outs

            for batch in spec.batches:
                bdir = mdir / f"b{batch}"
                sample = out / "samples" / f"torch_eager_{name}_b{batch}_{precision}.json"
                if sample.exists():
                    continue
                stats = json.loads((bdir / f"{precision}.stats.json").read_text())
                bench = [to_dev(np.fromfile(bdir / f"input_{k}.bin",
                                            dtype=np.float32 if spec.input_dtypes()[k] == "float32"
                                            else np.int64).reshape(spec.input_shapes(batch)[k]))
                         for k in spec.input_names]
                ref_bench = np.fromfile(bdir / f"ref_{spec.output_name}.bin", dtype=np.float32)
                holder = {}

                def enqueue():
                    with torch.no_grad(), torch.cuda.stream(stream):
                        holder["y"] = module(*bench)

                def checksum():
                    stream.synchronize()
                    return sanity_checksum({spec.output_name: holder["y"].float().cpu().numpy()},
                                           {spec.output_name: ref_bench})

                print(f"[baselines] torch_eager {name} b{batch} {precision}", flush=True)
                doc = run(f"torch_eager/{name}_b{batch}/{precision}",
                          {"backend": "torch_eager", "model": name, "batch": str(batch),
                           "precision": precision, "torch_version": torch.__version__},
                          precision, enqueue, stream, cfg, stats["bytes_lower_bound"],
                          stats["flops"], checksum, clocks)
                sample.parent.mkdir(parents=True, exist_ok=True)
                sample.write_text(json.dumps(doc))
            module = None
            torch.cuda.empty_cache()

        _onnxruntime(work, out, spec, ev, labels, cfg, clocks, stream, acc_path)


def _onnxruntime(work, out, spec, ev, labels, cfg, clocks, stream, acc_path) -> None:
    import torch

    from .pyrunner import run, sanity_checksum

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"CUDAExecutionProvider unavailable "
                               f"(have {ort.get_available_providers()})")
    except Exception as e:  # noqa: BLE001 - a missing baseline is recorded, not fatal
        msg = f"ONNX Runtime CUDA EP unavailable: {type(e).__name__}: {e}"
        print(f"[baselines] {msg}")
        (out / "missing_baselines.txt").open("a").write(f"{spec.name}: {msg}\n")
        return

    name = spec.name
    mdir = work / name
    ref = np.load(mdir / "eval_ref_fp32.npy") if name == "bert_base" else None
    for batch in spec.batches:
        bdir = mdir / f"b{batch}"
        opts = {"device_id": 0, "user_compute_stream": str(stream.cuda_stream)}
        sess = ort.InferenceSession(str(bdir / "fp32.onnx"),
                                    providers=[("CUDAExecutionProvider", opts)])
        result_name = f"onnxruntime/{name}_b{batch}/fp32"

        outs = []
        n = len(next(iter(ev.values())))
        for i in range(0, n, batch):
            outs.append(sess.run(None, {k: ev[k][i:i + batch] for k in spec.input_names})[0])
        _add_accuracy(acc_path, result_name,
                      accuracy.score(name, np.concatenate(outs), ev, labels, ref))
        del outs

        sample = out / "samples" / f"onnxruntime_{name}_b{batch}_fp32.json"
        if sample.exists():
            continue
        stats = json.loads((bdir / "fp32.stats.json").read_text())
        binding = sess.io_binding()
        keep = []
        for k in spec.input_names:
            a = np.fromfile(bdir / f"input_{k}.bin",
                            dtype=np.float32 if spec.input_dtypes()[k] == "float32" else np.int64
                            ).reshape(spec.input_shapes(batch)[k])
            t = torch.from_numpy(a).cuda()
            keep.append(t)
            binding.bind_input(k, "cuda", 0, a.dtype, list(a.shape), t.data_ptr())
        out_shape = sess.get_outputs()[0].shape
        y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        binding.bind_output(spec.output_name, "cuda", 0, np.float32, list(out_shape), y.data_ptr())
        ro = ort.RunOptions()
        # Run() otherwise synchronizes the stream before returning, which would
        # fold the device time into the host time.
        ro.add_run_config_entry("disable_synchronize_execution_providers", "1")
        ref_bench = np.fromfile(bdir / f"ref_{spec.output_name}.bin", dtype=np.float32)

        def enqueue():
            sess.run_with_iobinding(binding, ro)

        def checksum():
            stream.synchronize()
            return sanity_checksum({spec.output_name: y.cpu().numpy()},
                                   {spec.output_name: ref_bench})

        print(f"[baselines] onnxruntime {name} b{batch} fp32", flush=True)
        doc = run(result_name,
                  {"backend": "onnxruntime", "model": name, "batch": str(batch),
                   "precision": "fp32", "ort_version": ort.__version__},
                  "fp32", enqueue, stream, cfg, stats["bytes_lower_bound"], stats["flops"],
                  checksum, clocks)
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(json.dumps(doc))
        sess = binding = None


# --------------------------------------------------------------------------- score


def score_engine(work: Path, meta_path: Path, outputs_path: Path, acc_path: Path) -> None:
    meta = json.loads(meta_path.read_text())
    name = meta["model"]
    spec = models.get(name)
    mdir = work / name
    ev = _load_eval(mdir, spec)
    n = meta["eval_count"]
    ref = np.load(mdir / "eval_ref_fp32.npy", mmap_mode="r") if name == "bert_base" else None
    per_sample = {"resnet50": [1000], "bert_base": [models.SEQ_LEN, 768],
                  "gpt2": [models.SEQ_LEN, 50257]}[name]
    outs = np.memmap(outputs_path, dtype=np.float32, mode="r", shape=tuple([n] + per_sample))
    entry = accuracy.score(name, outs, ev, np.load(mdir / "eval_labels.npy"), ref)
    _add_accuracy(acc_path, f"tensorrt/{name}_b{meta['batch']}/{meta['precision_label']}", entry)
    print(f"[score] tensorrt {name} b{meta['batch']} {meta['precision_label']}: "
          f"{entry['metric']} = {entry['value']:.6g}")


# --------------------------------------------------------------------------- cli


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trtbench.phase2")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "engines", "baselines"):
        s = sub.add_parser(c)
        s.add_argument("work", type=Path)
        if c == "baselines":
            s.add_argument("out", type=Path)
            s.add_argument("--mode", default="latency")
        s.add_argument("--models", nargs="*", default=models.model_ids())
        s.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "trtbench")
    s = sub.add_parser("_modelopt")  # internal: one conversion, see _modelopt_child
    s.add_argument("kind", choices=("fp16", "int8"))
    s.add_argument("bdir", type=Path)
    s.add_argument("calib", type=Path)
    s.add_argument("batch", type=int)
    s = sub.add_parser("score")
    s.add_argument("work", type=Path)
    s.add_argument("meta", type=Path)
    s.add_argument("outputs", type=Path)
    s.add_argument("accuracy", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        prepare(a.work, a.models, a.cache)
    elif a.cmd == "engines":
        from .engines import build_all

        build_all(a.work, a.models)
    elif a.cmd == "_modelopt":
        _modelopt(a.kind, a.bdir, a.calib, a.batch)
    elif a.cmd == "baselines":
        baselines(a.work, a.out, a.models, {"mode": a.mode})
    else:
        score_engine(a.work, a.meta, a.outputs, a.accuracy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
