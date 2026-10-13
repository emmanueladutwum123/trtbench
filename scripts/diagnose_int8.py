#!/usr/bin/env python3
"""Reproduce the INT8 diagnosis behind python/trtbench/int8_recipe.py on CPU.

    PYTHONPATH=python python3 scripts/diagnose_int8.py WORKDIR [--out FILE]

Exports BERT-base and GPT-2 at batch 1, calibrates on 64 WikiText-2 train
windows and scores 16 (BERT) / 8 (GPT-2) test windows in ONNX Runtime on CPU.
Ops left out of INT8 run in FP32 here, because ONNX Runtime's CPU FP16
kernels lose accuracy that TensorRT's do not: the first T4 run's broken BERT
INT8 engines scored 0.65-0.66 cosine, which matches this FP32 setup and not
the same graph with FP16 ops (both printed below). Needs torch, transformers, datasets, onnx, onnxruntime and
nvidia-modelopt; no GPU.

These are diagnosis numbers on a small CPU set, not the published accuracy:
that comes from the T4 run, on the full evaluation sets, through TensorRT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

N_CALIB, N_EVAL = 64, {"bert_base": 16, "gpt2": 8}


def _prepare(model: str, work: Path):
    from trtbench import data, models

    work.mkdir(parents=True, exist_ok=True)
    if not (work / "fp32.onnx").exists():
        models.export_onnx(models.get(model), 1, work / "fp32.onnx")
    if not (work / "calib.npz").exists():
        tok = "bert-base-uncased" if model == "bert_base" else "gpt2"
        bert = model == "bert_base"
        c = data.text_windows("train", tok, N_CALIB, bert_framing=bert)
        e = data.text_windows("test", tok, N_EVAL[model], bert_framing=bert)
        feeds = {"input_ids": c} | ({"attention_mask": np.ones_like(c)} if bert else {})
        np.savez(work / "calib.npz", **feeds)
        np.save(work / "eval.npy", e)
    return dict(np.load(work / "calib.npz")), np.load(work / "eval.npy")


def _run(model: str, path: Path, ev: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = []
    for i in range(len(ev)):
        f = {"input_ids": ev[i:i + 1]}
        if model == "bert_base":
            f["attention_mask"] = np.ones_like(ev[i:i + 1])
        outs.append(s.run(None, f)[0])
    return np.concatenate(outs).astype(np.float32)


def _score(model: str, out: np.ndarray, ref: np.ndarray, ev: np.ndarray) -> str:
    from trtbench.accuracy import cosine_vs_ref, perplexity

    if model == "bert_base":
        return f"cosine vs FP32 {cosine_vs_ref(out, ref):.4f}"
    return f"perplexity {perplexity(out, ev):.2f} (FP32 {perplexity(ref, ev):.2f})"


def _quantize(src: Path, dst: Path, calib, high_precision_dtype="fp32", **kw) -> None:
    from modelopt.onnx.quantization import quantize

    quantize(onnx_path=str(src), quantize_mode="int8", calibration_data=calib,
             calibration_method="entropy", calibration_eps=["cpu"],
             high_precision_dtype=high_precision_dtype, output_path=str(dst),
             log_level="ERROR", **kw)


def main() -> int:
    import onnx

    from trtbench import int8_recipe

    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    lines = []

    def say(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"INT8 diagnosis, ONNX Runtime CPU, batch 1, {N_CALIB} calibration windows, "
        "unquantized ops in FP32")
    for model in ("bert_base", "gpt2"):
        w = a.work / model
        calib, ev = _prepare(model, w)
        ref = _run(model, w / "fp32.onnx", ev)
        say(f"\n{model} ({len(ev)} test windows)")

        _quantize(w / "fp32.onnx", w / "default.onnx", calib)
        say(f"  ModelOpt default INT8 (every MatMul/Gemm + residual Adds): "
            f"{_score(model, _run(model, w / 'default.onnx', ev), ref, ev)}")

        if model == "bert_base":
            m = onnx.load(str(w / "fp32.onnx"))
            lns = [n.output[0] for n in m.graph.node if n.op_type == "LayerNormalization"
                   and "attention/output" in n.name]
            feeds = [{"input_ids": ev[i:i + 1], "attention_mask": np.ones_like(ev[i:i + 1])}
                     for i in range(len(ev))]
            amax = int8_recipe.activation_absmax(m, feeds, lns)
            del m
            for layer, t in enumerate(lns):
                ch = amax[t]
                top = np.argsort(ch)[-2:][::-1]
                say(f"  layer {layer:2d} attention LayerNorm output: max/median channel range "
                    f"{ch.max() / np.median(ch):5.1f}x, top channels {list(map(int, top))}")
            _quantize(w / "fp32.onnx", w / "default16.onnx", calib, high_precision_dtype="fp16")
            say(f"  same, other ops FP16 (ORT CPU FP16 kernels; not TRT):    "
                f"{_score(model, _run(model, w / 'default16.onnx', ev), ref, ev)}")
            up = [r".*/intermediate/dense/MatMul$"]
            _quantize(w / "fp32.onnx", w / "fc1.onnx", calib, nodes_to_quantize=up)
            say(f"  only FFN up-projections (fed by LayerNorm):              "
                f"{_score(model, _run(model, w / 'fc1.onnx', ev), ref, ev)}")

            sm = onnx.load(str(w / "fp32.onnx"))
            feeds = [{k: v[i:i + 1] for k, v in calib.items()} for i in range(N_CALIB)]
            int8_recipe.smooth(sm, feeds)
            onnx.save(sm, str(w / "smoothed.onnx"))
            say(f"  SmoothQuant alone, FP32 (must be ~1):                    "
                f"{_score(model, _run(model, w / 'smoothed.onnx', ev), ref, ev)}")
            _quantize(w / "smoothed.onnx", w / "fc1_sq.onnx", calib, nodes_to_quantize=up)
            say(f"  only FFN up-projections, after SmoothQuant:              "
                f"{_score(model, _run(model, w / 'fc1_sq.onnx', ev), ref, ev)}")
            _quantize(w / "smoothed.onnx", w / "fc2_sq.onnx", calib,
                      nodes_to_quantize=[r".*/layer\.2/output/dense/MatMul$"])
            say(f"  only layer 2 FFN down-projection, after SmoothQuant:     "
                f"{_score(model, _run(model, w / 'fc2_sq.onnx', ev), ref, ev)}")

        info = int8_recipe.quantize_int8(model, str(w / "fp32.onnx"), str(w / "recipe.onnx"),
                                         calib, 1, ["cpu"], high_precision_dtype="fp32")
        say(f"  trtbench recipe ({info['smoothed_tensors']} tensors smoothed, "
            f"{info['kept_fp16_nodes']} linears/BMMs kept out):        "
            f"{_score(model, _run(model, w / 'recipe.onnx', ev), ref, ev)}")

    if a.out:
        a.out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
