"""INT8 post-training quantization recipes, chosen by graph structure.

ModelOpt's default INT8 recipe (Q/DQ on every Conv/Gemm/MatMul input and on
residual Adds, per-tensor activation scales) is kept for ResNet-50, where it
costs < 1 top-1 point. On the transformers it destroys the output: on T4,
BERT-base final hidden states fell to cosine 0.66 vs FP32 and GPT-2
perplexity rose from 59.5 to 345,425. The same graphs score the same in ONNX
Runtime on CPU, so the damage is in the quantized graph, not in TensorRT.

Diagnosed by quantizing one layer type at a time (BERT, ORT CPU, FP32 for
the unquantized ops):

  linears fed by LayerNorm (QKV, FFN up)    per-tensor scales vs channels
                                            (308/381) spanning 13-37x the
                                            median -> SmoothQuant fixes it
  FFN down-projection (fc2)                 input from GELU; still the
                                            costliest layer after smoothing,
                                            cause not established -> FP16
  attention BMMs, residual Adds             small but real loss -> FP16

So for transformers:
  1. SmoothQuant (Xiao et al. 2023) on every linear with a constant weight:
     per input channel j, s_j = amax(X_j)^a / amax(W_j,:)^(1-a), a = 0.5,
     X' = X / s (an explicit Mul, because LayerNorm's output also feeds the
     residual and its affine cannot absorb the scale), W' = s * W. Exact in
     real arithmetic; it only moves range from activations into weights.
  2. Quantize only "up" linears (K <= N) with a constant weight; keep
     down-projections (K > N), activation x activation MatMuls (which
     includes GPT-2's tied LM head), and Adds in FP16.

Every rule above is structural, so the same code serves BERT and GPT-2.
"""

from __future__ import annotations

import collections
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

ALPHA = 0.5


def _linears(model) -> List[Tuple[object, np.ndarray]]:
    """(node, weight as [K, N]) for MatMul/Gemm with a constant 2-D weight."""
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm") or n.input[0] in inits or n.input[1] not in inits:
            continue
        if len(inits[n.input[1]].dims) != 2:
            continue
        attrs = {a.name: a for a in n.attribute}
        if n.op_type == "Gemm" and (("transA" in attrs and attrs["transA"].i)
                                    or ("transB" in attrs and attrs["transB"].i)):
            continue  # not produced by our exporters; skip rather than guess the layout
        out.append((n, numpy_helper.to_array(inits[n.input[1]])))
    return out


def activation_absmax(model, feeds: Iterable[Dict[str, np.ndarray]],
                      tensors: List[str]) -> Dict[str, np.ndarray]:
    """Per-channel (last axis) max |x| of `tensors` over the calibration feeds."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    have = {o.name for o in probe.graph.output}
    for t in tensors:
        if t not in have:
            probe.graph.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    amax: Dict[str, np.ndarray] = {}
    for f in feeds:
        for t, v in zip(tensors, sess.run(tensors, f)):
            m = np.abs(v).reshape(-1, v.shape[-1]).max(axis=0)
            amax[t] = m if t not in amax else np.maximum(amax[t], m)
    return amax


def smooth(model, feeds: Iterable[Dict[str, np.ndarray]], alpha: float = ALPHA) -> int:
    """SmoothQuant in place. Returns the number of activation tensors smoothed."""
    from onnx import helper, numpy_helper

    groups: Dict[str, List[Tuple[object, np.ndarray]]] = collections.defaultdict(list)
    for n, w in _linears(model):
        groups[n.input[0]].append((n, w))
    weight_users = collections.Counter(n.input[1] for n, _ in _linears(model))
    names = list(groups)
    amax = activation_absmax(model, feeds, names)

    inits = {i.name: i for i in model.graph.initializer}
    order = {id(n): k for k, n in enumerate(model.graph.node)}
    inserts = []
    for t in names:
        users = groups[t]
        if any(weight_users[n.input[1]] > 1 for n, _ in users):
            continue  # a shared weight cannot take two different scalings
        wmax = np.max([np.abs(w).max(axis=1) for _, w in users], axis=0)
        s = np.maximum(amax[t], 1e-5) ** alpha / np.maximum(wmax, 1e-5) ** (1.0 - alpha)
        s = np.maximum(s, 1e-5).astype(np.float32)
        inv, out = f"{t}__smooth_inv", f"{t}__smoothed"
        model.graph.initializer.append(numpy_helper.from_array((1.0 / s).astype(np.float32), inv))
        for n, w in users:
            inits[n.input[1]].CopyFrom(
                numpy_helper.from_array((w * s[:, None]).astype(np.float32), n.input[1]))
            n.input[0] = out
        first = min((n for n, _ in users), key=lambda n: order[id(n)])
        inserts.append((order[id(first)], helper.make_node("Mul", [t, inv], [out], name=f"{t}__smooth")))
    for idx, node in sorted(inserts, key=lambda x: -x[0]):  # back to front keeps indices valid
        model.graph.node.insert(idx, node)
    return len(inserts)


def transformer_excludes(model) -> List[str]:
    """Node names kept out of INT8: down-projections and activation x activation MatMuls."""
    inits = {i.name for i in model.graph.initializer}
    keep = [n.name for n, w in _linears(model) if w.shape[0] > w.shape[1]]
    keep += [n.name for n in model.graph.node
             if n.op_type == "MatMul" and n.input[0] not in inits and n.input[1] not in inits]
    return keep


RECIPES = {
    "resnet50": "modelopt-default",
    "bert_base": "smoothquant-transformer",
    "gpt2": "smoothquant-transformer",
}


def describe(model_name: str) -> str:
    if RECIPES[model_name] == "modelopt-default":
        return ("modelopt.onnx.quantization int8, entropy calibration, default op selection, "
                "high_precision_dtype=fp16")
    return (f"SmoothQuant (alpha={ALPHA}) on constant-weight linears, then modelopt int8 "
            "(entropy) on up-projection linears only; down-projections, attention "
            "BMMs/LM head and residual Adds kept FP16")


def quantize_int8(model_name: str, src: str, dst: str, calib: Dict[str, np.ndarray],
                  batch: int, eps: List[str], high_precision_dtype: str = "fp16") -> Dict:
    """Write the INT8 variant of `src` to `dst`; return a description of what was done."""
    import os
    import tempfile

    import onnx
    from modelopt.onnx.quantization import quantize

    common = dict(quantize_mode="int8", calibration_data=calib, calibration_method="entropy",
                  calibration_eps=eps, high_precision_dtype=high_precision_dtype,
                  output_path=dst)
    if RECIPES[model_name] == "modelopt-default":
        quantize(onnx_path=src, **common)
        return {"recipe": RECIPES[model_name]}

    model = onnx.load(src)
    n = len(next(iter(calib.values())))
    feeds = ({k: v[i:i + batch] for k, v in calib.items()} for i in range(0, n - batch + 1, batch))
    smoothed = smooth(model, feeds)
    excludes = transformer_excludes(model)
    fd, tmp = tempfile.mkstemp(suffix=".onnx", dir=os.path.dirname(dst))
    os.close(fd)
    try:
        onnx.save(model, tmp)
        del model
        # ModelOpt treats these as regexes: anchor and escape, or node_addmm_3
        # would also exclude node_addmm_31.
        quantize(onnx_path=tmp, op_types_to_exclude=["Add"],
                 nodes_to_exclude=[f"^{re.escape(x)}$" for x in excludes], **common)
    finally:
        os.unlink(tmp)
    return {"recipe": RECIPES[model_name], "smoothquant_alpha": ALPHA,
            "smoothed_tensors": smoothed, "kept_fp16_nodes": len(excludes)}
