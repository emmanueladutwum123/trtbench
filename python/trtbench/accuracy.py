"""Accuracy metrics. Pure numpy, so they are unit-tested on CPU.

    resnet50   top-1 (%) against Imagenette labels, argmax over 1000 classes
    bert_base  mean per-sequence cosine similarity of the final hidden states
               with FP32 PyTorch eager (1.0 = identical). A fidelity metric,
               not task accuracy: the encoder has no task head.
    gpt2       perplexity over the evaluation windows (next-token NLL)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

METRIC = {"resnet50": "top1_pct", "bert_base": "cosine_vs_fp32", "gpt2": "perplexity"}
# Direction of "better", for wording deltas.
HIGHER_IS_BETTER = {"top1_pct": True, "cosine_vs_fp32": True, "perplexity": False}


def top1_pct(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(100.0 * np.mean(np.argmax(logits, axis=-1) == labels))


def cosine_vs_ref(hidden: np.ndarray, ref: np.ndarray) -> float:
    h = hidden.reshape(len(hidden), -1).astype(np.float64)
    r = ref.reshape(len(ref), -1).astype(np.float64)
    cos = np.sum(h * r, axis=1) / np.sqrt(np.sum(h * h, axis=1) * np.sum(r * r, axis=1))
    return float(np.mean(cos))


def perplexity(logits: np.ndarray, ids: np.ndarray, chunk: int = 4) -> float:
    """exp(mean NLL of token t+1 given tokens <= t), computed in float64 in
    chunks so a [32,128,50257] array never needs a second full-size copy."""
    total, count = 0.0, 0
    for i in range(0, len(logits), chunk):
        lg = np.asarray(logits[i : i + chunk, :-1], dtype=np.float64)
        tgt = ids[i : i + chunk, 1:]
        m = lg.max(axis=-1, keepdims=True)
        lse = (m + np.log(np.exp(lg - m).sum(axis=-1, keepdims=True)))[..., 0]
        picked = np.take_along_axis(lg, tgt[..., None], axis=-1)[..., 0]
        total += float(np.sum(lse - picked))
        count += tgt.size
    return float(np.exp(total / count))


def score(model: str, outputs: np.ndarray, eval_inputs: Dict[str, np.ndarray],
          labels: np.ndarray, fp32_ref: np.ndarray | None) -> Dict:
    metric = METRIC[model]
    if model == "resnet50":
        value = top1_pct(outputs, labels)
    elif model == "bert_base":
        if fp32_ref is None:
            raise ValueError("BERT fidelity needs the FP32 PyTorch reference outputs")
        value = cosine_vs_ref(outputs, fp32_ref)
    else:
        value = perplexity(outputs, eval_inputs["input_ids"])
    return {"metric": metric, "value": value, "n": int(len(outputs))}
