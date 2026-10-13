"""Calibration and evaluation data. Calibration and evaluation never overlap.

    resnet50   Imagenette (fast.ai's 10-class ImageNet subset, 320px release).
               Calibration: 512 train images. Evaluation: 992 val images
               (divisible by both benchmark batch sizes), scored as top-1
               over the full 1000 ImageNet classes.
    bert_base  WikiText-2 (raw). Calibration: 512 windows of 128 tokens from
               train. Evaluation: 256 windows from test. No task head, so the
               metric is fidelity to FP32 PyTorch (cosine of hidden states).
    gpt2       WikiText-2 (raw). Calibration: 512 windows from train.
               Evaluation: 32 windows of 128 tokens from test, scored as
               perplexity (4,064 predicted tokens). Small, but every variant
               is scored on the same tokens, so deltas between them are paired.

Selection is deterministic (sorted paths, evenly strided), so every run sees
the same samples.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .models import SEQ_LEN

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
# Imagenette synsets -> ImageNet class names as torchvision lists them.
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888605": "parachute",
}

N_CALIB = 512
N_EVAL = {"resnet50": 992, "bert_base": 256, "gpt2": 32}
CALIBRATION_SET = {
    "resnet50": "512 Imagenette-320 train images (evenly strided over sorted paths), "
                "ResNet50_Weights.IMAGENET1K_V2 preprocessing",
    "bert_base": "512 WikiText-2 train windows of [CLS] + 126 tokens + [SEP]",
    "gpt2": "512 WikiText-2 train windows of 128 tokens",
}


def _strided(items: List, n: int) -> List:
    if len(items) < n:
        raise ValueError(f"need {n} samples, dataset has {len(items)}")
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


# --------------------------------------------------------------------------- images


def imagenette_root(cache: Path) -> Path:
    root = cache / "imagenette2-320"
    if not root.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tgz = cache / "imagenette2-320.tgz"
        if not tgz.exists():
            urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(cache)  # noqa: S202 - fixed, trusted archive
    return root


def imagenet_label_index() -> Dict[str, int]:
    import torchvision

    cats = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    out = {}
    for syn, name in IMAGENETTE_CLASSES.items():
        if name not in cats:
            raise ValueError(f"ImageNet category '{name}' not found in torchvision metadata")
        out[syn] = cats.index(name)
    return out


def _preprocess(paths: List[Path]) -> np.ndarray:
    import torch
    from PIL import Image

    tf = __import__("torchvision").models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    with torch.no_grad():
        return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).numpy()


def image_split(cache: Path, split: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    root = imagenette_root(cache) / split
    labels = imagenet_label_index()
    paths = sorted(p for p in root.glob("*/*.JPEG"))
    chosen = _strided(paths, n)
    y = np.array([labels[p.parent.name] for p in chosen], dtype=np.int64)
    return _preprocess(chosen).astype(np.float32), y


# --------------------------------------------------------------------------- text


def _wikitext_ids(split: str, tokenizer_id: str) -> np.ndarray:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    # Tokenize the whole split once; windows are cut from the token stream.
    ids = tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    return np.asarray(ids, dtype=np.int64)


def text_windows(split: str, tokenizer_id: str, n: int, bert_framing: bool = False) -> np.ndarray:
    """n windows of SEQ_LEN tokens. With bert_framing, each window is
    [CLS] + 126 tokens + [SEP], the input shape BERT was trained on."""
    ids = _wikitext_ids(split, tokenizer_id)
    body = SEQ_LEN - 2 if bert_framing else SEQ_LEN
    windows = [ids[i : i + body] for i in range(0, len(ids) - body, body)]
    out = np.stack(_strided(windows, n))
    if bert_framing:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_id)
        cls = np.full((n, 1), tok.cls_token_id, dtype=np.int64)
        sep = np.full((n, 1), tok.sep_token_id, dtype=np.int64)
        out = np.concatenate([cls, out, sep], axis=1)
    return out


def model_inputs(model: str, cache: Path, split: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Inputs as {onnx input name: array[N, ...]} and labels (images only)."""
    n = N_CALIB if split == "calib" else N_EVAL[model]
    if model == "resnet50":
        x, y = image_split(cache, "train" if split == "calib" else "val", n)
        return {"x": x}, y
    hf_split = "train" if split == "calib" else "test"
    if model == "bert_base":
        ids = text_windows(hf_split, "bert-base-uncased", n, bert_framing=True)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}, np.empty(0)
    ids = text_windows(hf_split, "gpt2", n)
    return {"input_ids": ids}, np.empty(0)
