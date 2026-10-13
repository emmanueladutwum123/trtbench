"""The three Phase 2 models, their benchmark shapes, and ONNX export.

Every backend sees the same model at the same shapes:

    resnet50   torchvision ResNet-50 (IMAGENET1K_V2)  x[B,3,224,224] fp32      -> logits[B,1000]
    bert_base  bert-base-uncased encoder             ids,mask[B,128] int64     -> hidden[B,128,768]
    gpt2       GPT-2 124M, prefill (no KV cache)      ids[B,128] int64          -> logits[B,128,50257]

GPT-2 is measured as a single prefill pass over 128 tokens. Token-by-token
decoding with a KV cache is a different workload and is not measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

SEQ_LEN = 128


@dataclass(frozen=True)
class ModelSpec:
    name: str
    batches: Tuple[int, ...]
    input_names: Tuple[str, ...]
    output_name: str
    hf_id: str = ""

    def input_shapes(self, batch: int) -> Dict[str, Tuple[int, ...]]:
        if self.name == "resnet50":
            return {"x": (batch, 3, 224, 224)}
        return {n: (batch, SEQ_LEN) for n in self.input_names}

    def input_dtypes(self) -> Dict[str, str]:
        return {"x": "float32"} if self.name == "resnet50" else {n: "int64" for n in self.input_names}


MODELS: Dict[str, ModelSpec] = {
    "resnet50": ModelSpec("resnet50", (1, 16), ("x",), "logits"),
    "bert_base": ModelSpec("bert_base", (1, 8), ("input_ids", "attention_mask"), "hidden",
                           "bert-base-uncased"),
    "gpt2": ModelSpec("gpt2", (1, 4), ("input_ids",), "logits", "gpt2"),
}


def build_module(spec: ModelSpec):
    """A torch.nn.Module in eval mode that returns exactly one tensor."""
    import torch

    if spec.name == "resnet50":
        import torchvision

        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        return m.eval()

    from transformers import AutoModel, AutoModelForCausalLM

    # Eager attention: identical math on every backend and a plain ONNX graph,
    # rather than a fused SDPA kernel only PyTorch gets.
    if spec.name == "bert_base":
        inner = AutoModel.from_pretrained(spec.hf_id, attn_implementation="eager")

        class Bert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = inner

            def forward(self, input_ids, attention_mask):
                return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        return Bert().eval()

    inner = AutoModelForCausalLM.from_pretrained(spec.hf_id, attn_implementation="eager")
    inner.config.use_cache = False

    class Gpt2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = inner

        def forward(self, input_ids):
            return self.m(input_ids=input_ids, use_cache=False).logits

    return Gpt2().eval()


def example_inputs(spec: ModelSpec, batch: int, device: str = "cpu") -> List:
    import torch

    shapes = spec.input_shapes(batch)
    if spec.name == "resnet50":
        return [torch.randn(shapes["x"], device=device)]
    ids = torch.randint(0, 1000, shapes[spec.input_names[0]], dtype=torch.int64, device=device)
    if spec.name == "bert_base":
        return [ids, torch.ones_like(ids)]
    return [ids]


def export_onnx(spec: ModelSpec, batch: int, out: Path, opset: int = 19) -> Dict[str, str]:
    """Export with static shapes (one file per batch size). Returns exporter info.

    Static shapes: every engine in this project is built for one shape, so
    shape inference is exact and FLOP counting needs no symbolic dimensions.
    """
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    module = build_module(spec)
    args = tuple(example_inputs(spec, batch))
    kwargs = dict(input_names=list(spec.input_names), output_names=[spec.output_name],
                  opset_version=opset)
    errors = []
    # TorchScript exporter first (mature on these three models); the dynamo
    # exporter is the fallback where TorchScript export is unavailable.
    for dynamo in (False, True):
        try:
            with torch.no_grad():
                torch.onnx.export(module, args, str(out), dynamo=dynamo, external_data=False, **kwargs)
        except Exception as e:  # noqa: BLE001 - report every exporter's failure
            errors.append(f"dynamo={dynamo}: {type(e).__name__}: {str(e)[:500]}")
            continue
        import onnx

        from .onnx_stats import fold_constants

        onnx.save(fold_constants(onnx.load(str(out))), str(out))
        return {"exporter": "dynamo" if dynamo else "torchscript", "torch": torch.__version__,
                "opset": str(opset), "constant_folded": "onnx-graphsurgeon"}
    raise RuntimeError(f"ONNX export of {spec.name} b{batch} failed:\n" + "\n".join(errors))


def model_ids() -> List[str]:
    return list(MODELS)


def get(name: str) -> ModelSpec:
    return MODELS[name]


def builder_for(name: str) -> Callable:
    return lambda: build_module(MODELS[name])
