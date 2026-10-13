"""Analytic FLOPs and lower-bound DRAM bytes of one inference, from ONNX.

FLOPs count Conv, Gemm and MatMul only (2 per multiply-accumulate).
Elementwise ops, softmax, normalization and pooling are not counted, so the
figure is a lower bound, dominated by the ops that dominate it in practice.

Bytes are a lower bound on DRAM traffic: every weight read once at the dtype
stored in this variant's ONNX file (FP32, FP16 or INT8), plus the network
inputs and outputs. A table read only through Gather (token and position
embeddings) counts only the rows gathered: GPT-2 reads 128 of its 50,257
embedding rows per sequence, and counting the whole table would overstate a
bound that must never be overstated. Likewise, a weight consumed only by
QuantizeLinear (explicit INT8 quantization stores FP16 weights followed by Q)
counts at 1 byte, because TensorRT folds that Q into INT8 weights at build. Real traffic is higher (activations that spill, weights
re-read across tiles), so achieved bandwidth computed from it is a lower bound
too. That is the safe direction for the auditor: a lower bound that already
exceeds the hardware peak is certainly impossible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

_ELEM_BYTES = {1: 4, 2: 1, 3: 1, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 16: 2}


@dataclass
class GraphStats:
    flops: int
    weight_bytes: int
    io_bytes: int
    weight_count: int
    flops_by_op: Dict[str, int]
    uncounted_ops: List[str]

    @property
    def bytes_lower_bound(self) -> int:
        return self.weight_bytes + self.io_bytes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bytes_lower_bound"] = self.bytes_lower_bound
        return d


def _shape(vi) -> List[int]:
    dims = vi.type.tensor_type.shape.dim
    if not all(d.HasField("dim_value") for d in dims):
        raise ValueError(f"tensor {vi.name} has a non-static shape; export with static shapes")
    return [d.dim_value for d in dims]


def _elem_bytes(elem_type: int) -> int:
    if elem_type not in _ELEM_BYTES:
        raise ValueError(f"unsupported ONNX element type {elem_type}")
    return _ELEM_BYTES[elem_type]


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            from onnx import helper

            return helper.get_attribute_value(a)
    return default


def static_shapes(model):
    """(inferred graph, {tensor: static shape}) for every tensor with one."""
    import onnx

    # data_prop resolves Shape -> Reshape chains the exporters emit.
    g = onnx.shape_inference.infer_shapes(model, data_prop=True).graph
    shapes: Dict[str, List[int]] = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            try:
                shapes[vi.name] = _shape(vi)
            except ValueError:
                pass
    for init in g.initializer:
        shapes[init.name] = list(init.dims)
    return g, shapes


def node_flops(node, sh) -> int:
    """FLOPs of one Conv/Gemm/MatMul node (2 per MAC); 0 for every other op."""
    if node.op_type == "Conv":
        out = sh(node.output[0])  # N, Cout, H, W
        w = sh(node.input[1])     # Cout, Cin/group, kh, kw
        return 2 * int(np.prod(out)) * int(np.prod(w[1:]))
    if node.op_type == "Gemm":
        a = sh(node.input[0])
        k = a[0] if _attr(node, "transA", 0) else a[1]
        return 2 * int(np.prod(sh(node.output[0]))) * k
    if node.op_type == "MatMul":
        return 2 * int(np.prod(sh(node.output[0]))) * sh(node.input[0])[-1]
    return 0


def graph_stats(model) -> GraphStats:
    g, shapes = static_shapes(model)

    def sh(name: str) -> List[int]:
        if name not in shapes:
            raise ValueError(f"no static shape for tensor '{name}'")
        return shapes[name]

    flops_by_op: Dict[str, int] = {}
    uncounted = set()
    for n in g.node:
        f = node_flops(n, sh)
        if n.op_type not in ("Conv", "Gemm", "MatMul"):
            uncounted.add(n.op_type)
        if f:
            flops_by_op[n.op_type] = flops_by_op.get(n.op_type, 0) + f

    # Initializers consumed only as the data input of Gather are read sparsely.
    # Initializers consumed only as QuantizeLinear's input are stored as 8-bit.
    gather_out: Dict[str, int] = {}
    quantized = set()
    other_use = set()
    for n in g.node:
        for idx, name in enumerate(n.input):
            if n.op_type == "Gather" and idx == 0:
                gather_out[name] = gather_out.get(name, 0) + int(np.prod(sh(n.output[0])))
            elif n.op_type == "QuantizeLinear" and idx == 0:
                quantized.add(name)
            else:
                other_use.add(name)

    weight_bytes = 0
    weight_count = 0
    for init in g.initializer:
        count = int(np.prod(init.dims)) if len(init.dims) else 1
        weight_count += count
        read, elem = count, _elem_bytes(init.data_type)
        sole_use = init.name not in other_use
        if init.name in gather_out and sole_use and init.name not in quantized:
            read = min(count, gather_out[init.name])
        if init.name in quantized and sole_use and init.name not in gather_out:
            elem = 1
        weight_bytes += read * elem

    init_names = {i.name for i in g.initializer}
    io_bytes = 0
    for vi in list(g.input) + list(g.output):
        if vi.name in init_names:
            continue
        io_bytes += int(np.prod(_shape(vi))) * _elem_bytes(vi.type.tensor_type.elem_type)

    return GraphStats(
        flops=sum(flops_by_op.values()),
        weight_bytes=weight_bytes,
        io_bytes=io_bytes,
        weight_count=weight_count,
        flops_by_op=flops_by_op,
        uncounted_ops=sorted(uncounted),
    )


def fold_constants(model, size_threshold: int = 16 << 20):
    """Constant-fold with onnx-graphsurgeon. Exporters leave shape arithmetic
    (Shape/Where/Expand chains) that hides static shapes from inference; folding
    removes it and gives every backend the same simpler graph. Outputs above
    `size_threshold` bytes are not folded, so a tied weight is not duplicated
    as a pre-transposed copy (GPT-2's lm_head)."""
    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(model)
    graph.fold_constants(size_threshold=size_threshold).cleanup().toposort()
    return gs.export_onnx(graph)


def file_stats(path: str) -> GraphStats:
    import onnx

    return graph_stats(onnx.load(path))
