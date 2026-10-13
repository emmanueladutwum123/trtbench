"""Phase 3: where an engine's time goes, and whether a faster kernel could help.

    python -m trtbench.attribution analyze PROFILE.json [...] --out attribution.json
    python -m trtbench.attribution render attribution.json        # markdown tables

Input per engine is what `trtbench --engine-profile` writes: IProfiler layer
times, end-to-end times with and without the profiler, clocks, device peaks
and the engine inspector's layer JSON; plus the ONNX graph the engine was
built from (found through the profile's meta). `analyze` needs the ONNX;
`render` needs only attribution.json, so committed results re-render anywhere.

For each engine layer:
  ONNX nodes   from the inspector's per-layer metadata ("[ONNX Layer: name]",
               written by TensorRT's ONNX parser, needs a kDETAILED build).
               Without metadata, from the layer name split on " + ". Never
               guessed: a layer with neither is "unmapped".
  FLOPs        Conv/Gemm/MatMul FLOPs of its ONNX nodes (onnx_stats).
               Elementwise work is not counted, as in Phase 2.
  bytes        a lower bound on what the layer must move: each input and
               output activation once, at the dimensions and data type the
               inspector reports (padding of vectorized formats ignored),
               plus the weights its ONNX nodes consume (1 byte when the
               weight goes through QuantizeLinear; gathered rows only for
               Gather). Not DRAM traffic: a small activation can stay in L2.
  bound        roofline at the clock the profile ran at: memory if
               bytes / DRAM peak >= FLOPs / compute peak, else compute. The
               compute peak is chosen by the layer's input data type (INT8 or
               FP16 tensor cores, FP32 CUDA cores). A layer faster than its
               bytes allow at the DRAM peak is "above DRAM roofline": its
               data came from L2, and the DRAM roofline says nothing about it.
  headroom     measured - max(bytes / DRAM peak, FLOPs / compute peak): the
               most a perfect kernel for the same fused layer could save.

Profiler timings are per layer with events between layers, so their sum
exceeds the unprofiled end-to-end time; both are reported, and shares are
shares of the summed layer time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DRIFT_TOL = 0.05  # same tolerance as the runner's clock audit
NEAR_RIDGE = 2.0  # within this factor of the ridge, the bound is called "near ridge"
QDQ = ("QuantizeLinear", "DequantizeLinear")
_ONNX_META = re.compile(r"\[ONNX Layer: (.*?)\](?=\x1e|$)", re.S)
_DTYPE_BYTES = (("fp32", 4), ("float", 4), ("fp16", 2), ("half", 2), ("bf16", 2),
                ("fp8", 1), ("uint8", 1), ("int8", 1), ("int32", 4), ("int64", 8), ("bool", 1))


# --------------------------------------------------------------------------- inspector

def inspector_layers(info: Any) -> List[Dict[str, Any]]:
    """The inspector's layer list as dicts. A kLAYER_NAMES_ONLY engine gives
    bare names; they become {"Name": name}."""
    if not isinstance(info, dict):
        return []
    out = []
    for layer in info.get("Layers", []):
        out.append(layer if isinstance(layer, dict) else {"Name": str(layer)})
    return out


def dtype_bytes(fmt: str) -> Optional[int]:
    f = fmt.lower()
    for token, n in _DTYPE_BYTES:
        if token in f:
            return n
    return None


def tensor_bytes(t: Dict[str, Any]) -> Optional[int]:
    dims, fmt = t.get("Dimensions"), t.get("Format/Datatype")
    if not isinstance(dims, list) or not fmt or any((not isinstance(d, int)) or d < 0 for d in dims):
        return None
    n = dtype_bytes(fmt)
    return None if n is None else int(np.prod(dims, dtype=np.int64)) * n


def onnx_nodes_of(layer: Dict[str, Any], node_names: set) -> Tuple[List[str], str]:
    meta = layer.get("Metadata")
    if isinstance(meta, str) and "[ONNX Layer:" in meta:
        return [m for m in _ONNX_META.findall(meta)], "metadata"
    parts = [p.strip() for p in str(layer.get("Name", "")).split(" + ")]
    hits = [p for p in parts if p in node_names]
    return (hits, "layer name") if hits else ([], "unmapped")


# --------------------------------------------------------------------------- ONNX side

class OnnxIndex:
    """What attribution needs from the ONNX graph, computed once."""

    def __init__(self, model):
        from onnx import numpy_helper  # noqa: F401  (keeps the import error local)

        from .onnx_stats import _elem_bytes, node_flops, static_shapes

        g, shapes = static_shapes(model)
        self.nodes = {n.name: n for n in g.node}
        inits = {i.name: i for i in g.initializer}
        producer = {o: n for n in g.node for o in n.output}

        const = set(inits)
        for n in g.node:  # topological order
            if n.op_type == "Constant" or (n.input and all(i in const or i == "" for i in n.input)):
                const.update(n.output)
        self.constant_tensors = const
        self.weight_prep = {n.name for n in g.node if n.output and all(o in const for o in n.output)}

        def sh(name):
            if name not in shapes:
                raise ValueError(f"no static shape for tensor '{name}'")
            return shapes[name]

        self.flops = {n.name: node_flops(n, sh) for n in g.node}

        # Weight bytes read by each node: walk back through constant-only
        # producers to the initializers.
        self.weight_bytes: Dict[str, int] = {}
        for n in g.node:
            if n.name in self.weight_prep:
                continue
            # Each initializer once per node. It is stored as 8-bit only if
            # every path to it passes through QuantizeLinear (a scale also
            # feeds DequantizeLinear directly and stays at its own type).
            reads: Dict[str, Tuple[bool, int]] = {}
            for idx, t in enumerate(n.input):
                if t not in const:
                    continue
                for init, quantized in _initializers_behind(t, producer, inits):
                    count = int(np.prod(inits[init].dims)) if len(inits[init].dims) else 1
                    if n.op_type == "Gather" and idx == 0:
                        count = min(count, int(np.prod(sh(n.output[0]))))
                    q0, c0 = reads.get(init, (True, 0))
                    reads[init] = (q0 and quantized, max(c0, count))
            self.weight_bytes[n.name] = sum(
                c * (1 if q else _elem_bytes(inits[k].data_type)) for k, (q, c) in reads.items())

    def compute_nodes(self) -> List[str]:
        """Nodes that do work at run time: not weight preparation, not Q/DQ, not Constant."""
        return [k for k, n in self.nodes.items()
                if k not in self.weight_prep and n.op_type not in QDQ and n.op_type != "Constant"]


def _initializers_behind(tensor, producer, inits):
    """(initializer, passes through QuantizeLinear) pairs feeding `tensor`."""
    out, stack, seen = [], [(tensor, False)], set()
    while stack:
        t, q = stack.pop()
        if (t, q) in seen:
            continue
        seen.add((t, q))
        if t in inits:
            out.append((t, q))
            continue
        n = producer.get(t)
        if n is None:
            continue
        for i in n.input:
            if i:
                stack.append((i, q or n.op_type == "QuantizeLinear"))
    return out


# --------------------------------------------------------------------------- analysis

def _peaks_at_clock(peaks: Dict[str, Any], sm_mhz: float) -> Dict[str, float]:
    scale = sm_mhz / peaks["peak_sm_clock_mhz"] if peaks.get("peak_sm_clock_mhz") and sm_mhz else 1.0
    bw = peaks.get("dram_gbps_measured") or peaks.get("dram_gbps_theoretical") or 0.0
    return {"dram_gbps": bw,
            "dram_source": "measured copy" if peaks.get("dram_gbps_measured") else "theoretical",
            "fp32": peaks.get("fp32_gflops", 0) * scale,
            "fp16": peaks.get("fp16_tc_gflops", 0) * scale,
            "int8": peaks.get("int8_tc_gops", 0) * scale}


def _layer_precision(layer: Dict[str, Any]) -> str:
    fmts = " ".join(str(t.get("Format/Datatype", "")) for t in layer.get("Inputs", [])).lower()
    if "int8" in fmts:
        return "int8"
    if "fp16" in fmts or "half" in fmts:
        return "fp16"
    return "fp32"


def classify(time_ms: float, flops: int, nbytes: Optional[int], prec: str,
             pk: Dict[str, float]) -> Dict[str, Any]:
    """Roofline placement of one layer. Pure, so it is unit-tested directly."""
    r: Dict[str, Any] = {"bound": "unknown", "attainable_ms": None, "headroom_ms": None,
                         "achieved_gbps": None, "achieved_gflops": None, "pct_of_roofline": None,
                         "intensity": None, "ridge": None}
    if nbytes is None or time_ms <= 0 or pk["dram_gbps"] <= 0:
        return r
    s = time_ms / 1e3
    r["achieved_gbps"] = nbytes / s / 1e9
    r["achieved_gflops"] = flops / s / 1e9
    peak_c = pk.get(prec, 0.0)
    t_mem = nbytes / (pk["dram_gbps"] * 1e9) * 1e3
    t_comp = flops / (peak_c * 1e9) * 1e3 if peak_c > 0 else 0.0
    if nbytes:
        r["intensity"] = flops / nbytes
    if peak_c > 0:
        r["ridge"] = peak_c / pk["dram_gbps"]
    if flops and peak_c <= 0:
        return r  # compute peak unknown: no honest classification
    attainable = max(t_mem, t_comp)
    if time_ms < t_mem:
        r["bound"] = "above DRAM roofline"
        return r
    if flops == 0 or t_mem >= t_comp:
        r["bound"] = "memory"
    else:
        r["bound"] = "compute"
    if flops and r["intensity"] is not None and r["ridge"]:
        if 1 / NEAR_RIDGE <= r["intensity"] / r["ridge"] <= NEAR_RIDGE:
            r["bound"] += " (near ridge)"
    r["attainable_ms"] = attainable
    r["headroom_ms"] = time_ms - attainable
    r["pct_of_roofline"] = 100 * attainable / time_ms
    return r


def analyze(profile: Dict[str, Any], model) -> Dict[str, Any]:
    idx = OnnxIndex(model)
    names = set(idx.nodes)
    layers_info = inspector_layers(profile.get("engine_information"))
    by_name = {}
    for layer in layers_info:
        by_name.setdefault(layer.get("Name"), layer)

    clocks = profile["clocks"]
    sm = clocks["sm_mhz"]
    pk = _peaks_at_clock(profile["peaks"], sm.get("median") or 0)
    findings = []
    if not profile["layer_profile"]["consistent"]:
        findings.append(("invalid", "inconsistent_layer_list", profile["layer_profile"]["inconsistency"]))
    if clocks.get("available") and clocks["sm_drift"] > DRIFT_TOL:
        findings.append(("invalid", "clock_drift",
                         f"SM {sm['min']}-{sm['max']} MHz, {100 * clocks['sm_drift']:.1f}% "
                         f"> {100 * DRIFT_TOL:.0f}%"))
    if not layers_info:
        findings.append(("warning", "no_inspector_layers",
                         "engine inspector returned no layer list (engine not built kDETAILED?)"))

    weight_inputs = idx.constant_tensors
    rows, mapping_modes = [], defaultdict(int)
    covered: Dict[str, int] = {}
    for prof in profile["layer_profile"]["layers"]:
        base = prof["name"].split("#")[0] if prof["name"] not in by_name else prof["name"]
        layer = by_name.get(base)
        row = {"name": prof["name"], "time_ms": prof["median_ms"], "p90_ms": prof["p90_ms"],
               "in_inspector": layer is not None}
        if layer is None:
            row.update(type="?", onnx_nodes=[], mapping="not in inspector", flops=0, bytes=None,
                       precision="?", **classify(0, 0, None, "fp32", pk))
            rows.append(row)
            mapping_modes["not in inspector"] += 1
            continue
        nodes, mode = onnx_nodes_of(layer, names)
        nodes = [n for n in nodes if n in names]
        mapping_modes[mode] += 1
        work = [n for n in nodes if n not in idx.weight_prep]
        for n in work:
            covered[n] = covered.get(n, 0) + 1
        flops = sum(idx.flops[n] for n in work)
        act = 0
        for t in layer.get("Inputs", []) + layer.get("Outputs", []):
            if t.get("Name") in weight_inputs:
                continue  # weights are counted once below, from the ONNX graph
            b = tensor_bytes(t)
            if b is None:
                act = None
                break
            act += b
        described = "Inputs" in layer or "Outputs" in layer  # absent without kDETAILED
        nbytes = None if act is None or not described \
            else act + sum(idx.weight_bytes.get(n, 0) for n in work)
        prec = _layer_precision(layer)
        row.update(type=layer.get("LayerType", "?"), onnx_nodes=work, mapping=mode, flops=flops,
                   bytes=nbytes, precision=prec, tactic=layer.get("TacticName"),
                   **classify(prof["median_ms"], flops, nbytes, prec, pk))
        rows.append(row)

    # Fusion: compute nodes per engine layer.
    compute = idx.compute_nodes()
    op_of = {k: n.op_type for k, n in idx.nodes.items()}
    fused = unfused = 0
    single: List[Dict[str, Any]] = []
    for row in rows:
        k = len([n for n in row["onnx_nodes"] if op_of.get(n) not in QDQ])
        if k >= 2:
            fused += k
        elif k == 1:
            unfused += 1
            single.append(row)
    mapped = sum(1 for n in compute if n in covered)
    largest_unfused = max(single, key=lambda r: r["time_ms"], default=None)

    total = sum(r["time_ms"] for r in rows)
    share = defaultdict(float)
    for r in rows:
        share[r["bound"].replace(" (near ridge)", "")] += r["time_ms"]
    headroom = sum(r["headroom_ms"] or 0 for r in rows)
    meta = profile["meta"]
    e2e = profile["end_to_end_ms"]
    return {
        "name": profile["name"],
        "model": meta["model"], "batch": meta["batch"], "precision": meta["precision_label"],
        "trt_version": meta.get("trt_version"),
        "environment": profile.get("environment"),
        "clocks": {"sm_mhz": sm, "sm_drift": clocks.get("sm_drift"), "locked": clocks.get("locked"),
                   "locked_mhz": clocks.get("locked_mhz"),
                   "throttle_reasons_seen": clocks.get("throttle_reasons_seen")},
        "peaks_at_run_clock": pk,
        "runs": profile["layer_profile"]["runs"],
        "trustworthy": not any(f[0] == "invalid" for f in findings),
        "findings": [{"severity": s, "flag": f, "detail": d} for s, f, d in findings],
        "end_to_end_ms_p50": e2e["without_profiler"]["p50"],
        "end_to_end_with_profiler_ms_p50": e2e["with_profiler"]["p50"],
        "sum_of_layers_ms": total,
        "mapping": dict(mapping_modes),
        "fusion": {
            "onnx_compute_nodes": len(compute),
            "onnx_nodes_in_engine_layers": mapped,
            "onnx_nodes_not_in_any_layer": len(compute) - mapped,
            "engine_layers": len(rows),
            "nodes_in_fused_layers": fused,
            "single_node_layers": unfused,
            "fusion_rate": fused / mapped if mapped else None,
            "largest_unfused": None if largest_unfused is None else {
                "layer": largest_unfused["name"],
                "op": op_of.get(next(n for n in largest_unfused["onnx_nodes"] if op_of.get(n) not in QDQ)),
                "time_ms": largest_unfused["time_ms"], "bound": largest_unfused["bound"]},
        },
        "time_share": {k: v / total for k, v in share.items()} if total else {},
        "roofline_headroom_ms": headroom,
        "layers": rows,
    }


# --------------------------------------------------------------------------- rendering

def _short(name: str, n: int = 60) -> str:
    name = name.replace("|", "\\|")
    return name if len(name) <= n else name[: n - 1] + "…"


def _num(x, fmt="{:.3f}") -> str:
    return "–" if x is None else fmt.format(x)


def render(doc: List[Dict[str, Any]], top: int = 10) -> str:
    out = ["#### Overview\n",
           "| Engine | SM MHz | End-to-end p50 (ms) | Σ layers (ms) | Engine layers | ONNX compute nodes "
           "| Fusion rate | Memory | Compute | Above DRAM roofline | Unclassified | Roofline headroom "
           "| Largest unfused op | Verdict |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for a in doc:
        sm = a["clocks"]["sm_mhz"]
        if not a["trustworthy"]:
            why = ", ".join(f["flag"] for f in a["findings"] if f["severity"] == "invalid")
            out.append(f"| {a['name']} | {sm['min']}–{sm['max']} | – | – | – | – | – | – | – | – | – "
                       f"| – | – | **rejected**: {why} |")
            continue
        fu, ts = a["fusion"], a["time_share"]
        lu = fu["largest_unfused"]
        lu_s = "–" if lu is None else f"{lu['op']} ({lu['time_ms']:.3f} ms, {lu['bound']})"
        pct = lambda k: f"{100 * ts.get(k, 0):.0f}%"  # noqa: E731
        warn = [f["flag"] for f in a["findings"] if f["severity"] == "warning"]
        out.append(
            f"| {a['name']} | {sm['min']}–{sm['max']} | {a['end_to_end_ms_p50']:.3f} "
            f"| {a['sum_of_layers_ms']:.3f} | {fu['engine_layers']} | {fu['onnx_compute_nodes']} "
            f"| {_num(fu['fusion_rate'] and 100 * fu['fusion_rate'], '{:.0f}%')} "
            f"| {pct('memory')} | {pct('compute')} | {pct('above DRAM roofline')} | {pct('unknown')} "
            f"| {a['roofline_headroom_ms']:.3f} ms ({100 * a['roofline_headroom_ms'] / a['sum_of_layers_ms']:.0f}%) "
            f"| {_short(lu_s, 50)} | accepted{(' (' + ', '.join(warn) + ')') if warn else ''} |")
    out.append("")
    for a in doc:
        if not a["trustworthy"]:
            continue
        total = a["sum_of_layers_ms"]
        out.append(f"#### {a['name']}: top {top} layers by time\n")
        out.append("| Layer | Type | ONNX nodes | Time p50 (ms) | Share | GFLOP | MB | Bound "
                   "| Achieved | % of roofline | Headroom (ms) |")
        out.append("|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
        for r in sorted(a["layers"], key=lambda r: -r["time_ms"])[:top]:
            ach = "–"
            if r["achieved_gbps"] is not None:
                ach = (f"{r['achieved_gflops']:,.0f} GFLOP/s" if r["bound"].startswith("compute")
                       else f"{r['achieved_gbps']:,.1f} GB/s")
            out.append(
                f"| {_short(r['name'])} | {r['type']} | {len(r['onnx_nodes'])} | {r['time_ms']:.4f} "
                f"| {100 * r['time_ms'] / total:.1f}% | {r['flops'] / 1e9:.3f} "
                f"| {_num(r['bytes'] and r['bytes'] / 1e6, '{:.2f}')} | {r['bound']} | {ach} "
                f"| {_num(r['pct_of_roofline'], '{:.0f}%')} | {_num(r['headroom_ms'], '{:.4f}')} |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    import onnx

    p = argparse.ArgumentParser(prog="python -m trtbench.attribution")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("profiles", nargs="+", type=Path)
    a.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("attribution", type=Path)
    r.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)

    if args.cmd == "analyze":
        doc = []
        for path in args.profiles:
            prof = json.loads(path.read_text())
            onnx_path = Path(prof["meta_path"]).parent / prof["meta"]["onnx"]
            print(f"[attribution] {prof['name']} <- {onnx_path}", flush=True)
            doc.append(analyze(prof, onnx.load(str(onnx_path))))
        args.out.write_text(json.dumps(doc, indent=1))
        return 0
    doc = json.loads(args.attribution.read_text())
    print(f"<!-- generated by python -m trtbench.attribution render {args.attribution} -->")
    print(render(doc, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
