"""Phase 4: the PyTorch baseline for the fused add + LayerNorm kernels, the
Nsight Compute summary, and the tables.

    python -m trtbench.phase4 baselines OUT [--rows 128,1024,8192,65536] [--hidden 768]
    python -m trtbench.phase4 tables phase4_latency.json [--ncu ncu_raw.csv]

The baseline is what a framework does without fusion: `torch.add` writes s,
then `F.layer_norm` reads it back. It is timed by pyrunner (the C++ runner's
protocol) and audited by `trtbench --ingest` with the kernels, against the
same calibration. Its outputs are checked with exactly the kernels' rule:
the inputs, reference and tolerance here are a NumPy port of
src/core/ln_reference.cpp, pinned to it by tests on both sides.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

EPS = 1e-5
_M = np.uint64(0xFFFFFFFF)


def hash24(index: np.ndarray, salt: int) -> np.ndarray:
    """ln::input_value's hash (murmur3 finalizer), top 24 bits."""
    h = (index.astype(np.uint64) * np.uint64(0x9E3779B1) + np.uint64(salt)) & _M
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & _M
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & _M
    h ^= h >> np.uint64(16)
    return (h >> np.uint64(8)).astype(np.uint32)


def input_value(index: np.ndarray, salt: int, lo: float, hi: float) -> np.ndarray:
    u = hash24(index, salt).astype(np.float32) * np.float32(1.0 / 16777216.0)
    return np.float32(lo) + np.float32(hi - lo) * u  # same float32 operations as C++


def round_to(dtype: str, a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32) if dtype == "fp16" else a.astype(np.float32)


def make_inputs(rows: int, hidden: int, dtype: str) -> Dict[str, np.ndarray]:
    i = np.arange(rows * hidden, dtype=np.uint64)
    j = np.arange(hidden, dtype=np.uint64)
    return {
        "x": round_to(dtype, input_value(i, 1, -2.0, 2.0)).reshape(rows, hidden),
        "r": round_to(dtype, input_value(i, 2, -2.0, 2.0)).reshape(rows, hidden),
        "gamma": round_to(dtype, input_value(j, 3, 0.5, 1.5)),
        "beta": round_to(dtype, input_value(j, 4, -0.5, 0.5)),
    }


def reference(inp: Dict[str, np.ndarray], dtype: str):
    """(s rounded to dtype, as float32; y in float64), as ln::reference_row."""
    s = round_to(dtype, inp["x"] + inp["r"])  # one float32 add, then rounding
    sd = s.astype(np.float64)
    mean = sd.mean(axis=1, keepdims=True)
    var = ((sd - mean) ** 2).mean(axis=1, keepdims=True)
    y = (sd - mean) / np.sqrt(var + np.float64(np.float32(EPS))) * inp["gamma"] + inp["beta"]
    return s, y


def tolerance(y_ref: np.ndarray, hidden: int, dtype: str) -> np.ndarray:
    """ln::tolerance: ulp_T(|y| + d) / 2 + d, d = 2 H u32 (1 + |y|)."""
    mant, min_exp = (10, -14) if dtype == "fp16" else (23, -126)
    a = np.abs(y_ref)
    d = 2.0 * hidden * 2.0 ** -24 * (1.0 + a)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(a + d))
    e = np.maximum(np.nan_to_num(e, neginf=min_exp), min_exp)
    return np.ldexp(1.0, (e - mant).astype(np.int64)) / 2.0 + d


def check(inp: Dict[str, np.ndarray], dtype: str, s: np.ndarray, y: np.ndarray) -> Dict:
    s_ref, y_ref = reference(inp, dtype)
    hidden = inp["x"].shape[1]
    mism = int((s.view(np.uint32) != s_ref.view(np.uint32)).sum())
    err = np.abs(y.astype(np.float64) - y_ref)
    ratio = err / tolerance(y_ref, hidden, dtype)
    finite = bool(np.isfinite(ratio).all())
    worst = float(np.nanmax(ratio)) if finite else float("inf")
    passed = mism == 0 and finite and worst < 1.0
    return {"verifiable": True, "passed": passed, "value": float(y.astype(np.float64).sum()),
            "note": f"s: {mism} of {s.size} not bit-exact; y: max |err| {float(np.nanmax(err)):.3g}, "
                    f"worst {worst:.2f} of tolerance, all {s.shape[0]} rows checked"}


def baselines(out: Path, rows_list: List[int], hidden: int, dtypes: List[str], mode: str) -> None:
    import torch
    import torch.nn.functional as F

    from .pyrunner import NvmlClocks, RunConfig, run, torch_pci_bus_id

    cfg = RunConfig(mode=mode)
    clocks = NvmlClocks(torch_pci_bus_id())
    stream = torch.cuda.Stream()
    for dtype in dtypes:
        tdt = torch.float16 if dtype == "fp16" else torch.float32
        for rows in rows_list:
            name = f"add_layernorm/torch/{dtype}/{rows}x{hidden}"
            path = out / f"torch_{dtype}_{rows}x{hidden}.json"
            if path.exists():
                continue
            inp = make_inputs(rows, hidden, dtype)
            dev = {k: torch.from_numpy(v).to("cuda", tdt) for k, v in inp.items()}
            s = torch.empty_like(dev["x"])
            holder = {}

            def enqueue():
                with torch.no_grad(), torch.cuda.stream(stream):
                    torch.add(dev["x"], dev["r"], out=s)
                    holder["y"] = F.layer_norm(s, (hidden,), dev["gamma"], dev["beta"], EPS)

            def checksum():
                stream.synchronize()
                return check(inp, dtype, s.float().cpu().numpy(), holder["y"].float().cpu().numpy())

            n = rows * hidden
            size = 2 if dtype == "fp16" else 4
            print(f"[phase4] {name}", flush=True)
            doc = run(name, {"backend": "torch-unfused", "kernel": "torch", "dtype": dtype,
                             "rows": str(rows), "hidden": str(hidden),
                             "torch_version": torch.__version__},
                      dtype, enqueue, stream, cfg, (4 * n + 2 * hidden) * size, 0, checksum, clocks)
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc))
            del dev, s, holder
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- Nsight Compute

NCU_METRICS = {
    "gpu__time_duration.sum": "Duration",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput % of peak",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "Memory throughput % of peak",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM throughput % of peak",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved occupancy %",
    "sm__maximum_warps_per_active_cycle_pct": "Theoretical occupancy %",
    "launch__registers_per_thread": "Registers / thread",
    "dram__bytes_read.sum": "DRAM read",
    "dram__bytes_write.sum": "DRAM write",
}
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def read_ncu_raw(path: Path) -> List[Dict[str, str]]:
    """Rows of `ncu --csv --page raw`: header row, units row, one row per kernel."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    start = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    head, units = rows[start], rows[start + 1]
    out = []
    for r in rows[start + 2:]:
        d = dict(zip(head, r))
        d["__units__"] = dict(zip(head, units))
        out.append(d)
    return out


def ncu_table(rows: List[Dict[str, str]]) -> str:
    present = [m for m in NCU_METRICS if any(m in r for r in rows)]
    head = ["Kernel"] + [NCU_METRICS[m] for m in present] + ["Top warp stalls (cycles per issue)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        units = r["__units__"]
        name = r.get("Kernel Name", "?").split("(")[0]
        cells = [f"{r[m]} {units.get(m, '')}".strip() for m in present]
        stalls = []
        for k, v in r.items():
            if k.startswith(STALL_PREFIX) and k.endswith(STALL_SUFFIX):
                try:
                    stalls.append((float(v.replace(",", "")), k[len(STALL_PREFIX):-len(STALL_SUFFIX)]))
                except ValueError:
                    pass
        top = ", ".join(f"{n} {v:.2f}" for v, n in sorted(stalls, reverse=True)[:3]) or "–"
        out.append("| " + " | ".join([name] + cells + [top]) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- tables

_KERNEL_ORDER = {"torch": 0, "naive": 1, "block": 2, "warp": 3}


def tables(report: Dict) -> str:
    from .results import validate_report

    validate_report(report)
    groups: Dict[tuple, List[Dict]] = {}
    for r in report["results"]:
        lb = r.get("labels", {})
        if "kernel" in lb:
            groups.setdefault((lb["dtype"], int(lb["hidden"]), int(lb["rows"])), []).append(r)
    peaks = report["peaks"]
    out = [f"DRAM reference: measured copy {peaks['dram_gbps_measured']:.1f} GB/s, theoretical "
           f"{peaks['dram_gbps_theoretical']:.1f} GB/s. Bytes = read x, r, γ, β once + write s, y once.",
           ""]
    for (dtype, hidden, rows) in sorted(groups, key=lambda k: (k[0] != "fp32", k[1], k[2])):
        rs = sorted(groups[(dtype, hidden, rows)], key=lambda r: _KERNEL_ORDER[r["labels"]["kernel"]])
        base = next((r for r in rs if r["labels"]["kernel"] == "torch" and r["trustworthy"]), None)
        mb = rs[0]["bytes_per_launch"] / 1e6
        out.append(f"#### {dtype}, {rows} × {hidden} ({mb:.2f} MB moved)\n")
        out.append("| Kernel | p50 (µs) | p99 (µs) | Achieved GB/s | % of measured copy | "
                   "% of theoretical | vs PyTorch unfused | SM clock (MHz) | Verdict |")
        out.append("|---|---:|---:|---:|---:|---:|---|---|---|")
        for r in rs:
            k = r["labels"]["kernel"]
            c = r["clocks"]
            clk = f"{c['sm_mhz']['min']}–{c['sm_mhz']['max']}" if c.get("available") else "?"
            if not r["trustworthy"]:
                why = ", ".join(f["flag"] for f in r["findings"] if f["severity"] == "invalid")
                out.append(f"| {k} | – | – | – | – | – | – | {clk} | **rejected**: {why} |")
                continue
            d, rf = r["device_ms"], r["roofline"]
            vs = "baseline" if r is base else (f"{base['device_ms']['p50'] / d['p50']:.2f}×"
                                               if base else "no trustworthy baseline")
            warn = [f["flag"] for f in r["findings"] if f["severity"] == "warning"]
            out.append(f"| {k} | {d['p50'] * 1e3:,.2f} | {d['p99'] * 1e3:,.2f} | {rf['achieved_gbps']:.1f} "
                       f"| {rf['pct_bw_peak']:.1f}% | {rf['pct_bw_theoretical']:.1f}% | {vs} | {clk} "
                       f"| accepted{(' (' + ', '.join(warn) + ')') if warn else ''} |")
        out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m trtbench.phase4")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("out", type=Path)
    b.add_argument("--rows", default="128,1024,8192,65536")
    b.add_argument("--hidden", type=int, default=768)
    b.add_argument("--dtypes", default="fp32,fp16")
    b.add_argument("--mode", default="latency")
    t = sub.add_parser("tables")
    t.add_argument("report", type=Path)
    t.add_argument("--ncu", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "baselines":
        baselines(a.out, [int(x) for x in a.rows.split(",")], a.hidden, a.dtypes.split(","), a.mode)
        return 0
    print(f"<!-- generated by python -m trtbench.phase4 tables {a.report}"
          f"{' --ncu ' + str(a.ncu) if a.ncu else ''} -->")
    print(tables(json.loads(a.report.read_text())))
    if a.ncu:
        print("#### Nsight Compute (one launch per kernel, largest shape)\n")
        print(ncu_table(read_ncu_raw(a.ncu)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
