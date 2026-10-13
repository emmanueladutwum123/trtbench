"""The C++ runner's sampling protocol, for backends that only run in Python.

PyTorch eager and ONNX Runtime are timed here with exactly the structure of
src/cuda/runner.cpp, and the raw samples are written as a "trtbench-samples-1"
document. Nothing is summarized or judged in Python: `trtbench --ingest`
audits these samples with the same rules, peaks and launch floor as the
TensorRT engines it times in the same process.

One sample:
    t0 = now; record(start); h0 = now; enqueue x N; h1 = now; record(stop)
    sync(stop); t2 = now
    device = elapsed(start, stop) / N, host = (h1 - h0) / N, wall = (t2 - t0) / N

The host figures include the Python interpreter, because that is what a
caller of these backends pays. They are reported, never mixed into device
time.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class RunConfig:
    mode: str = "latency"
    warmup_iters: int = 50
    warmup_min_ms: float = 250.0
    min_iters: int = 200
    min_time_ms: float = 1000.0
    max_iters: int = 100000
    launches_per_sample: int = 100
    flush_l2: bool = False
    clock_sample_every: int = 25
    clock_drift_tol: float = 0.05


class NvmlClocks:
    """Clock snapshots through NVML (nvidia-ml-py), matching nvml.cpp."""

    def __init__(self, pci_bus_id: Optional[str]):
        self.available = False
        self.source = "none"
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = (pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id) if pci_bus_id
                       else pynvml.nvmlDeviceGetHandleByIndex(0))
            self._reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
            self.available = True
            self.source = "nvml (pynvml)"
        except Exception as e:  # noqa: BLE001 - absence of NVML is a reportable state
            self.source = f"none ({type(e).__name__}: {e})"

    def sample(self, t_ms: float) -> Dict:
        nv, h = self._nv, self._h
        return {
            "t_ms": t_ms,
            "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "throttle_reasons": int(self._reasons(h)),
            "temp_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1e3,
        }


def torch_pci_bus_id() -> Optional[str]:
    import torch

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    if all(hasattr(p, a) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        return f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
    return None


def sanity_checksum(outputs: Dict[str, np.ndarray], refs: Dict[str, np.ndarray]) -> Dict:
    """Same rule as the C++ engine workload: finite, cosine >= 0.99 vs FP32."""
    notes, passed, value = [], True, 0.0
    for name, ref in refs.items():
        got = outputs[name].astype(np.float64).ravel()
        r = ref.astype(np.float64).ravel()
        finite = bool(np.isfinite(got).all())
        cos = float(got @ r / np.sqrt((got @ got) * (r @ r))) if finite else 0.0
        value += cos
        notes.append(f"{name}: cosine {cos:.6f} vs FP32 reference, max |diff| "
                     f"{float(np.abs(got - r).max()):.3g}{'' if finite else ', NON-FINITE'}")
        passed &= finite and cos >= 0.99
    return {"verifiable": bool(refs), "passed": bool(passed and refs), "value": value,
            "note": "; ".join(notes) + " (sanity check, not an accuracy measurement)"}


def run(name: str, labels: Dict[str, str], precision: str, enqueue: Callable[[], None],
        stream, cfg: RunConfig, bytes_moved: int, flops: int,
        checksum: Callable[[], Dict], clocks: NvmlClocks) -> Dict:
    """Time `enqueue` (which must only enqueue work on `stream`) and return the
    samples document for `trtbench --ingest`."""
    import torch

    launches = 1 if cfg.mode == "latency" else cfg.launches_per_sample
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    ns = time.perf_counter_ns

    def measure(n: int):
        t0 = ns()
        start.record(stream)
        h0 = ns()
        for _ in range(n):
            enqueue()
        h1 = ns()
        stop.record(stream)
        stop.synchronize()  # DELIBERATE: elapsed time is readable only after stop
        t2 = ns()
        dev = start.elapsed_time(stop)
        return dev / n, (h1 - h0) / 1e6 / n, (t2 - t0) / 1e6 / n

    stream.synchronize()  # DELIBERATE: setup work must not land in the first sample
    origin = ns()
    snap = (lambda: clocks.sample((ns() - origin) / 1e6)) if clocks.available else None
    pre = snap() if snap else {}

    first = measure(1)
    w0 = ns()
    i = 0
    while cfg.warmup_iters and (i < cfg.warmup_iters or (ns() - w0) / 1e6 < cfg.warmup_min_ms):
        measure(launches)
        i += 1

    steady: List[Dict] = [snap()] if snap else []
    dev, host, wall = [], [], []
    s0 = ns()
    for i in range(cfg.max_iters):
        if i >= cfg.min_iters and (ns() - s0) / 1e6 >= cfg.min_time_ms:
            break
        d, h, w = measure(launches)
        dev.append(d)
        host.append(h)
        wall.append(w)
        if snap and cfg.clock_sample_every and (i + 1) % cfg.clock_sample_every == 0:
            steady.append(snap())
    if snap:
        steady.append(snap())

    lock = os.environ.get("TRTBENCH_EXTERNAL_LOCK_MHZ")
    return {
        "schema": "trtbench-samples-1",
        "name": name,
        "labels": labels,
        "precision": precision,
        "config": asdict(cfg) | {"launches_per_sample": launches},
        "bytes": int(bytes_moved),
        "flops": int(flops),
        "raw": {"device_ms": dev, "host_enqueue_ms": host, "wall_ms": wall,
                "first_device_ms": first[0], "first_host_ms": first[1], "first_wall_ms": first[2]},
        "clocks": {"available": clocks.available, "source": clocks.source,
                   "pre_warmup": pre or {}, "steady": steady,
                   "locked": bool(lock),
                   "lock_status": (f"locked externally at {lock} MHz (by the calling script)"
                                   if lock else "not requested")},
        "checksum": checksum(),
    }
