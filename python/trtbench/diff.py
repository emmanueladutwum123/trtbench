"""Diff two harness reports, refusing comparisons that would not mean anything.

    python -m trtbench.diff baseline.json candidate.json [--allow-env-mismatch]

Exit status: 0 every shared op compared cleanly, 3 at least one comparison
was refused or involved an untrustworthy result, 2 usage / malformed input.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .results import ReportError, load_report, results_by_name

# Environment fields that must match for latencies to be comparable at all.
ENV_KEYS_THAT_MUST_MATCH = ("gpu_name", "driver_version", "cuda_runtime_version", "ecc_enabled")


@dataclass
class Row:
    name: str
    status: str  # ok | refused | untrustworthy
    reason: str = ""
    base_p50_ms: Optional[float] = None
    cand_p50_ms: Optional[float] = None
    speedup_p50: Optional[float] = None
    speedup_p99: Optional[float] = None
    within_spread: Optional[bool] = None


@dataclass
class DiffOutcome:
    env_mismatches: List[str] = field(default_factory=list)
    rows: List[Row] = field(default_factory=list)
    only_in_base: List[str] = field(default_factory=list)
    only_in_cand: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.env_mismatches and all(r.status == "ok" for r in self.rows)


def _spread(r: Dict[str, Any]) -> float:
    d = r["device_ms"]
    return (d["p90"] - d["p50"]) / d["p50"] if d["p50"] > 0 else float("inf")


def compare_results(base: Dict[str, Any], cand: Dict[str, Any]) -> Row:
    """Same rules as the C++ compare(): trust, mode and precision gate the number."""
    name = base["name"]
    if not base["trustworthy"] or not cand["trustworthy"]:
        side = "baseline" if not base["trustworthy"] else "candidate"
        return Row(name, "untrustworthy", f"{side} failed its own audit")
    if base["config"]["mode"] != cand["config"]["mode"]:
        return Row(name, "refused",
                   f"modes differ ({base['config']['mode']} vs {cand['config']['mode']})")
    if base["precision"] != cand["precision"]:
        return Row(name, "refused",
                   f"precisions differ ({base['precision']} vs {cand['precision']}) "
                   "and reports carry no accuracy delta")
    b, c = base["device_ms"], cand["device_ms"]
    if b["p50"] <= 0 or c["p50"] <= 0:
        return Row(name, "refused", "zero p50")
    speedup = b["p50"] / c["p50"]
    # A change smaller than either run's own p50->p90 spread is not evidence
    # of anything; say so next to the number instead of hiding it.
    within = abs(c["p50"] - b["p50"]) / b["p50"] <= max(_spread(base), _spread(cand))
    return Row(
        name,
        "ok",
        base_p50_ms=b["p50"],
        cand_p50_ms=c["p50"],
        speedup_p50=speedup,
        speedup_p99=(b["p99"] / c["p99"]) if c["p99"] > 0 else None,
        within_spread=within,
    )


def diff_reports(base: Dict[str, Any], cand: Dict[str, Any],
                 allow_env_mismatch: bool = False) -> DiffOutcome:
    out = DiffOutcome()
    be, ce = base["environment"], cand["environment"]
    for k in ENV_KEYS_THAT_MUST_MATCH:
        if be.get(k) != ce.get(k):
            out.env_mismatches.append(f"{k}: {be.get(k)!r} vs {ce.get(k)!r}")

    br, cr = results_by_name(base), results_by_name(cand)
    out.only_in_base = sorted(br.keys() - cr.keys())
    out.only_in_cand = sorted(cr.keys() - br.keys())
    for name in sorted(br.keys() & cr.keys()):
        if out.env_mismatches and not allow_env_mismatch:
            out.rows.append(Row(name, "refused", "environments differ"))
        else:
            out.rows.append(compare_results(br[name], cr[name]))
    if allow_env_mismatch:
        out.env_mismatches = [m + " (allowed)" for m in out.env_mismatches]
    return out


def _fmt(x: Optional[float], spec: str) -> str:
    return "-" if x is None else format(x, spec)


def render(o: DiffOutcome) -> str:
    lines = []
    for m in o.env_mismatches:
        lines.append(f"ENV MISMATCH  {m}")
    lines.append(f"{'op':<32} {'status':<14} {'base p50 us':>12} {'cand p50 us':>12} "
                 f"{'speedup':>8}  note")
    for r in o.rows:
        note = r.reason
        if r.within_spread:
            note = "within run-to-run spread"
        lines.append(
            f"{r.name:<32} {r.status:<14} "
            f"{_fmt(r.base_p50_ms and r.base_p50_ms * 1e3, '12.2f'):>12} "
            f"{_fmt(r.cand_p50_ms and r.cand_p50_ms * 1e3, '12.2f'):>12} "
            f"{_fmt(r.speedup_p50, '7.3f'):>8}x  {note}")
    for n in o.only_in_base:
        lines.append(f"{n:<32} only in baseline")
    for n in o.only_in_cand:
        lines.append(f"{n:<32} only in candidate")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="trtbench.diff", description=__doc__.splitlines()[0])
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--allow-env-mismatch", action="store_true",
                   help="compare across GPU/driver/runtime changes (the mismatch is still printed)")
    a = p.parse_args(argv)
    try:
        o = diff_reports(load_report(a.baseline), load_report(a.candidate), a.allow_env_mismatch)
    except (OSError, ReportError, ValueError) as e:
        print(f"trtbench.diff: {e}", file=sys.stderr)
        return 2
    print(render(o))
    clean = all(r.status == "ok" for r in o.rows) and (
        a.allow_env_mismatch or not o.env_mismatches)
    return 0 if clean else 3


if __name__ == "__main__":
    sys.exit(main())
