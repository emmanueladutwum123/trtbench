"""Load and validate the JSON reports written by the C++ harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = 1

REQUIRED_REPORT_KEYS = {
    "schema_version",
    "environment",
    "peaks",
    "calibration",
    "results",
    "all_trustworthy",
}

REQUIRED_RESULT_KEYS = {
    "name",
    "precision",
    "config",
    "device_ms",
    "host_enqueue_ms",
    "first_call_device_ms",
    "clocks",
    "roofline",
    "checksum",
    "findings",
    "trustworthy",
}

REQUIRED_PERCENTILE_KEYS = {"p50", "p90", "p99", "min", "max", "n"}


class ReportError(ValueError):
    """A report is malformed or from an incompatible schema."""


def validate_report(report: Dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ReportError(f"report missing keys: {sorted(missing)}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ReportError(
            f"schema_version {report['schema_version']} != supported {SCHEMA_VERSION}"
        )
    names = set()
    for r in report["results"]:
        missing = REQUIRED_RESULT_KEYS - r.keys()
        if missing:
            raise ReportError(f"result {r.get('name', '?')!r} missing keys: {sorted(missing)}")
        for key in ("device_ms", "host_enqueue_ms"):
            pm = REQUIRED_PERCENTILE_KEYS - r[key].keys()
            if pm:
                raise ReportError(f"result {r['name']!r} {key} missing {sorted(pm)}")
        if r["name"] in names:
            raise ReportError(f"duplicate result name {r['name']!r}")
        names.add(r["name"])


def load_report(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    validate_report(report)
    return report


def results_by_name(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["name"]: r for r in report["results"]}
