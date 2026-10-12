import copy
import json
import os
from pathlib import Path

import pytest

from trtbench.diff import diff_reports, main
from trtbench.results import ReportError, load_report, validate_report


def _result(name="copy_256MiB", p50=2.0, p90=2.02, p99=2.05, trustworthy=True,
            mode="latency", precision="fp32"):
    pct = {"p50": p50, "p90": p90, "p99": p99, "min": p50 * 0.99, "max": p99, "n": 300}
    return {
        "name": name,
        "precision": precision,
        "config": {"mode": mode},
        "device_ms": pct,
        "host_enqueue_ms": {"p50": 0.005, "p90": 0.006, "p99": 0.008, "min": 0.004,
                            "max": 0.01, "n": 300},
        "first_call_device_ms": 3.0,
        "clocks": {},
        "roofline": {},
        "checksum": {"verifiable": True, "passed": True},
        "findings": [],
        "trustworthy": trustworthy,
    }


def _report(*results, gpu="NVIDIA L4", driver="560.35.03"):
    return {
        "schema_version": 1,
        "environment": {"gpu_name": gpu, "driver_version": driver,
                        "cuda_runtime_version": 12080, "ecc_enabled": 1},
        "peaks": {},
        "calibration": [],
        "results": list(results),
        "all_trustworthy": all(r["trustworthy"] for r in results),
    }


def test_clean_speedup_is_reported():
    o = diff_reports(_report(_result(p50=2.0, p90=2.02)), _report(_result(p50=1.0, p90=1.01)))
    assert o.clean
    (row,) = o.rows
    assert row.status == "ok"
    assert row.speedup_p50 == pytest.approx(2.0)
    assert row.within_spread is False


def test_change_inside_spread_is_labelled():
    # 1% faster, but each run's own p50->p90 spread is 5%.
    o = diff_reports(_report(_result(p50=2.0, p90=2.1)), _report(_result(p50=1.98, p90=2.08)))
    assert o.rows[0].within_spread is True


def test_untrustworthy_side_gives_no_number():
    o = diff_reports(_report(_result()), _report(_result(trustworthy=False)))
    row = o.rows[0]
    assert row.status == "untrustworthy"
    assert row.speedup_p50 is None
    assert not o.clean


def test_precision_change_is_refused():
    o = diff_reports(_report(_result()), _report(_result(precision="fp16", p50=1.0)))
    assert o.rows[0].status == "refused"
    assert "accuracy" in o.rows[0].reason
    assert o.rows[0].speedup_p50 is None


def test_mode_change_is_refused():
    o = diff_reports(_report(_result()), _report(_result(mode="throughput")))
    assert o.rows[0].status == "refused"


def test_different_gpu_is_refused_unless_allowed():
    a = _report(_result())
    b = _report(_result(p50=1.0, p90=1.01), gpu="NVIDIA A10G")
    o = diff_reports(a, b)
    assert o.env_mismatches and o.rows[0].status == "refused"
    o = diff_reports(a, b, allow_env_mismatch=True)
    assert o.rows[0].status == "ok"
    assert all("(allowed)" in m for m in o.env_mismatches)


def test_ops_present_on_one_side_are_listed():
    o = diff_reports(_report(_result("a"), _result("b")), _report(_result("b"), _result("c")))
    assert o.only_in_base == ["a"]
    assert o.only_in_cand == ["c"]
    assert [r.name for r in o.rows] == ["b"]


def test_validation_rejects_missing_keys_and_wrong_schema():
    bad = _report(_result())
    del bad["results"][0]["findings"]
    with pytest.raises(ReportError):
        validate_report(bad)
    bad = _report(_result())
    bad["schema_version"] = 2
    with pytest.raises(ReportError):
        validate_report(bad)
    bad = _report(_result("x"), _result("x"))
    with pytest.raises(ReportError):
        validate_report(bad)


def test_cli_exit_codes(tmp_path: Path):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(_report(_result())))
    b.write_text(json.dumps(_report(_result(p50=1.5, p90=1.51))))
    c.write_text(json.dumps(_report(_result(trustworthy=False))))
    assert main([str(a), str(b)]) == 0
    assert main([str(a), str(c)]) == 3
    assert main([str(a), str(tmp_path / "missing.json")]) == 2


SAMPLE = os.environ.get("TRTBENCH_SAMPLE_JSON")


@pytest.mark.skipif(not SAMPLE, reason="TRTBENCH_SAMPLE_JSON not set (C++ tests write it)")
def test_cpp_writer_matches_python_reader():
    """The report the C++ harness writes must load and diff in Python."""
    rep = load_report(SAMPLE)
    assert rep["all_trustworthy"] is False  # the fixture includes an impossible result
    o = diff_reports(rep, copy.deepcopy(rep))
    by_name = {r.name: r for r in o.rows}
    assert by_name["copy_256MiB"].status == "ok"
    assert by_name["copy_256MiB"].speedup_p50 == pytest.approx(1.0)
    assert by_name["copy_impossible"].status == "untrustworthy"
