import json
from pathlib import Path

from test_diff import _report, _result
from trtbench.report import environment_line, render, results_table

ENV = {"gpu_name": "Tesla T4", "compute_capability": "7.5", "sm_count": 40,
       "driver_version": "580.82.07", "cuda_runtime_version": 13000, "ecc_enabled": 1,
       "persistence_mode": 0, "power_limit_w": 70.0, "git_sha": "abc123",
       "timestamp_utc": "2026-10-13T00:27:02Z"}


def _full(r, **roof):
    r = dict(r)
    r["clocks"] = {"available": True, "locked": False, "sm_mhz": {"min": 1170, "max": 1260}}
    r["roofline"] = {"bound": "memory", "achieved_gbps": 231.2, "pct_bw_theoretical": 72.2,
                     "achieved_gflops": 0, "pct_compute_peak": 0, **roof}
    return r


def test_environment_line_states_everything_needed_to_reproduce():
    line = environment_line(ENV, clocks_locked=False)
    for s in ("Tesla T4", "sm_75", "driver 580.82.07", "CUDA runtime 13.0", "ECC on",
              "power limit 70 W", "clocks unlocked", "abc123"):
        assert s in line, s


def test_rejected_result_shows_reason_and_no_numbers():
    bad = _full(_result(name="fma", p50=1.74, trustworthy=False))
    bad["findings"] = [{"flag": "clock_drift", "severity": "invalid", "detail": ""}]
    table = results_table(_report(bad))
    row = [l for l in table.splitlines() if l.startswith("| fma")][0]
    assert "**rejected**: clock_drift" in row
    assert "1,740" not in row and "1740" not in row


def test_accepted_result_shows_numbers_and_warnings():
    ok = _full(_result(name="copy", p50=2.32223, p99=2.32485))
    ok["findings"] = [{"flag": "power_cap_throttle", "severity": "warning", "detail": ""}]
    table = results_table(_report(ok))
    assert "2,322.23" in table and "231.2 GB/s (72.2% of theoretical)" in table
    assert "accepted (power_cap_throttle)" in table


def test_committed_t4_results_render(tmp_path: Path):
    """The README tables come from these files; they must keep rendering."""
    root = Path(__file__).resolve().parents[2] / "results" / "Tesla_T4-20261013"
    for name in ("phase1_latency.json", "phase1_throughput.json", "self_test.json"):
        out = render(json.loads((root / name).read_text()))
        assert "Tesla T4" in out
