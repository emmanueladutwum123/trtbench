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

    from trtbench.report import phase2_tables

    p2 = root / "phase2-lock1005"
    t = phase2_tables(json.loads((p2 / "phase2_latency.json").read_text()),
                      json.loads((p2 / "accuracy.json").read_text()))
    assert t.count("**rejected**") == 31


def _p2(backend, precision, p50, trust=True, model="resnet50", batch=1):
    r = _full(_result(name=f"{backend}/{model}_b{batch}/{precision}", p50=p50, p90=p50 * 1.01,
                      p99=p50 * 1.02, precision=precision, trustworthy=trust))
    r["labels"] = {"backend": backend, "model": model, "batch": str(batch), "precision": precision}
    r["host_enqueue_ms"] = {"p50": 0.02, "p90": 0.02, "p99": 0.03, "min": 0.01, "max": 0.04, "n": 300}
    if not trust:
        r["findings"] = [{"flag": "clock_drift", "severity": "invalid", "detail": ""}]
    return r


def test_phase2_speedups_carry_accuracy_and_refuse_without_it():
    from trtbench.report import phase2_tables

    rep = _report(_p2("torch_eager", "fp32", 6.0), _p2("tensorrt", "fp32", 3.0),
                  _p2("tensorrt", "fp16", 1.5), _p2("tensorrt", "int8", 1.0),
                  _p2("tensorrt+graph", "fp16", 1.4))
    rep["results"][4]["name"] = "tensorrt/resnet50_b1/fp16+graph"
    acc = {"torch_eager/resnet50_b1/fp32": {"metric": "top1_pct", "value": 88.0, "n": 992},
           "tensorrt/resnet50_b1/fp32": {"metric": "top1_pct", "value": 88.0, "n": 992},
           "tensorrt/resnet50_b1/fp16": {"metric": "top1_pct", "value": 87.9, "n": 992}}
    t = phase2_tables(rep, acc)
    lines = {l.split("|")[1].strip() + "/" + l.split("|")[2].strip(): l
             for l in t.splitlines() if l.startswith("| ") and "Backend" not in l and "Model" not in l}
    assert "2.00×" in lines["tensorrt/fp32"]                       # same precision: no delta needed
    assert "4.00× (Δ top-1 % -0.1 pts)" in lines["tensorrt/fp16"]  # with its accuracy cost
    assert "refused" in lines["tensorrt/int8"]                     # no INT8 accuracy -> no speedup
    assert "1.400" in t and "-6.7%" in t                           # graph row and its delta


def test_phase2_rejected_rows_have_no_numbers():
    from trtbench.report import phase2_tables

    rep = _report(_p2("torch_eager", "fp32", 6.0), _p2("tensorrt", "fp16", 1.5, trust=False))
    t = phase2_tables(rep, {})
    row = [l for l in t.splitlines() if l.startswith("| tensorrt")][0]
    assert "**rejected**: clock_drift" in row and "1.500" not in row



def test_phase2_cli_runs_as_module(tmp_path: Path):
    # main() must run after every function it calls is defined: a guard placed
    # mid-file passed the import-based tests and failed only under `python -m`.
    import subprocess
    import sys

    rep = _report(_p2("torch_eager", "fp32", 6.0), _p2("tensorrt", "fp32", 3.0))
    rep["environment"] = ENV
    (tmp_path / "rep.json").write_text(json.dumps(rep))
    (tmp_path / "acc.json").write_text("{}")
    r = subprocess.run([sys.executable, "-m", "trtbench.report", "--phase2",
                        str(tmp_path / "rep.json"), str(tmp_path / "acc.json")],
                       cwd=Path(__file__).resolve().parents[2] / "python",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "#### resnet50, batch 1" in r.stdout and "2.00×" in r.stdout
