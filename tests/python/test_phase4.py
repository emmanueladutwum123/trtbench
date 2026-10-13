import numpy as np
import pytest

from trtbench import phase4 as p4


def test_generator_matches_the_cpp_pins():
    # Same constants as tests/cpp/test_ln_reference.cpp (Inputs.GeneratorIsPinned).
    idx = np.array([0, 1], dtype=np.uint64)
    assert [hex(v) for v in p4.hash24(idx, 1)] == ["0x514e28", "0xa64b3d"]
    assert hex(p4.hash24(np.array([2], dtype=np.uint64), 2)[0]) == "0x45e53a"
    assert hex(p4.hash24(np.array([12345], dtype=np.uint64), 3)[0]) == "0x8848f4"


def test_inputs_are_representable_and_in_range():
    inp = p4.make_inputs(4, 256, "fp16")
    assert np.array_equal(inp["x"], inp["x"].astype(np.float16).astype(np.float32))
    assert inp["gamma"].min() >= 0.5 and inp["gamma"].max() <= 1.5
    assert inp["x"].dtype == np.float32


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_correctly_rounded_output_passes_and_two_ulps_fail(dtype):
    inp = p4.make_inputs(8, 768, dtype)
    s, y = p4.reference(inp, dtype)
    y_out = p4.round_to(dtype, y.astype(np.float32))
    assert p4.check(inp, dtype, s, y_out)["passed"]
    if dtype == "fp16":
        bad = y_out.copy()
        h = bad.astype(np.float16)
        h.view(np.uint16)[3, 100] += 2
        bad = h.astype(np.float32)
        assert not p4.check(inp, dtype, s, bad)["passed"]


def test_sum_must_be_bit_exact():
    inp = p4.make_inputs(2, 256, "fp32")
    s, y = p4.reference(inp, "fp32")
    s2 = s.copy()
    s2[0, 3] = np.nextafter(s2[0, 3], np.float32(100))
    r = p4.check(inp, "fp32", s2, y.astype(np.float32))
    assert not r["passed"] and "1 of 512 not bit-exact" in r["note"]


def test_tolerance_matches_the_cpp_formula():
    # tests/cpp: tolerance(1.0, 768, F16) == 2^-11 + 2*768*2^-24*2
    t = p4.tolerance(np.array([1.0]), 768, "fp16")[0]
    assert t == pytest.approx(2.0 ** -11 + 2 * 768 * 2.0 ** -24 * 2, rel=1e-12)


def _result(kernel, p50, trust=True, gbps=100.0):
    return {"name": f"add_layernorm/{kernel}", "trustworthy": trust,
            "labels": {"kernel": kernel, "dtype": "fp16", "rows": "128", "hidden": "768"},
            "bytes_per_launch": 786432 + 3072,
            "device_ms": {"p50": p50, "p99": p50 * 1.1},
            "roofline": {"achieved_gbps": gbps, "pct_bw_peak": gbps / 2.3, "pct_bw_theoretical": gbps / 3.2},
            "clocks": {"available": True, "sm_mhz": {"min": 495, "max": 495}},
            "findings": [] if trust else [{"flag": "clock_drift", "severity": "invalid", "detail": ""}]}


def test_tables_speedup_vs_torch_and_rejections(monkeypatch):
    monkeypatch.setattr("trtbench.results.validate_report", lambda r: None)
    rep = {"peaks": {"dram_gbps_measured": 230.9, "dram_gbps_theoretical": 320.1},
           "results": [_result("warp", 0.010), _result("torch", 0.030),
                       _result("naive", 0.5, trust=False)]}
    md = p4.tables(rep)
    lines = md.splitlines()
    torch_i = next(i for i, l in enumerate(lines) if l.startswith("| torch |"))
    warp_i = next(i for i, l in enumerate(lines) if l.startswith("| warp |"))
    assert torch_i < warp_i
    assert "| 3.00× |" in lines[warp_i]
    assert "**rejected**: clock_drift" in md


def test_ncu_raw_csv_summary(tmp_path):
    csv_text = (
        '"ID","Kernel Name","gpu__time_duration.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",'
        '"smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"\n'
        '"","","usecond","%","ratio","ratio"\n'
        '"0","void add_ln_warp<__half, 3>(const __half *)","12.5","88.1","3.20","0.00"\n')
    path = tmp_path / "ncu.csv"
    path.write_text(csv_text)
    md = p4.ncu_table(p4.read_ncu_raw(path))
    assert "| void add_ln_warp<__half, 3> | 12.5 usecond | 88.1 % | long_scoreboard 3.20" in md
