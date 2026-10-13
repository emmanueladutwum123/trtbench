import copy
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from trtbench import attribution as at  # noqa: E402

FP32 = "Row major linear FP32"
FP16 = "Channel major FP16 format where channel % 8 == 0"


def _model():
    """x[1,4,8,8] -> Conv(3x3, 8 out) -> Relu -> GlobalAveragePool -> Flatten -> Gemm(8->10)."""
    rng = np.random.default_rng(0)
    init = [numpy_helper.from_array(rng.standard_normal((8, 4, 3, 3)).astype(np.float32), "w"),
            numpy_helper.from_array(rng.standard_normal((8, 10)).astype(np.float32), "fc_w")]
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("GlobalAveragePool", ["r"], ["p"], name="pool"),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "fc_w"], ["y"], name="fc"),
    ]
    g = helper.make_graph(nodes, "net", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])], init)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _t(name, dims, fmt=FP32):
    return {"Name": name, "Dimensions": dims, "Format/Datatype": fmt}


def _profile(sm=(500, 500)):
    layers = [
        {"Name": "conv + relu", "LayerType": "CaskConvolution",
         "Metadata": "[ONNX Layer: conv]\x1e[ONNX Layer: relu]",
         "Inputs": [_t("x", [1, 4, 8, 8])], "Outputs": [_t("r", [1, 8, 8, 8], FP16)]},
        {"Name": "Reformatting CopyNode for Input Tensor 0 to pool", "LayerType": "Reformat",
         "Inputs": [_t("r", [1, 8, 8, 8], FP16)], "Outputs": [_t("r2", [1, 8, 8, 8])]},
        {"Name": "pool", "LayerType": "Reduce", "Metadata": "[ONNX Layer: pool]",
         "Inputs": [_t("r2", [1, 8, 8, 8])], "Outputs": [_t("p", [1, 8, 1, 1])]},
        {"Name": "flatten + fc", "LayerType": "CaskGemm",  # no metadata: mapped by name
         "Inputs": [_t("p", [1, 8, 1, 1])], "Outputs": [_t("y", [1, 10])]},
    ]
    times = {"conv + relu": 0.010, "Reformatting CopyNode for Input Tensor 0 to pool": 0.002,
             "pool": 0.004, "flatten + fc": 0.003, "__myl_unknown_kernel": 0.001}
    return {
        "name": "tensorrt/toy_b1/fp32", "meta_path": "/w/toy/b1/fp32.meta.json",
        "meta": {"model": "toy", "batch": 1, "precision_label": "fp32", "onnx": "fp32.onnx"},
        "layer_profile": {"runs": 200, "consistent": True, "inconsistency": "",
                          "layers": [{"name": k, "median_ms": v, "p90_ms": v} for k, v in times.items()]},
        "end_to_end_ms": {"without_profiler": {"p50": 0.015}, "with_profiler": {"p50": 0.030}},
        "clocks": {"available": True, "sm_mhz": {"min": sm[0], "median": sm[0], "max": sm[1]},
                   "sm_drift": (sm[1] - sm[0]) / sm[1], "locked": True, "locked_mhz": 500},
        "peaks": {"dram_gbps_measured": 100.0, "fp32_gflops": 1000.0, "fp16_tc_gflops": 8000.0,
                  "int8_tc_gops": 16000.0, "peak_sm_clock_mhz": 1000},
        "engine_information": {"Layers": layers},
    }


def test_dtype_parsing_of_inspector_formats():
    assert at.dtype_bytes(FP32) == 4
    assert at.dtype_bytes(FP16) == 2
    assert at.dtype_bytes("Thirty-two wide channel vectorized row major Int8 format") == 1
    assert at.dtype_bytes("Row major linear Int32") == 4
    assert at.dtype_bytes("something new") is None


def test_metadata_names_may_contain_brackets():
    layer = {"Metadata": "[ONNX Layer: /a/Gather[0]]\x1e[ONNX Layer: /a/Add]"}
    nodes, mode = at.onnx_nodes_of(layer, set())
    assert (nodes, mode) == (["/a/Gather[0]", "/a/Add"], "metadata")


def test_layer_bytes_flops_and_mapping():
    a = at.analyze(_profile(), _model())
    rows = {r["name"]: r for r in a["layers"]}
    conv = rows["conv + relu"]
    assert conv["onnx_nodes"] == ["conv", "relu"]
    assert conv["flops"] == 2 * (8 * 8 * 8) * (4 * 3 * 3)
    # input 1x4x8x8 FP32 + output 1x8x8x8 FP16 + weights 8x4x3x3 FP32
    assert conv["bytes"] == 256 * 4 + 512 * 2 + 288 * 4
    assert rows["flatten + fc"]["mapping"] == "layer name"
    assert rows["Reformatting CopyNode for Input Tensor 0 to pool"]["mapping"] == "unmapped"
    assert rows["Reformatting CopyNode for Input Tensor 0 to pool"]["bound"] == "memory"
    # A profiled kernel the inspector does not list is reported, not guessed at.
    assert rows["__myl_unknown_kernel"]["mapping"] == "not in inspector"
    assert rows["__myl_unknown_kernel"]["bound"] == "unknown"
    assert a["mapping"] == {"metadata": 2, "unmapped": 1, "layer name": 1, "not in inspector": 1}


def test_fusion_counts_and_largest_unfused_op():
    f = at.analyze(_profile(), _model())["fusion"]
    assert f["onnx_compute_nodes"] == 5
    assert f["onnx_nodes_in_engine_layers"] == 5
    assert f["nodes_in_fused_layers"] == 4  # conv+relu, flatten+fc
    assert f["single_node_layers"] == 1
    assert f["fusion_rate"] == pytest.approx(0.8)
    assert f["largest_unfused"]["op"] == "GlobalAveragePool"


def test_compute_peak_is_scaled_to_the_profiled_clock():
    pk = at.analyze(_profile(), _model())["peaks_at_run_clock"]
    assert pk["fp32"] == pytest.approx(500.0)  # 1000 GFLOP/s at 1000 MHz, run at 500
    assert pk["dram_gbps"] == 100.0


PK = {"dram_gbps": 100.0, "fp32": 1000.0, "fp16": 8000.0, "int8": 16000.0}


def test_classify_memory_compute_and_l2():
    # 1 MB at 100 GB/s takes 0.01 ms; no FLOPs -> memory-bound, 50% of roofline at 0.02 ms.
    m = at.classify(0.02, 0, 1_000_000, "fp32", PK)
    assert m["bound"] == "memory"
    assert m["pct_of_roofline"] == pytest.approx(50.0)
    assert m["headroom_ms"] == pytest.approx(0.01)
    # 1 GFLOP over 1 kB: compute-bound; 1 ms ideal at 1000 GFLOP/s.
    c = at.classify(2.0, 10**9, 1000, "fp32", PK)
    assert c["bound"] == "compute"
    assert c["attainable_ms"] == pytest.approx(1.0)
    # Faster than DRAM allows: data was in L2; no headroom is claimed.
    l2 = at.classify(0.005, 0, 1_000_000, "fp32", PK)
    assert l2["bound"] == "above DRAM roofline"
    assert l2["headroom_ms"] is None
    assert at.classify(1.0, 0, None, "fp32", PK)["bound"] == "unknown"


def test_classify_marks_layers_near_the_ridge():
    # ridge (fp32) = 1000 / 100 = 10 FLOP/B; intensity 12 is within 2x of it.
    r = at.classify(1.0, 12_000_000, 1_000_000, "fp32", PK)
    assert r["bound"] == "compute (near ridge)"


def test_int8_layer_is_judged_against_int8_peak():
    r = at.classify(1.0, 10**9, 1000, "int8", PK)
    assert r["attainable_ms"] == pytest.approx(10**9 / 16000e9 * 1e3)


def test_drifting_clock_rejects_the_engine_and_hides_its_numbers():
    a = at.analyze(_profile(sm=(450, 500)), _model())
    assert not a["trustworthy"]
    md = at.render([a])
    assert "**rejected**: clock_drift" in md
    assert "top 10 layers" not in md


def test_inconsistent_layer_list_rejects():
    p = _profile()
    p["layer_profile"]["consistent"] = False
    p["layer_profile"]["inconsistency"] = "run 3 reported 4 layers"
    assert not at.analyze(p, _model())["trustworthy"]


def test_render_lists_top_layers_and_round_trips_json():
    a = at.analyze(_profile(), _model())
    a = json.loads(json.dumps(a))  # render works from the committed JSON alone
    md = at.render([a], top=3)
    assert "| tensorrt/toy_b1/fp32 | 500–500 |" in md
    assert md.count("| conv + relu |") == 1
    assert "top 3 layers" in md


def test_quantized_weight_counts_one_byte_and_qdq_is_not_compute():
    w = numpy_helper.from_array(np.ones((4, 4), np.float32), "w")
    s = numpy_helper.from_array(np.array(0.1, np.float32), "s")
    z = numpy_helper.from_array(np.array(0, np.int8), "z")
    nodes = [helper.make_node("QuantizeLinear", ["w", "s", "z"], ["wq"], name="q"),
             helper.make_node("DequantizeLinear", ["wq", "s", "z"], ["wd"], name="dq"),
             helper.make_node("MatMul", ["x", "wd"], ["y"], name="mm")]
    g = helper.make_graph(nodes, "q", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])], [w, s, z])
    idx = at.OnnxIndex(helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)]))
    assert idx.compute_nodes() == ["mm"]
    # 16 weights at 1 byte, plus the scale (4 B) and zero point (1 B) behind both Q and DQ
    assert idx.weight_bytes["mm"] == 16 + 4 + 1


def test_gather_counts_only_the_rows_it_reads():
    table = numpy_helper.from_array(np.zeros((1000, 8), np.float32), "emb")
    nodes = [helper.make_node("Gather", ["emb", "ids"], ["e"], name="g")]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 5])],
                          [helper.make_tensor_value_info("e", TensorProto.FLOAT, [1, 5, 8])], [table])
    idx = at.OnnxIndex(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]))
    assert idx.weight_bytes["g"] == 5 * 8 * 4


def test_duplicate_profile_names_map_to_their_inspector_layer():
    p = copy.deepcopy(_profile())
    p["layer_profile"]["layers"].append({"name": "pool#2", "median_ms": 0.001, "p90_ms": 0.001})
    rows = {r["name"]: r for r in at.analyze(p, _model())["layers"]}
    assert rows["pool#2"]["onnx_nodes"] == ["pool"]
