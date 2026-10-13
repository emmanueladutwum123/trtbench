"""Exact FLOP and byte counts on graphs small enough to count by hand."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from trtbench.onnx_stats import graph_stats  # noqa: E402


def _model(nodes, inputs, outputs, inits):
    g = helper.make_graph(nodes, "t", inputs, outputs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])


def test_conv_gemm_matmul_flops_and_bytes():
    w_conv = numpy_helper.from_array(np.zeros((8, 3, 3, 3), np.float32), "wc")
    w_mm = numpy_helper.from_array(np.zeros((8 * 4 * 4, 10), np.float32), "wm")
    w_g = numpy_helper.from_array(np.zeros((10, 5), np.float32), "wg")
    shape = numpy_helper.from_array(np.array([1, 128], np.int64), "shape")
    nodes = [
        helper.make_node("Conv", ["x", "wc"], ["c"], pads=[1, 1, 1, 1]),  # 1x8x4x4
        helper.make_node("Reshape", ["c", "shape"], ["r"]),
        helper.make_node("MatMul", ["r", "wm"], ["m"]),                    # 1x10
        helper.make_node("Gemm", ["m", "wg"], ["y"]),                      # 1x5
    ]
    m = _model(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
               [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5])],
               [w_conv, w_mm, w_g, shape])
    st = graph_stats(m)
    assert st.flops_by_op["Conv"] == 2 * (1 * 8 * 4 * 4) * (3 * 3 * 3)
    assert st.flops_by_op["MatMul"] == 2 * 10 * 128
    assert st.flops_by_op["Gemm"] == 2 * 5 * 10
    weights = (8 * 3 * 3 * 3 + 128 * 10 + 10 * 5) * 4 + 2 * 8
    assert st.weight_bytes == weights
    assert st.io_bytes == (3 * 4 * 4 + 5) * 4
    assert st.bytes_lower_bound == weights + st.io_bytes


def test_gather_only_table_counts_rows_read_not_the_table():
    table = numpy_helper.from_array(np.zeros((50000, 16), np.float32), "emb")
    nodes = [helper.make_node("Gather", ["emb", "ids"], ["y"])]
    m = _model(nodes, [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 8])],
               [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16])], [table])
    st = graph_stats(m)
    assert st.weight_bytes == 8 * 16 * 4  # 8 rows, not 50,000
    assert st.weight_count == 50000 * 16


def test_table_also_used_densely_counts_in_full():
    # GPT-2 ties its embedding to the output projection: then the whole table is read.
    table = numpy_helper.from_array(np.zeros((100, 16), np.float32), "emb")
    nodes = [helper.make_node("Gather", ["emb", "ids"], ["h"]),
             helper.make_node("Transpose", ["emb"], ["embT"]),
             helper.make_node("MatMul", ["h", "embT"], ["y"])]
    m = _model(nodes, [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 8])],
               [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 100])], [table])
    assert graph_stats(m).weight_bytes == 100 * 16 * 4


def test_weights_feeding_quantizelinear_count_one_byte():
    w = numpy_helper.from_array(np.zeros((64, 32), np.float16), "w")
    scale = numpy_helper.from_array(np.array(0.1, np.float16), "s")
    zp = numpy_helper.from_array(np.array(0, np.int8), "z")
    nodes = [helper.make_node("QuantizeLinear", ["w", "s", "z"], ["wq"]),
             helper.make_node("DequantizeLinear", ["wq", "s", "z"], ["wd"]),
             helper.make_node("MatMul", ["x", "wd"], ["y"])]
    m = _model(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [4, 64])],
               [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [4, 32])], [w, scale, zp])
    st = graph_stats(m)
    assert st.weight_bytes == 64 * 32 * 1 + 2 + 1  # int8 weight, fp16 scale, int8 zero point
