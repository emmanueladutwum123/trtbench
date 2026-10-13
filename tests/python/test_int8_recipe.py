import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from trtbench.int8_recipe import smooth, transformer_excludes  # noqa: E402

H, F = 8, 32


def _block(seed=0):
    """LayerNorm -> up (H->F) -> Relu -> down (F->H) -> residual Add, plus an
    activation x activation MatMul, with one channel carrying a 50x outlier."""
    rng = np.random.default_rng(seed)
    init = [
        numpy_helper.from_array(np.ones(H, np.float32), "g"),
        numpy_helper.from_array(np.zeros(H, np.float32), "b"),
        numpy_helper.from_array(rng.standard_normal((H, F)).astype(np.float32), "w_up"),
        numpy_helper.from_array(rng.standard_normal((F, H)).astype(np.float32), "w_down"),
    ]
    nodes = [
        helper.make_node("LayerNormalization", ["x", "g", "b"], ["ln"], name="ln"),
        helper.make_node("MatMul", ["ln", "w_up"], ["up"], name="up"),
        helper.make_node("Relu", ["up"], ["act"], name="relu"),
        helper.make_node("MatMul", ["act", "w_down"], ["down"], name="down"),
        helper.make_node("Add", ["ln", "down"], ["res"], name="residual"),
        helper.make_node("Transpose", ["res"], ["resT"], perm=[0, 2, 1], name="t"),
        helper.make_node("MatMul", ["res", "resT"], ["y"], name="bmm"),
    ]
    g = helper.make_graph(nodes, "block",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, H])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4])], init)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])


def _feeds(n=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.standard_normal((1, 4, H)).astype(np.float32)
        x[..., 3] *= 50.0  # an outlier channel, as in BERT's LayerNorm outputs
        out.append({"x": x})
    return out


def _run(model, feeds):
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return [s.run(None, f)[0] for f in feeds]


def test_smoothing_is_exact_and_flattens_the_outlier_channel():
    base = _block()
    smoothed = _block()
    assert smooth(smoothed, _feeds()) == 2  # inputs of `up` and `down`
    for a, b in zip(_run(base, _feeds(seed=7)), _run(smoothed, _feeds(seed=7))):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    # The tensor `up` now quantizes is ln / s: its channel ranges are closer together.
    from trtbench.int8_recipe import activation_absmax

    before = activation_absmax(base, _feeds(), ["ln"])["ln"]
    after = activation_absmax(smoothed, _feeds(), ["ln__smoothed"])["ln__smoothed"]
    assert after.max() / after.min() < before.max() / before.min()


def test_residual_branch_still_sees_the_unsmoothed_tensor():
    m = _block()
    smooth(m, _feeds())
    add = next(n for n in m.graph.node if n.name == "residual")
    assert add.input[0] == "ln"


def test_excludes_down_projections_and_activation_matmuls():
    assert sorted(transformer_excludes(_block())) == ["bmm", "down"]
