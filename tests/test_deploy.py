"""The export patch removes DeBERTa-v2's ``If`` nodes without changing outputs (tiny random model, no download)."""

import warnings

import numpy as np
import torch
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from transformers import DebertaV2Model, DebertaV2Config  # noqa: E402

from experiments.deploy import count_if_nodes, traceable_deberta_rpos  # noqa: E402


def _tiny_deberta():
    torch.manual_seed(0)
    cfg = DebertaV2Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        norm_rel_ebd="layer_norm",
        share_att_key=True,
    )
    return DebertaV2Model(cfg).eval()


def _export(model, path):
    ids = torch.randint(0, 100, (2, 9))
    mask = torch.ones_like(ids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (ids, mask),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}},
            opset_version=19,
            dynamo=False,
        )


def test_patch_removes_if_nodes_and_keeps_outputs(tmp_path):
    model = _tiny_deberta()
    _export(model, tmp_path / "orig.onnx")
    with traceable_deberta_rpos():
        _export(model, tmp_path / "patched.onnx")

    assert count_if_nodes(tmp_path / "orig.onnx") > 0
    assert count_if_nodes(tmp_path / "patched.onnx") == 0

    ids = np.random.default_rng(0).integers(0, 100, (3, 13)).astype(np.int64)  # shape differs from export
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    a = ort.InferenceSession(str(tmp_path / "orig.onnx")).run(None, feed)[0]
    b = ort.InferenceSession(str(tmp_path / "patched.onnx")).run(None, feed)[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(ids), torch.from_numpy(feed["attention_mask"])).last_hidden_state.numpy()
    assert np.allclose(a, b, atol=1e-6)
    assert np.allclose(ref, b, atol=1e-4)


def test_patch_is_restored():
    import transformers.models.deberta_v2.modeling_deberta_v2 as dv2  # noqa: PLC0415

    original = dv2.build_rpos
    with traceable_deberta_rpos():
        assert dv2.build_rpos is not original
    assert dv2.build_rpos is original
