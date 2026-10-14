"""CPU deployment backends for a GLiNER span model: ONNX Runtime and OpenVINO.

Every backend is wrapped so that it plugs into GLiNER's own ONNX inference
path (``GLiNER(..., model=UniEncoderSpanORTModel(session))``). Pre-processing,
span decoding and scoring are therefore identical across backends; only the
forward pass differs.

OpenVINO compatibility
----------------------
DeBERTa-v2 in ``transformers`` wraps ``build_rpos`` in ``@torch.jit.script``.
The legacy ONNX exporter turns its ``if key_len != query_len`` branch into an
ONNX ``If`` node whose two branches return tensors of different rank, which
the OpenVINO CPU plugin refuses to compile ("If operation with dynamic rank").
In GLiNER's encoder, attention is always self-attention, so the branch is never
taken; :func:`traceable_deberta_rpos` replaces the helper during export only
with its always-taken branch. ``tests/test_deploy.py`` and the benchmark check
that the patched graph has no ``If`` node and the same outputs.
"""

from __future__ import annotations

import os
import shutil
import contextlib
from typing import TYPE_CHECKING, Any, Dict, List, Iterator, Optional
from pathlib import Path
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np

ONNX_INPUTS = ("input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask")


@contextlib.contextmanager
def traceable_deberta_rpos() -> Iterator[None]:
    """Temporarily replace DeBERTa-v2's scripted ``build_rpos`` with its self-attention branch."""
    import transformers.models.deberta_v2.modeling_deberta_v2 as dv2  # noqa: PLC0415

    original = dv2.build_rpos

    def build_rpos(query_layer, key_layer, relative_pos, position_buckets: int, max_relative_positions: int):  # noqa: ARG001
        return relative_pos

    dv2.build_rpos = build_rpos
    try:
        yield
    finally:
        dv2.build_rpos = original


def count_if_nodes(onnx_path: Path) -> int:
    import onnx  # noqa: PLC0415

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    return sum(node.op_type == "If" for node in graph.node)


def export_onnx(model: Any, out_dir: Path) -> Dict[str, Path]:
    """Export FP32 ONNX (OpenVINO-compatible) and ONNX Runtime dynamic INT8.

    The INT8 file comes from GLiNER's own ``export_to_onnx(quantize=True)``,
    i.e. ``onnxruntime.quantization.quantize_dynamic`` with QUInt8 weights.
    """
    out_dir = Path(out_dir)
    with traceable_deberta_rpos():
        paths = model.export_to_onnx(out_dir, quantize=True)
    return {"onnx_fp32": Path(paths["onnx_path"]), "onnx_int8_dynamic": Path(paths["quantized_path"])}


class _IO:
    def __init__(self, name: str):
        self.name = name


class OpenVINOSession:
    """Minimal ``onnxruntime.InferenceSession`` look-alike backed by an OpenVINO compiled model."""

    def __init__(self, compiled_model: Any):
        self.compiled = compiled_model
        self.request = compiled_model.create_infer_request()
        self._inputs = [_IO(p.get_any_name()) for p in compiled_model.inputs]
        self._outputs = [_IO(p.get_any_name()) for p in compiled_model.outputs]

    def get_inputs(self) -> List[_IO]:
        return self._inputs

    def get_outputs(self) -> List[_IO]:
        return self._outputs

    def run(self, output_names: Optional[List[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        result = self.request.infer(feed)
        return [result[self.compiled.output(o.name)] for o in self._outputs]


@dataclass
class OpenVINOOptions:
    num_threads: int
    performance_hint: str = "LATENCY"
    inference_precision: str = "f32"  # the ARM CPU plugin defaults to f16

    def as_config(self) -> Dict[str, str]:
        return {
            "INFERENCE_NUM_THREADS": str(self.num_threads),
            "PERFORMANCE_HINT": self.performance_hint,
            "INFERENCE_PRECISION_HINT": self.inference_precision,
        }


def convert_openvino(onnx_path: Path, out_dir: Path) -> Path:
    """ONNX -> OpenVINO IR, FP32 weights."""
    import openvino as ov  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / "model.xml"
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml), compress_to_fp16=False)
    return xml


def compress_weights_int8(fp32_xml: Path, out_dir: Path) -> Path:
    """NNCF weight-only INT8 (``nncf.compress_weights``, INT8_ASYM). Activations stay FP32; no calibration data."""
    import nncf  # noqa: PLC0415
    import openvino as ov  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(str(fp32_xml))
    compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    xml = out_dir / "model.xml"
    ov.save_model(compressed, str(xml), compress_to_fp16=False)
    return xml


def quantize_static_int8(
    fp32_xml: Path,
    calibration_feeds: List[Dict[str, np.ndarray]],
    out_dir: Path,
    ignored_patterns: Optional[List[str]] = None,
    ignored_types: Optional[List[str]] = None,
) -> Path:
    """NNCF post-training static quantization (``nncf.quantize``): INT8 weights and activations.

    ``model_type=TRANSFORMER`` applies NNCF's transformer-specific scheme
    (e.g. keeps some sensitive operations in floating point). Operations
    matching ``ignored_patterns`` / ``ignored_types`` stay in floating point.
    """
    import nncf  # noqa: PLC0415
    import openvino as ov  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(str(fp32_xml))
    ignored = None
    if ignored_patterns or ignored_types:
        ignored = nncf.IgnoredScope(patterns=ignored_patterns or [], types=ignored_types or [], validate=False)
    quantized = nncf.quantize(
        model,
        nncf.Dataset(calibration_feeds),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(calibration_feeds),
        ignored_scope=ignored,
    )
    xml = out_dir / "model.xml"
    ov.save_model(quantized, str(xml), compress_to_fp16=False)
    return xml


def load_onnxruntime(export_dir: Path, onnx_file: str, num_threads: int) -> Any:
    import onnxruntime as ort  # noqa: PLC0415

    from gliner import GLiNER  # noqa: PLC0415

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    return GLiNER.from_pretrained(
        str(export_dir), load_onnx_model=True, onnx_model_file=onnx_file, session_options=so, load_tokenizer=True
    )


def load_openvino(export_dir: Path, xml: Path, options: OpenVINOOptions) -> Any:
    """GLiNER instance whose forward pass runs on an OpenVINO compiled model."""
    import openvino as ov  # noqa: PLC0415

    model = load_onnxruntime(export_dir, "model.onnx", options.num_threads)
    compiled = ov.Core().compile_model(str(xml), "CPU", options.as_config())
    model.model = type(model.model)(OpenVINOSession(compiled))
    return model


def calibration_feeds(model: Any, examples: List[Dict], labels: List[str], batch_size: int = 1) -> List[Dict]:
    """Collate examples with GLiNER's own pre-processing into ONNX input feeds."""
    collator = model.create_collator()
    feeds = []
    for i in range(0, len(examples), batch_size):
        batch = model.collate_batch(examples[i : i + batch_size], labels, collator)
        feeds.append({k: batch[k].cpu().numpy() for k in ONNX_INPUTS})
    return feeds


def disk_size_mb(paths: List[Path]) -> float:
    total = 0
    for raw in paths:
        path = Path(raw)
        total += path.stat().st_size
        if path.suffix == ".xml" and path.with_suffix(".bin").exists():
            total += path.with_suffix(".bin").stat().st_size
    return total / 2**20


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
