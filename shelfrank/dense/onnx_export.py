"""Export a bi-encoder to an int8 ONNX file for CPU serving, and check it against PyTorch."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from shelfrank.dense.finetune import embed

INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OPSET = 14


class _Encoder(nn.Module):
    """Takes plain tensors and returns the finished vector, so serving needs no pooling code."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return embed(
            self.model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )


def export_fp32(model: nn.Module, tokenizer: Any, path: Path) -> None:  # noqa: ANN401
    """Trace with variable batch and sequence length."""
    model.to("cpu").eval()
    sample = tokenizer(["a lamp", "a red velvet sofa"], padding=True, return_tensors="pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {}
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        extra["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(
            _Encoder(model),
            tuple(sample[name] for name in INPUTS),
            str(path),
            input_names=list(INPUTS),
            output_names=["embedding"],
            dynamic_axes={
                **{name: {0: "batch", 1: "tokens"} for name in INPUTS},
                "embedding": {0: "batch"},
            },
            opset_version=OPSET,
            **extra,
        )


def quantize(source: Path, target: Path) -> None:
    """Dynamic int8 quantisation of the weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def onnx_encode(path: Path, tokenizer: Any, texts: Sequence[str], limit: int) -> np.ndarray:  # noqa: ANN401
    """Encode one text at a time, as serving does."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for text in texts:
        encoded = tokenizer([text], truncation=True, max_length=limit, return_tensors="np")
        feed = {name: encoded[name].astype(np.int64) for name in INPUTS}
        out.append(session.run(["embedding"], feed)[0][0])
    return np.stack(out)


def agreement(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Cosine similarity between two sets of unit vectors."""
    cosine = (reference * other).sum(axis=1) / np.linalg.norm(other, axis=1)
    return {"mean": float(cosine.mean()), "min": float(cosine.min())}
