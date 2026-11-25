"""Query encoding with ONNX, one query at a time, as the demo does."""

from pathlib import Path
from typing import Any

import numpy as np

INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class OnnxEncoder:
    """An exported bi-encoder. The graph already pools and normalises."""

    def __init__(self, path: Path, tokenizer: Any, limit: int) -> None:  # noqa: ANN401
        """Load the graph once."""
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer
        self.limit = limit

    def __call__(self, text: str) -> np.ndarray:
        """Unit vector of one text."""
        encoded = self.tokenizer(
            [text], truncation=True, max_length=self.limit, return_tensors="np"
        )
        feed = {name: encoded[name].astype(np.int64) for name in INPUTS}
        vector = self.session.run(["embedding"], feed)[0][0]
        return vector / np.linalg.norm(vector)
