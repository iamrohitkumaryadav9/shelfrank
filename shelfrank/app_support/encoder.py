"""Query encoding with the fp32 ONNX file of bge-small, downloaded once and verified."""

from pathlib import Path

import numpy as np

from shelfrank.data.download import ensure_file

PREFIX = "Represent this sentence for searching relevant passages: "
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BASE = f"https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{REVISION}"
FILES = {
    "model.onnx": (
        f"{BASE}/onnx/model.onnx",
        "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    ),
}
MAX_TOKENS = 128


class QueryEncoder:
    """First-token pooling and normalisation, which is how bge pools."""

    def __init__(self, model: Path, tokenizer: Path) -> None:
        """Load the graph and the tokenizer file."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(MAX_TOKENS)

    def __call__(self, query: str) -> np.ndarray:
        """Unit vector of a query."""
        encoded = self.tokenizer.encode(PREFIX + query)
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }
        hidden = self.session.run(None, feed)[0]
        vector = hidden[0, 0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)


def fetch_model(folder: Path) -> Path:
    """Download the graph if it is missing and check its hash."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in FILES.items():
        ensure_file(url, folder / name, sha256)
    return folder / "model.onnx"
