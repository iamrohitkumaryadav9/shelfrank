"""Export both bge-small models to fp32 ONNX on this laptop, and check them against PyTorch.

The Colab package holds int8 files only. PyTorch writes the graph itself; the one step that needs the
``onnx`` wheel only inserts custom operators, which these models do not have, so that step is skipped
instead of installing another compiled package. The check against PyTorch is what makes this safe.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finetune_dense import REFERENCE_TEXTS
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.dense.finetune import encode
from shelfrank.dense.onnx_export import agreement, export_fp32, onnx_encode

FOLDERS = {"bge_small": "bge_small_base", "bge_small_esci": "bge_small_esci"}


def main() -> None:
    """Write artifacts/models/<folder>/model_fp32.onnx and refuse a graph that differs."""
    models = load_config("retrieval")["dense"]
    for name, folder in FOLDERS.items():
        cfg = models[name]
        local = cfg["revision"] == "local"
        source = str(REPO_ROOT / cfg["model"]) if local else cfg["model"]
        options = {} if local else {"revision": cfg["revision"]}
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        model = AutoModel.from_pretrained(source, add_pooling_layer=False, **options)
        target = REPO_ROOT / "artifacts/models" / folder / "model_fp32.onnx"
        place = "torch.onnx._internal.onnx_proto_utils._add_onnxscript_fn"
        with patch(place, lambda graph, opsets: graph):
            export_fp32(model, tokenizer, target)
        texts = [cfg["query_prefix"] + t for t in REFERENCE_TEXTS]
        ours = onnx_encode(target, tokenizer, texts, cfg["max_seq_length"])
        theirs = np.concatenate(
            [encode(model, tokenizer, [t], cfg["max_seq_length"], "cpu").numpy() for t in texts]
        )
        out = agreement(theirs, ours)
        if out["min"] < 0.9999:
            target.unlink()
            raise SystemExit(f"{name}: fp32 ONNX differs from PyTorch: {out}")
        print(f"{name}: {target.stat().st_size / 1e6:.0f} MB, cosine to PyTorch {out}")


if __name__ == "__main__":
    main()
