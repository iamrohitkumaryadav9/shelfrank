"""Assemble the Colab notebook that fine-tunes the dense retriever, from tested modules."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shelfrank.qu.seed_notebook import _cell

EMBEDDED = (
    "shelfrank/text/normalize.py",
    "shelfrank/text/near_duplicates.py",
    "shelfrank/data/download.py",
    "shelfrank/qu/seed_selection.py",
    "shelfrank/dense/esci_data.py",
    "shelfrank/dense/finetune.py",
    "shelfrank/dense/onnx_export.py",
    "scripts/finetune_dense.py",
)

INTRO = """# ShelfRank: fine-tune bge-small on ESCI

Needs a GPU: **Runtime > Change runtime type > T4 GPU**. Then **Runtime > Run all** and allow
access to Google Drive. Everything that matters is kept in one Drive folder, about 2 GB while
training.

If the session ends, choose **Run all** again. Finished stages are skipped and training continues
from the last checkpoint. The last cell downloads `bge_small_esci.zip`.

The code cells are copies of the repository's tested modules. Nothing needs editing."""

SETUP = """# Every path and setting is here.
import os, subprocess, sys
from pathlib import Path

PACKAGES = __PACKAGES__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DEFAULT_WORK, DEFAULT_KEEP = "/content/shelfrank_ft", "/content/drive/MyDrive/shelfrank_ft"
elif Path("/kaggle").exists():
    DEFAULT_WORK, DEFAULT_KEEP = "/tmp/shelfrank_ft", "/kaggle/working/shelfrank_ft"
else:
    DEFAULT_WORK, DEFAULT_KEEP = "shelfrank_ft_work", "shelfrank_ft"
WORK = Path(os.environ.get("SHELFRANK_WORK", DEFAULT_WORK)).resolve()
KEEP = Path(os.environ.get("SHELFRANK_KEEP", DEFAULT_KEEP)).resolve()

SETTINGS = __SETTINGS__
TRAINING = __TRAINING__
SOURCES = __SOURCES__
MODEL = __MODEL__
PROVENANCE = __PROVENANCE__

for folder in ("shelfrank/text", "shelfrank/data", "shelfrank/qu", "shelfrank/dense", "scripts"):
    (WORK / folder).mkdir(parents=True, exist_ok=True)
    (WORK / folder / "__init__.py").touch()
(WORK / "shelfrank" / "__init__.py").touch()
KEEP.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEVICE, "| kept in", KEEP)
assert DEVICE == "cuda" or os.environ.get("SHELFRANK_ALLOW_CPU"), (
    "No GPU. Choose Runtime > Change runtime type > T4 GPU, then Run all again.")"""

DATA = """from scripts import finetune_dense as ft
from shelfrank.dense.finetune import TrainConfig

paths = ft.Paths(WORK, KEEP)
config = TrainConfig(**TRAINING)
data = ft.stage_data(paths, SOURCES, SETTINGS)
base = ft.download_base(WORK, MODEL["model"], MODEL["revision"])"""

TRAIN = """# Resumable. If it stops at the time limit or the session ends, choose Run all again.
out = ft.stage_train(paths, data, base, config, MODEL["query_prefix"], DEVICE)
print(out["step"], "of", out["total"], "steps, finished:", out["finished"])"""

EVALUATE = """ft.stage_evaluate(paths, data, base, config, MODEL["query_prefix"], DEVICE)"""

PACKAGE = """prefix = MODEL["query_prefix"]
archive = ft.stage_package(paths, data, base, config, prefix, PROVENANCE)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print("copy this file:", archive)"""

PACKAGES = (
    "transformers==4.45.2",
    "onnx",
    "onnxruntime",
)


def build(
    root: Path, cfg: Mapping[str, Any], model: Mapping[str, Any], git_commit: str
) -> dict[str, Any]:
    """Build the notebook as a dictionary."""
    embedded = {name: (root / name).read_text(encoding="utf-8") for name in EMBEDDED}
    hashes = {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in embedded.items()}
    provenance = {
        "git_commit": git_commit,
        "base_model": model["model"],
        "base_revision": model["revision"],
        "code_sha256": hashes,
    }
    values = {
        "__PACKAGES__": list(PACKAGES),
        "__SETTINGS__": dict(cfg["finetune"]["data"]),
        "__TRAINING__": dict(cfg["finetune"]["training"]),
        "__SOURCES__": dict(cfg["sources"]),
        "__MODEL__": {k: model[k] for k in ("model", "revision", "query_prefix")},
        "__PROVENANCE__": provenance,
    }
    setup = SETUP
    for key, value in values.items():
        setup = setup.replace(key, repr(value))
    parts = [("markdown", INTRO), ("code", setup)]
    parts += [("code", f"%%writefile {name}\n{text}") for name, text in embedded.items()]
    parts += [("code", c) for c in (DATA, TRAIN, EVALUATE, PACKAGE)]
    return {
        "cells": [_cell(kind, text, i) for i, (kind, text) in enumerate(parts)],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def commit_in(notebook: Mapping[str, Any]) -> str:
    """Commit that a built notebook records."""
    setup = "".join(notebook["cells"][1]["source"])
    return setup.split("'git_commit': '", 1)[1].split("'", 1)[0]
