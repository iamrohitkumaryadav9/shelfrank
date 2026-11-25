"""Check the fine-tuned model that came back from Colab. Writes results/phase3_checks.json.

Uses product classes and invented texts only: no relevance label and no WANDS query.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finetune_dense import model_card
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.download import sha256_of
from shelfrank.data.wands import load_products
from shelfrank.dense.encoder import (
    fingerprint,
    load_sentence_transformer,
    model_identity,
    product_texts,
)
from shelfrank.dense.finetune import embed
from shelfrank.dense.finetune_notebook import EMBEDDED

SEED = 20260
TEXTS = (
    "grey sofa",
    "Represent this sentence for searching relevant passages: grey sofa",
    "Solid oak round dining table . Dining Tables . Seats four people",
    "Ceramic table lamp with linen shade . Table Lamps . 60 watt bulb not included",
    "Wool area rug in navy with a low pile . Area Rugs . Hand woven , spot clean only",
)
MODELS = ("bge_small", "bge_small_esci")


def pooling(name: str) -> dict[str, Any]:
    """Whether our vectors are the first token, and whether the pooler layer has any effect."""
    cfg = load_config("retrieval")["dense"][name]
    identity = model_identity(cfg, REPO_ROOT)
    ours = np.asarray(load_sentence_transformer(**identity)(list(TEXTS)))
    local = cfg["revision"] == "local"
    source = str(REPO_ROOT / cfg["model"]) if local else cfg["model"]
    options: dict[str, Any] = {} if local else {"revision": cfg["revision"]}
    # Loaded with the layer, as any ordinary caller would.
    network = SentenceTransformer(source, device="cpu", **options)
    network.max_seq_length = cfg["max_seq_length"]
    body = network[0].auto_model
    batch = network.tokenize(list(TEXTS))

    def vectors() -> np.ndarray:
        return network.encode(list(TEXTS), normalize_embeddings=True)

    with torch.no_grad():
        first = embed(body, batch).numpy()
        pooler_output = torch.nn.functional.normalize(body(**batch).pooler_output, dim=-1).numpy()
        with_layer = vectors()
        body.pooler.dense.weight.normal_(0, 100)
        body.pooler.dense.bias.normal_(0, 100)
        noised = vectors()
        body.pooler = None
        removed = vectors()
    control = (ours * pooler_output).sum(axis=1)
    return {
        "modules": [type(m).__name__ for m in network],
        "pooling": network[1].get_pooling_mode_str(),
        "max_difference_from_first_token": float(np.abs(ours - first).max()),
        "cosine_to_pooler_output": [float(control.min()), float(control.max())],
        "identical_with_layer": bool(np.array_equal(ours, with_layer)),
        "identical_noised": bool(np.array_equal(ours, noised)),
        "identical_removed": bool(np.array_equal(ours, removed)),
    }


def pooler_tensors() -> dict[str, list[str]]:
    """Pooler weights present in each weight file, which explains who warns."""
    from huggingface_hub import hf_hub_download

    out = {}
    for name in MODELS:
        cfg = load_config("retrieval")["dense"][name]
        if cfg["revision"] == "local":
            path = str(REPO_ROOT / cfg["model"] / "model.safetensors")
        else:
            path = hf_hub_download(cfg["model"], "model.safetensors", revision=cfg["revision"])
        with safe_open(path, "pt") as handle:
            out[name] = [k for k in handle.keys() if "pooler" in k]  # noqa: SIM118
    return out


def embeddings(name: str, texts: list[str], classes: np.ndarray) -> dict[str, Any]:
    """Manifest, reproducibility, and how well the vectors separate product classes."""
    folder = REPO_ROOT / "artifacts/embeddings" / name
    cfg = load_config("retrieval")["dense"][name]
    identity = model_identity(cfg, REPO_ROOT)
    vectors = np.load(folder / "embeddings.npy")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    expected = {**identity, "texts_sha256": fingerprint(texts), "count": len(texts)}
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(texts), 1000, replace=False)
    again = rng.choice(len(texts), 40, replace=False)
    encode = load_sentence_transformer(**identity)
    cosine = (np.asarray(encode([texts[i] for i in again])) * vectors[again]).sum(axis=1)
    upper = np.triu_indices(1000, 1)
    pairwise = (vectors[sample] @ vectors[sample].T)[upper]

    rng = np.random.default_rng(SEED)
    probe = rng.choice(np.flatnonzero(classes != ""), 1000, replace=False)
    scores = vectors[probe] @ vectors.T
    scores[np.arange(1000), probe] = -9
    top = np.argpartition(-scores, 10, axis=1)[:, :10]
    pairs = (vectors[probe] @ vectors[probe].T)[upper]
    same = classes[probe][upper[0]] == classes[probe][upper[1]]
    inside, outside = pairs[same], pairs[~same]
    order = np.sort(outside)
    auc = np.searchsorted(order, inside).mean() / len(order)
    query = encode([cfg["query_prefix"] + "grey sofa"])[0]
    best = np.argsort(-(vectors @ query))[:10]
    return {
        "manifest_matches": manifest == expected,
        "shape": list(vectors.shape),
        "finite": bool(np.isfinite(vectors).all()),
        "reencoded_min_cosine": float(cosine.min()),
        "pairwise_mean": float(pairwise.mean()),
        "pairwise_sd": float(pairwise.std()),
        "same_class_mean": float(inside.mean()),
        "other_class_mean": float(outside.mean()),
        "same_class_pairs": len(inside),
        "class_auc": float(auc),
        "neighbours_same_class": float((classes[top] == classes[probe][:, None]).mean()),
        "grey_sofa_classes": [str(classes[i]) for i in best],
        "_top": top,
    }


def provenance(training: dict[str, Any]) -> dict[str, Any]:
    """Compare what the package says about itself with the repository at that commit."""
    recorded = training["provenance"]
    commit = recorded["git_commit"]

    def blob(path: str) -> str:
        command = ["git", "show", f"{commit}:{path}"]
        done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, check=True)
        return done.stdout.decode("utf-8").replace("\r\n", "\n")

    import hashlib

    import yaml

    files = {
        name: hashlib.sha256(blob(name).encode("utf-8")).hexdigest() == digest
        for name, digest in recorded["code_sha256"].items()
    }
    settings = yaml.safe_load(blob("configs/esci.yaml"))["finetune"]
    base = load_config("retrieval")["dense"]["bge_small"]
    config = training["config"]
    return {
        "commit": commit,
        "code_files_match": files,
        "every_embedded_file_recorded": set(files) == set(EMBEDDED),
        "training_settings_match": all(config[k] == v for k, v in settings["training"].items()),
        "data_settings_match": training["settings"] == settings["data"],
        "base_model_matches": (recorded["base_model"], recorded["base_revision"])
        == (base["model"], base["revision"]),
    }


def main() -> None:
    """Run every check."""
    warnings.simplefilter("ignore")
    folder = REPO_ROOT / "artifacts/models/bge_small_esci"
    training = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    copy = REPO_ROOT / "results/finetune_esci.json"
    copy.write_text(json.dumps(training, indent=1) + "\n", encoding="utf-8", newline="\n")
    card = model_card(training)
    for path in (folder / "README.md", REPO_ROOT / "docs/model_card_bge_small_esci.md"):
        path.write_text(card, encoding="utf-8", newline="\n")
    products = load_products(REPO_ROOT / load_config("data")["wands"]["raw_dir"])
    products = products.sort_values("product_id").reset_index(drop=True)
    texts = product_texts(products)
    classes = products["product_class"].fillna("").to_numpy()
    checks = {name: embeddings(name, texts, classes) for name in MODELS}
    tops = [checks[name].pop("_top") for name in MODELS]
    shared = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(*tops, strict=True)])
    out = {
        "pooler_tensors_in_file": pooler_tensors(),
        "pooling": {name: pooling(name) for name in MODELS},
        "embeddings": checks,
        "shared_neighbours": float(shared),
        "provenance": provenance(training),
        "weights_sha256": sha256_of(folder / "model.safetensors"),
        "reference_embeddings_in_package": (folder / "reference_embeddings.json").exists(),
        "run_log_in_package": "run" in training,
        "per_query_held_out_in_package": "held_out_after_per_query" in training,
    }
    path = REPO_ROOT / "results/phase3_checks.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
