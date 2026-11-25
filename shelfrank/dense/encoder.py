"""Encode products once, cache to disk, and refuse a cache that does not match its inputs."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shelfrank.config import REPO_ROOT
from shelfrank.data.download import sha256_of

Encode = Callable[[list[str]], np.ndarray]


class StaleCacheError(RuntimeError):
    """The cached embeddings were made from different inputs."""


def product_text(name: object, product_class: object, description: object) -> str:
    """Build the passage, most specific field first so truncation removes only description.

    Feature values are left out: at 128 tokens they would displace the description with
    compliance flags and dimensions.
    """
    parts = [p.strip() for p in (name, product_class, description) if isinstance(p, str)]
    return " . ".join(p.replace("|", " , ") for p in parts if p)


def product_texts(products: pd.DataFrame) -> list[str]:
    """Passages in catalogue order."""
    columns = (products["product_name"], products["product_class"], products["product_description"])
    return [product_text(*row) for row in zip(*columns, strict=True)]


def fingerprint(texts: Sequence[str]) -> str:
    """Hash of the exact passages, so a changed template invalidates the cache."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def encode_cached(
    texts: Sequence[str],
    directory: Path,
    identity: dict[str, Any],
    encode: Encode,
    chunk: int = 2048,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """Encode in resumable chunks and store the result with a manifest.

    Texts are sorted by length, so a batch holds similar lengths and little padding is wasted.
    The original order is restored at the end.
    """
    manifest = {**identity, "texts_sha256": fingerprint(texts), "count": len(texts)}
    final, manifest_path = directory / "embeddings.npy", directory / "manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise StaleCacheError(f"{directory} was built from other inputs; delete it to rebuild")
        if final.exists():
            return np.load(final)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = []
    for start in range(0, len(texts), chunk):
        path = directory / f"part_{start:06d}.npy"
        if not path.exists():
            block = np.asarray(encode([texts[i] for i in order[start : start + chunk]]), np.float32)
            partial = path.with_suffix(".tmp.npy")
            np.save(partial, block)
            partial.replace(path)
            log(f"encoded {min(start + chunk, len(texts)):,} / {len(texts):,}")
        parts.append(np.load(path))
    ordered = np.concatenate(parts)
    restored = np.empty_like(ordered)
    restored[order] = ordered
    np.save(final, restored)
    for path in directory.glob("part_*.npy"):
        path.unlink()
    return restored


def model_identity(cfg: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """What the cache key and loader need. A local model is pinned by the hash of its weights."""
    identity = {k: cfg[k] for k in ("model", "revision", "max_seq_length")}
    if cfg["revision"] == "local":
        weights = root / cfg["model"] / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"{weights} is missing: see docs/COLAB.md")
        identity["revision"] = f"sha256:{sha256_of(weights)}"
    return identity


def load_sentence_transformer(model: str, revision: str, max_seq_length: int) -> Encode:
    """Load a pinned model on CPU and return an encoder of unit vectors.

    Imported here so that metric and BM25 code never needs torch.
    """
    from sentence_transformers import SentenceTransformer

    if revision.startswith("sha256:"):
        # Our own file has no pooler layer. bge pools the first token, so nothing ever reads that
        # layer, and creating it with random weights would only raise a warning.
        encoder = SentenceTransformer(
            str(REPO_ROOT / model),
            device="cpu",
            local_files_only=True,
            model_kwargs={"add_pooling_layer": False},
        )
    else:
        encoder = SentenceTransformer(model, revision=revision, device="cpu")
    encoder.max_seq_length = max_seq_length

    def encode(texts: list[str]) -> np.ndarray:
        return encoder.encode(texts, batch_size=32, normalize_embeddings=True)

    return encode
