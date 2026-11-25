"""Detect WANDS queries inside derived files."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from shelfrank.text.near_duplicates import jaccard, leakage_keys


def key_hashes(query: str) -> set[str]:
    """Hashes of a query's order-free token keys.

    Committed for all 480 queries, so CI can check any file for a WANDS query without holding the
    query text. WANDS is public; this limits exposure and is not secrecy.
    """
    return {
        hashlib.sha256(" ".join(sorted(key)).encode("utf-8")).hexdigest()
        for key in leakage_keys(query)
    }


def load_key_hashes(path: Path) -> set[str]:
    """Read the committed hashes."""
    return set(json.loads(path.read_text(encoding="utf-8"))["hashes"])


def exact_matches(lines: Iterable[str], hashes: set[str]) -> list[int]:
    """Indices of lines with the same tokens as a WANDS query. Indices, so nothing is printed."""
    return [i for i, line in enumerate(lines) if key_hashes(line) & hashes]


def near_matches(lines: Iterable[str], queries: Iterable[str], threshold: float) -> list[int]:
    """Indices of lines that resemble a WANDS query. Needs the raw data."""
    protected = [key for query in queries for key in leakage_keys(query)]
    return [
        i
        for i, line in enumerate(lines)
        if any(jaccard(own, key) >= threshold for own in leakage_keys(line) for key in protected)
    ]
