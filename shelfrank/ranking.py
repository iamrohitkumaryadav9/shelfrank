"""Turn scores into rankings, and fuse rankings."""

from collections.abc import Sequence

import numpy as np


def top_products(
    scores: np.ndarray, product_ids: np.ndarray, k: int, positive_only: bool = False
) -> list[int]:
    """Best ``k`` product ids. Ties go to the lower product id, so every run is reproducible.

    ``positive_only`` drops products that match no query term. BM25 gives them 0, and padding a
    ranking with them would return arbitrary products.
    """
    order = np.argsort(-scores, kind="stable")[:k]
    if positive_only:
        order = order[scores[order] > 0]
    return [int(p) for p in product_ids[order]]


def rank_all(
    matrix: np.ndarray,
    query_ids: Sequence[int],
    product_ids: np.ndarray,
    k: int,
    positive_only: bool = False,
) -> dict[int, list[int]]:
    """Rank every query. Requires ascending product ids, which is what makes ties deterministic."""
    if np.any(np.diff(product_ids) <= 0):
        raise ValueError("product_ids must be strictly ascending")
    return {
        int(q): top_products(row, product_ids, k, positive_only)
        for q, row in zip(query_ids, matrix, strict=True)
    }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, depth: int = 1000
) -> list[int]:
    """Fuse rankings by summed reciprocal rank, which needs no score calibration between systems."""
    totals: dict[int, float] = {}
    for ranking in rankings:
        for rank, product in enumerate(ranking[:depth], 1):
            totals[product] = totals.get(product, 0.0) + 1 / (k + rank)
    return sorted(totals, key=lambda p: (-totals[p], p))[:depth]


def union(rankings: Sequence[Sequence[int]], depth: int) -> list[int]:
    """Products in any ranking's top ``depth``, in first-seen order."""
    return list(dict.fromkeys(p for ranking in rankings for p in ranking[:depth]))
