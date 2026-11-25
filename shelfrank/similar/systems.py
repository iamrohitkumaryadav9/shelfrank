"""Systems that rank the catalogue for a seed product."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from shelfrank.bm25.fields import BM25Index
from shelfrank.features.attributes import Listed, contains
from shelfrank.ranking import reciprocal_rank_fusion, top_products

Ranking = list[int]


@dataclass(frozen=True)
class RerankWeights:
    """Weights of the terms added to the scaled fusion score."""

    shared_class: float = 0.0
    attributes: float = 0.0
    prior: float = 0.0


def bm25_rankings(
    index: BM25Index,
    seed_tokens: Mapping[int, Sequence[str]],
    product_ids: np.ndarray,
    depth: int,
) -> dict[int, Ranking]:
    """Ranking per seed product id, without the seed."""
    out = {}
    for product, tokens in seed_tokens.items():
        ranked = top_products(index.scores(tokens), product_ids, depth + 1, positive_only=True)
        out[product] = [p for p in ranked if p != product][:depth]
    return out


def dense_rankings(
    vectors: np.ndarray,
    rows: Mapping[int, int],
    product_ids: np.ndarray,
    depth: int,
    chunk: int = 256,
) -> dict[int, Ranking]:
    """Nearest products by cosine, without the seed. ``rows`` gives each seed's row."""
    out: dict[int, Ranking] = {}
    seeds = list(rows)
    for start in range(0, len(seeds), chunk):
        part = seeds[start : start + chunk]
        scores = vectors[[rows[s] for s in part]] @ vectors.T
        for product, row in zip(part, scores, strict=True):
            ranked = top_products(row, product_ids, depth + 1)
            out[product] = [p for p in ranked if p != product][:depth]
    return out


def fuse(
    a: Mapping[int, Ranking], b: Mapping[int, Ranking], k: int, depth: int
) -> dict[int, Ranking]:
    """Reciprocal rank fusion per seed."""
    return {p: reciprocal_rank_fusion([a[p], b[p]], k, depth) for p in a}


def shared_class(a: Sequence[str], b: Sequence[str]) -> float:
    """1 when the two products share a catalogue class. Unknown counts as not shared."""
    return float(bool(set(a) & set(b)))


def agreement(a: Listed, b: Listed) -> float:
    """Share of attribute families that both products mention in which a value agrees.

    0 when they have no family in common, so silence is neither rewarded nor punished more than a
    disagreement.
    """
    pairs = ((a.union, b.union, {"color", "material_finish"}), (a.style, b.style, {"style"}))
    both = hits = 0
    for left, right, names in (*pairs, (a.room, b.room, {"room"})):
        if names & a.mentioned and names & b.mentioned:
            both += 1
            hits += any(contains(x, y) or contains(y, x) for x in left for y in right)
    return hits / both if both else 0.0


def rating_prior(rating: np.ndarray, reviews: np.ndarray) -> np.ndarray:
    """Shrunk rating times log reviews, scaled to [0, 1] over the catalogue."""
    score = rating * np.log1p(reviews)
    top = float(score.max()) or 1.0
    return score / top


def rerank(
    seed: int,
    ranking: Ranking,
    k: int,
    runs: Sequence[Ranking],
    weights: RerankWeights,
    terms: Mapping[int, tuple[float, float, float]],
) -> Ranking:
    """Order the fused top by fusion score plus weighted terms. ``terms`` is per candidate."""
    ranks = [{p: i for i, p in enumerate(run, 1)} for run in runs]
    top = len(runs) / (k + 1)

    def score(product: int) -> float:
        fused = sum(1 / (k + r[product]) for r in ranks if product in r) / top
        same, agree, prior = terms[product]
        return (
            fused + weights.shared_class * same + weights.attributes * agree + weights.prior * prior
        )

    return sorted((p for p in ranking if p != seed), key=lambda p: (-score(p), p))
