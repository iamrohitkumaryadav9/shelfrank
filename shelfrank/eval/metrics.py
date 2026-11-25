"""Ranking metrics for one query. Definitions follow docs/PROTOCOL.md section 3."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

EXACT = 2
RELEVANT = 1


def dcg(grades: Sequence[int], k: int) -> float:
    """Discounted cumulative gain with linear gain, the convention of the WANDS paper's 1/0.5/0."""
    return sum(grade / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def ndcg(ranked: Sequence[int], ideal_pool: Sequence[int], k: int) -> float:
    """DCG over the best DCG reachable from all judged products, 0 when nothing is relevant."""
    best = dcg(sorted(ideal_pool, reverse=True), k)
    return dcg(ranked, k) / best if best > 0 else 0.0


@dataclass(frozen=True)
class QueryMetrics:
    """Every metric for one query. None marks a metric the protocol does not define for it."""

    ndcg10: float
    condensed_ndcg10: float
    judged10: float
    recall100: float | None
    exact10: float | None
    exact10_share: float | None
    mrr_exact: float | None
    recall100_exact: float | None


METRIC_NAMES = tuple(QueryMetrics.__dataclass_fields__)


def score_query(ranking: Sequence[int], judged: Mapping[int, int]) -> QueryMetrics:
    """Score a ranking of product ids, best first, against one query's judgments.

    Exact-based metrics are undefined without an Exact product. Scoring them 0 would lower every
    system by the same amount and hide the real level.
    """
    if len(set(ranking)) != len(ranking):
        raise ValueError("a ranking must not repeat a product")
    grades = [judged.get(product, 0) for product in ranking]
    condensed = [judged[product] for product in ranking if product in judged]
    pool = list(judged.values())
    n_exact = sum(grade >= EXACT for grade in pool)
    n_relevant = sum(grade >= RELEVANT for grade in pool)

    top_exact = sum(grade >= EXACT for grade in grades[:10])
    first = next((rank for rank, grade in enumerate(grades, 1) if grade >= EXACT), None)
    hits = sum(grade >= RELEVANT for grade in grades[:100])
    exact_hits = sum(grade >= EXACT for grade in grades[:100])
    return QueryMetrics(
        ndcg10=ndcg(grades, pool, 10),
        condensed_ndcg10=ndcg(condensed, pool, 10),
        judged10=sum(product in judged for product in ranking[:10]) / 10,
        recall100=hits / n_relevant if n_relevant else None,
        exact10=top_exact / 10 if n_exact else None,
        exact10_share=top_exact / min(10, n_exact) if n_exact else None,
        mrr_exact=(1 / first if first else 0.0) if n_exact else None,
        recall100_exact=exact_hits / n_exact if n_exact else None,
    )


def perfect_ranking(judged: Mapping[int, int]) -> list[int]:
    """Judged products by descending grade: the ranking that attains every ceiling."""
    return sorted(judged, key=lambda product: (-judged[product], product))
