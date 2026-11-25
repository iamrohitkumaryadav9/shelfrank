"""Substitute sets from human labels, and their metrics. docs/PROTOCOL.md section 16."""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from shelfrank.eval.harness import Qrels
from shelfrank.eval.metrics import EXACT
from shelfrank.stats.bootstrap import Interval

MAX_SEEDS = 20
DEPTH = 100
METRICS = ("ndcg10", "condensed_ndcg10", "recall10", "condensed_recall10", "mrr")


@dataclass(frozen=True)
class Seed:
    """One product of one query's substitute set."""

    query_id: int
    product_id: int


def substitute_sets(qrels: Qrels) -> dict[int, list[int]]:
    """Exact products per query, for queries with at least 2."""
    sets = {q: sorted(p for p, g in judged.items() if g >= EXACT) for q, judged in qrels.items()}
    return {q: products for q, products in sets.items() if len(products) >= 2}


def draw_seeds(
    sets: Mapping[int, Sequence[int]], limit: int = MAX_SEEDS, seed: int = 20260
) -> list[Seed]:
    """At most ``limit`` seeds per set. The draw of one set does not depend on the others."""
    out = []
    for q in sorted(sets):
        products = list(sets[q])
        if len(products) > limit:
            products = sorted(random.Random(seed * 1_000_003 + q).sample(products, limit))
        out += [Seed(q, p) for p in products]
    return out


def _dcg(hits: Sequence[bool]) -> float:
    return sum(1 / math.log2(i + 2) for i, hit in enumerate(hits[:10]) if hit)


def score_seed(
    ranking: Sequence[int], relevant: set[int], judged: Mapping[int, int]
) -> dict[str, float]:
    """Metrics for one seed. ``relevant`` and the ranking must not contain the seed."""
    hits = [p in relevant for p in ranking]
    condensed = [p in relevant for p in ranking if p in judged]
    best = _dcg([True] * len(relevant))
    first = next((i for i, hit in enumerate(hits, 1) if hit), None)
    return {
        "ndcg10": _dcg(hits) / best,
        "condensed_ndcg10": _dcg(condensed) / best,
        "recall10": sum(hits[:10]) / len(relevant),
        "condensed_recall10": sum(condensed[:10]) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "recall10_ceiling": min(10, len(relevant)) / len(relevant),
    }


def evaluate_seeds(
    rankings: Mapping[Seed, Sequence[int]], sets: Mapping[int, Sequence[int]], qrels: Qrels
) -> dict[Seed, dict[str, float]]:
    """Score every seed, without the seed in its own ranking or set."""
    out = {}
    for seed, ranking in rankings.items():
        others = set(sets[seed.query_id]) - {seed.product_id}
        kept = [p for p in ranking if p != seed.product_id]
        out[seed] = score_seed(kept, others, qrels[seed.query_id])
    return out


def by_query(scores: Mapping[Seed, Mapping[str, float]], metric: str) -> dict[int, list[float]]:
    """Seed scores grouped by query."""
    out: dict[int, list[float]] = {}
    for seed in sorted(scores, key=lambda s: (s.query_id, s.product_id)):
        out.setdefault(seed.query_id, []).append(scores[seed][metric])
    return out


def clustered_mean(
    groups: Mapping[int, Sequence[float]],
    weighted: bool,
    n_resamples: int = 10_000,
    seed: int = 20260,
) -> Interval:
    """Mean with an interval from resampling queries.

    Not ``weighted``: the mean of query means. ``weighted``: the mean over seeds, where a resampled
    query brings all its seeds.
    """
    sums = np.array([sum(v) if weighted else sum(v) / len(v) for v in groups.values()])
    sizes = np.array([len(v) if weighted else 1 for v in groups.values()], dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(sums), (n_resamples, len(sums)))
    means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return Interval(float(sums.sum() / sizes.sum()), float(low), float(high), len(sums))


def difference(
    system: Mapping[Seed, Mapping[str, float]],
    baseline: Mapping[Seed, Mapping[str, float]],
    metric: str,
    weighted: bool = False,
    **bootstrap: Any,  # noqa: ANN401
) -> Interval:
    """Paired by seed, resampled by query."""
    if set(system) != set(baseline):
        raise ValueError("both systems must cover the same seeds")
    gaps = {s: {metric: system[s][metric] - baseline[s][metric]} for s in system}
    return clustered_mean(by_query(gaps, metric), weighted, **bootstrap)
