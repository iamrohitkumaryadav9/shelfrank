"""First-stage scores and the candidate pairs that the ranker re-orders."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shelfrank.bm25.fields import BM25Index, combine, score_matrix
from shelfrank.ranking import rank_all, reciprocal_rank_fusion, union

Run = dict[int, list[int]]


@dataclass
class FirstStage:
    """Scores of every query against the whole catalogue, and the runs made from them."""

    query_ids: list[int]
    texts: list[str]
    product_ids: np.ndarray
    fields: dict[str, np.ndarray]
    tuned: np.ndarray
    cosine: np.ndarray
    runs: dict[str, Run]


def first_stage(
    query_ids: Sequence[int],
    texts: Sequence[str],
    product_ids: np.ndarray,
    tokens: Mapping[str, list[list[str]]],
    weights: Mapping[str, float],
    query_vectors: np.ndarray,
    vectors: np.ndarray,
    k1: float,
    b: float,
    rrf_k: int,
    depth: int,
) -> FirstStage:
    """Score and rank. Every BM25 field is scored, including those with weight 0, as features."""
    fields = {name: score_matrix(BM25Index(tokens[name], k1, b), list(texts)) for name in weights}
    tuned = combine(fields, weights)
    cosine = (query_vectors @ vectors.T).astype(np.float32)
    runs = {
        "bm25_multi": rank_all(tuned, query_ids, product_ids, depth, positive_only=True),
        "dense": rank_all(cosine, query_ids, product_ids, depth),
    }
    runs["hybrid_rrf"] = {
        q: reciprocal_rank_fusion([runs["bm25_multi"][q], runs["dense"][q]], rrf_k, depth)
        for q in runs["dense"]
    }
    return FirstStage(
        [int(q) for q in query_ids], list(texts), product_ids, fields, tuned, cosine, runs
    )


def candidate_pairs(stage: FirstStage, depth: int) -> pd.DataFrame:
    """Union of both generators' top ``depth``, with the row of each query and product.

    Sorted by query and product id, so nothing downstream can depend on generator order.
    """
    rows = []
    position = {int(p): i for i, p in enumerate(stage.product_ids)}
    for qi, q in sorted(enumerate(stage.query_ids), key=lambda item: item[1]):
        found = union([stage.runs["bm25_multi"][q], stage.runs["dense"][q]], depth)
        rows += [(q, p, qi, position[p]) for p in sorted(found)]
    return pd.DataFrame(rows, columns=["query_id", "product_id", "qi", "pi"])
