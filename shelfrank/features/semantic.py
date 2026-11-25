"""G2: dense similarity and its fusion with BM25."""

import pandas as pd

from shelfrank.features.lexical import reciprocal_ranks
from shelfrank.ltr.first_stage import FirstStage


def semantic_features(pairs: pd.DataFrame, stage: FirstStage, rrf_k: int) -> pd.DataFrame:
    """Columns g2_*. The gap to the query's best product makes cosines comparable across queries.

    The fused score and the fused rank are exactly what hybrid RRF orders by, so a ranker that uses
    nothing else reproduces it. A product below the run's depth gets the first rank outside it.
    """
    qi, pi = pairs["qi"].to_numpy(), pairs["pi"].to_numpy()
    cosine = stage.cosine[qi, pi]
    dense = reciprocal_ranks(pairs, stage.runs["dense"], rrf_k)
    bm25 = reciprocal_ranks(pairs, stage.runs["bm25_multi"], rrf_k)
    ranks = {q: {p: i for i, p in enumerate(r, 1)} for q, r in stage.runs["hybrid_rrf"].items()}
    outside = {q: len(r) + 1 for q, r in ranks.items()}
    fused = [
        ranks[q].get(p, outside[q])
        for q, p in zip(pairs["query_id"], pairs["product_id"], strict=True)
    ]
    return pd.DataFrame(
        {
            "g2_cosine": cosine,
            "g2_cosine_gap": stage.cosine.max(axis=1)[qi] - cosine,
            "g2_rr_dense": dense,
            "g2_rrf": [a + b for a, b in zip(dense, bm25, strict=True)],
            "g2_rrf_rank": fused,
        }
    )
