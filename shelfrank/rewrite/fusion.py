"""Put rewrites into retrieval, before the ranker."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from shelfrank.ltr.first_stage import FirstStage, Run
from shelfrank.ranking import reciprocal_rank_fusion


def fuse_runs(
    original: Run, rewrites: Mapping[int, Sequence[Sequence[int]]], k: int, depth: int
) -> Run:
    """Fuse each query's ranking with those of its rewrites. Queries without one are unchanged."""
    return {
        q: reciprocal_rank_fusion([ranking, *rewrites[q]], k, depth) if rewrites.get(q) else ranking
        for q, ranking in original.items()
    }


def with_rewrites(
    stage: FirstStage,
    bm25: Mapping[int, Sequence[Sequence[int]]],
    dense: Mapping[int, Sequence[Sequence[int]]],
    k: int,
    depth: int,
) -> FirstStage:
    """The same stage with fused runs.

    Only runs change. Text features still describe the query the shopper typed, so the ranker sees a
    wider candidate set and better ranks but no words the shopper did not write.
    """
    runs = {
        "bm25_multi": fuse_runs(stage.runs["bm25_multi"], bm25, k, depth),
        "dense": fuse_runs(stage.runs["dense"], dense, k, depth),
    }
    runs["hybrid_rrf"] = {
        q: reciprocal_rank_fusion([runs["bm25_multi"][q], runs["dense"][q]], k, depth)
        for q in runs["dense"]
    }
    return replace(stage, runs=runs)
