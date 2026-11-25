"""Score whole runs and summarise them with the protocol's denominators."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import pandas as pd

from shelfrank.eval.metrics import METRIC_NAMES, perfect_ranking, score_query

Run = Mapping[int, Sequence[int]]
Qrels = Mapping[int, Mapping[int, int]]
PerQuery = dict[int, dict[str, float | None]]


def to_qrels(judged: pd.DataFrame, query_ids: Iterable[int]) -> dict[int, dict[int, int]]:
    """Restrict judgments to the given queries, so a run cannot be scored on any other."""
    wanted = [int(q) for q in query_ids]
    out: dict[int, dict[int, int]] = {q: {} for q in wanted}
    rows = judged[judged["query_id"].isin(out)]
    for query_id, product_id, grade in zip(
        rows["query_id"], rows["product_id"], rows["grade"], strict=True
    ):
        out[int(query_id)][int(product_id)] = int(grade)
    return out


def evaluate(run: Run, qrels: Qrels) -> PerQuery:
    """Score every query in ``qrels``.

    A run that lacks a query is an error and not a zero: silently scoring 0 would look like a weak
    system, and silently skipping would look like a strong one.
    """
    missing = set(qrels) - set(run)
    if missing:
        raise KeyError(f"run has no ranking for {len(missing)} queries")
    return {q: asdict(score_query(run[q], qrels[q])) for q in sorted(qrels)}


def summarise(per_query: PerQuery) -> dict[str, dict[str, float | int]]:
    """Mean and n per metric, over the queries where the metric is defined."""
    summary: dict[str, dict[str, float | int]] = {}
    for name in METRIC_NAMES:
        values = [row[name] for row in per_query.values() if row[name] is not None]
        summary[name] = {
            "mean": sum(values) / len(values) if values else float("nan"),
            "n": len(values),
        }
    return summary


def ceilings(qrels: Qrels) -> dict[str, dict[str, float | int]]:
    """Score of a perfect ranker. Judged@10 is capped by how many products were judged."""
    return summarise(evaluate({q: perfect_ranking(j) for q, j in qrels.items()}, qrels))


def candidate_report(candidates: Run, qrels: Qrels) -> dict[str, Any]:
    """Describe a candidate set as an upper bound for any re-ranker over it.

    The oracle ordering is the bound that matters for nDCG@10. Recall alone does not show it,
    because missing the 101st relevant product cannot hurt a top-10 metric.
    """
    oracle = {q: sorted(candidates[q], key=lambda p: (-qrels[q].get(p, 0), p)) for q in qrels}
    summary = summarise(evaluate(oracle, qrels))
    sizes = [len(candidates[q]) for q in qrels]
    judged = [
        sum(p in qrels[q] for p in candidates[q]) / len(candidates[q])
        for q in qrels
        if candidates[q]
    ]
    recalls, exact_recalls = [], []
    for q, grades in qrels.items():
        found = set(candidates[q])
        for level, sink in ((1, recalls), (2, exact_recalls)):
            wanted = {p for p, g in grades.items() if g >= level}
            if wanted:
                sink.append(len(wanted & found) / len(wanted))
    return {
        "mean_size": sum(sizes) / len(sizes),
        "recall": sum(recalls) / len(recalls),
        "recall_exact": sum(exact_recalls) / len(exact_recalls),
        "judged_share": sum(judged) / len(judged),
        "oracle_ndcg10": summary["ndcg10"]["mean"],
        "oracle_exact10_share": summary["exact10_share"]["mean"],
    }
