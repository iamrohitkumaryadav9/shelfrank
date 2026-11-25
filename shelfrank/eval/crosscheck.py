"""Compare our metric code with ranx, query by query."""

from collections.abc import Callable

from shelfrank.eval.harness import PerQuery, Qrels, Run

# Their name for each of our metrics. "l2" restricts relevance to Exact. ranx "ndcg" is linear gain.
EQUIVALENT = {
    "ndcg10": "ndcg@10",
    "recall100": "recall@100",
    "exact10": "precision@10-l2",
    "mrr_exact": "mrr-l2",
    "recall100_exact": "recall@100-l2",
}
Evaluator = Callable[[dict, dict, list[str]], dict[str, dict[str, float]]]


def ranx_scores(qrels: dict, run: dict, metrics: list[str]) -> dict[str, dict[str, float]]:
    """Per-query ranx scores. Imported lazily because only the cross-check needs ranx."""
    import ranx

    result = ranx.Run(run)
    ranx.evaluate(ranx.Qrels(qrels), result, metrics, make_comparable=True)
    return {metric: dict(result.scores[metric]) for metric in metrics}


def _as_scores(run: Run, queries: Qrels) -> dict[str, dict[str, float]]:
    """Encode order as strictly decreasing scores, so ranx cannot break ties differently."""
    return {
        str(q): {str(p): float(len(run[q]) - i) for i, p in enumerate(run[q])} or {"none": 1.0}
        for q in queries
    }


def max_differences(
    run: Run, qrels: Qrels, ours: PerQuery, evaluator: Evaluator = ranx_scores
) -> dict[str, float]:
    """Largest absolute per-query difference per metric, where both are defined.

    Per query and not on the mean, because errors of opposite sign would cancel in a mean.
    """
    truth = {str(q): {str(p): g for p, g in grades.items()} for q, grades in qrels.items()}
    theirs = evaluator(truth, _as_scores(run, qrels), list(EQUIVALENT.values()))
    condensed = {q: [p for p in run[q] if p in qrels[q]] for q in qrels}
    theirs_condensed = evaluator(truth, _as_scores(condensed, qrels), ["ndcg@10"])["ndcg@10"]

    out = {}
    for name, other in EQUIVALENT.items():
        out[name] = max(
            (
                abs(row[name] - theirs[other][str(q)])
                for q, row in ours.items()
                if row[name] is not None
            ),
            default=0.0,
        )
    out["condensed_ndcg10"] = max(
        abs(row["condensed_ndcg10"] - theirs_condensed[str(q)]) for q, row in ours.items()
    )
    return out
