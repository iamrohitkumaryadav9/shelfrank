"""The sentence at the top, chosen by the verdict from texts fixed in docs/README_FRAMING.md."""

from collections.abc import Mapping
from typing import Any

from shelfrank.eval.claims import GAIN, LOSS, UNCLEAR
from shelfrank.eval.final_report import ALPHA, adjusted, ids_of, mean

A, B, C, D = "A", "B", "C", "D"


def main_rows(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], float, float]:
    """Primary row, secondary row and their Holm-adjusted p-values, on the primary queries."""
    rows = data["comparisons"]["primary"]["main"]
    ps = adjusted(rows)
    order = {r["baseline"]: i for i, r in enumerate(rows)}
    first, second = order["hybrid_rrf"], order["bm25_multi"]
    return rows[first], rows[second], ps[first], ps[second]


def holds(row: Mapping[str, Any], p: float) -> bool:
    """Whether a gain may be claimed."""
    return bool(row["verdict"] == GAIN and p < ALPHA)


def version(data: Mapping[str, Any]) -> str:
    """Which framing applies. A gain that does not survive Holm is not a gain."""
    row, _, p, _ = main_rows(data)
    if holds(row, p):
        return A
    if row["verdict"] == LOSS:
        return D
    return C if row["verdict"] == UNCLEAR else B


def _gap(c: Mapping[str, float]) -> str:
    return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def levels(data: Mapping[str, Any]) -> dict[str, float]:
    """Primary nDCG@10 of the three main systems."""
    ids = ids_of(data, "primary")
    return {s: mean(data, s, "ndcg10", ids) for s in ("bm25_multi", "hybrid_rrf", "ltr_full")}


def earning_groups(data: Mapping[str, Any]) -> list[str]:
    """Groups whose removal is a loss that survives Holm."""
    rows = data["comparisons"]["primary"]["ablation"]
    return [
        r["system"].removeprefix("without_")
        for r, p in zip(rows, adjusted(rows), strict=True)
        if r["system"].startswith("without_") and r["verdict"] == LOSS and p < ALPHA
    ]


def headline(data: Mapping[str, Any]) -> str:
    """The main result in words."""
    row, second, _, p2 = main_rows(data)
    at = levels(data)
    n = data["queries"]["primary"]
    features = len(data["shap_by_feature"])
    groups = ", ".join(earning_groups(data)) or "none on its own"
    kind = version(data)
    if kind == A:
        return (
            f"On WANDS, a LambdaMART ranker raised nDCG@10 from {at['hybrid_rrf']:.3f} (hybrid RRF) to "
            f"{at['ltr_full']:.3f}, a difference of {_gap(row['ndcg10'])} on {n} held-out queries, and the gain holds "
            f"under condensed evaluation ({_gap(row['condensed_ndcg10'])}). Tuned BM25 scores {at['bm25_multi']:.3f}. "
            f"Feature groups whose removal hurts: {groups}."
        )
    if kind == B:
        both = (
            f"Both beat tuned BM25 ({at['bm25_multi']:.3f})"
            if holds(second, p2)
            else f"Tuned BM25 scores {at['bm25_multi']:.3f}"
        )
        return (
            f"On WANDS, a LambdaMART ranker with {features} features did not clearly beat hybrid RRF, which needs no "
            f"training: {at['ltr_full']:.3f} against {at['hybrid_rrf']:.3f} nDCG@10, a difference of {_gap(row['ndcg10'])} "
            f"(condensed {_gap(row['condensed_ndcg10'])}) on {n} held-out queries. {both}. With this much training data the "
            "fusion of BM25 and a dense encoder is hard to improve on, and the ablation shows what each group of features added."
        )
    if kind == C:
        return (
            f"On WANDS, a LambdaMART ranker is {_gap(row['ndcg10'])} against hybrid RRF on standard nDCG@10 and "
            f"{_gap(row['condensed_ndcg10'])} on condensed nDCG@10, on {n} held-out queries ({at['ltr_full']:.3f} against "
            f"{at['hybrid_rrf']:.3f} nDCG@10). The direction depends on how unjudged products are treated, so no gain is claimed."
        )
    return (
        f"On WANDS, a LambdaMART ranker scored below hybrid RRF, {at['ltr_full']:.3f} against {at['hybrid_rrf']:.3f} "
        f"nDCG@10, {_gap(row['ndcg10'])} on {n} held-out queries. The simpler system is the better one here."
    )


def secondary(data: Mapping[str, Any]) -> str:
    """The BM25 comparison, never shown without the primary one."""
    _, row, _, p = main_rows(data)
    word = "a gain that holds under both evaluations" if holds(row, p) else str(row["verdict"])
    return (
        f"Secondary comparison, LTR against tuned BM25: {_gap(row['ndcg10'])} standard, "
        f"{_gap(row['condensed_ndcg10'])} condensed, Holm p {p:.4f}: {word}."
    )
