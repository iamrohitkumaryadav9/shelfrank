"""Render results/retrieval_<split>.json as Markdown."""

from typing import Any

from shelfrank.data.audit_report import table
from shelfrank.stats.bootstrap import paired_difference

LABELS = {
    "bm25_title": "BM25, title only",
    "bm25_title_desc": "BM25, title + description",
    "bm25_multi_equal": "BM25 multi-field, equal weights",
    "bm25_multi": "BM25 multi-field, tuned (baseline)",
    "dense_bge_small": "Dense, bge-small off the shelf",
    "hybrid_rrf": "Hybrid, RRF of tuned BM25 and dense",
    "union_multi_dense": "Union of tuned BM25 and dense",
}
COLUMNS = {
    "ndcg10": "nDCG@10",
    "condensed_ndcg10": "Condensed nDCG@10",
    "judged10": "Judged@10",
    "recall100": "Recall@100",
    "exact10": "Exact@10",
    "exact10_share": "Exact@10 / ceiling",
    "mrr_exact": "MRR Exact",
    "recall100_exact": "Exact-only Recall@100",
}


def _f(value: float) -> str:
    return f"{value:.3f}"


def _interval(row: dict[str, float]) -> str:
    return f"{row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]"


def main_table(data: dict[str, Any]) -> str:
    """Every system and metric, with the ceiling row and n the protocol requires."""
    rows = [
        [LABELS[name], *(_f(system["summary"][m]["mean"]) for m in COLUMNS)]
        for name, system in data["systems"].items()
    ]
    ceiling = ["*Perfect ranker (ceiling)*", *(_f(data["ceilings"][m]["mean"]) for m in COLUMNS)]
    counts = ["*n queries*", *(data["ceilings"][m]["n"] for m in COLUMNS)]
    return table(["System", *COLUMNS.values()], [*rows, ceiling, counts])


def ci_table(data: dict[str, Any]) -> str:
    """Level and paired difference for the primary metric and its condensed form."""
    rows = []
    for name, system in data["systems"].items():
        level = system["summary"]["ndcg10"]
        versus = system["versus_baseline"]
        is_baseline = name == data["baseline"]
        rows.append(
            [
                LABELS[name],
                f"{level['mean']:.3f} [{level['low']:.3f}, {level['high']:.3f}]",
                "baseline" if is_baseline else _interval(versus["ndcg10"]),
                "baseline" if is_baseline else _interval(versus["condensed_ndcg10"]),
            ]
        )
    header = ["System", "nDCG@10 [95% CI]", "Difference, nDCG@10", "Difference, condensed nDCG@10"]
    return table(header, rows)


def candidate_table(data: dict[str, Any]) -> str:
    """Candidate sets as ceilings for the ranker."""
    rows = [
        [
            LABELS[name],
            f"{c['mean_size']:.0f}",
            _f(c["recall"]),
            _f(c["recall_exact"]),
            _f(c["judged_share"]),
            _f(c["oracle_ndcg10"]),
            _f(c["oracle_exact10_share"]),
        ]
        for name, c in data["candidates"].items()
    ]
    header = [
        "Candidates (top 100 per generator)",
        "Mean size",
        "Recall",
        "Exact-only recall",
        "Judged share",
        "Oracle nDCG@10",
        "Oracle Exact@10 / ceiling",
    ]
    return table(header, rows)


def tuning_table(tuning: dict[str, Any]) -> str:
    """In-sample and held-out scores side by side, so tuning optimism is visible."""
    weights = ", ".join(f"{k} {v:g}" for k, v in tuning["weights"].items())
    rows = [
        ["Equal weights", _f(tuning["dev_ndcg10_equal_weights"])],
        ["Tuned, scored on the queries it was tuned on", _f(tuning["dev_ndcg10_tuned_in_sample"])],
        ["Tuned on 3/4 of dev, scored on the rest", _f(tuning["dev_ndcg10_held_out"])],
    ]
    lines = [table(["Dev nDCG@10", "Value"], rows), f"Weights: {weights}."]
    lines.append(f"Configurations tried: {tuning['configs_tried']}.")
    return "\n\n".join(lines)


def held_out_table(data: dict[str, Any], tuning: dict[str, Any]) -> str:
    """Whether tuning helps on queries it did not see.

    The in-sample difference cannot answer this, because the weights were chosen to maximise it.
    """
    held = tuning["held_out_per_query"]
    queries = sorted(held)
    title = data["systems"]["bm25_title"]["per_query"]
    others = {
        "Title only": [title[q]["ndcg10"] for q in queries],
        "Equal weights": [tuning["equal_weights_per_query"][q] for q in queries],
    }
    rows = []
    for name, values in others.items():
        out = paired_difference([held[q] for q in queries], values)
        interval = {"mean": out.mean, "low": out.low, "high": out.high}
        rows.append([name, _interval(interval), "yes" if out.excludes_zero() else "no"])
    return table(["Held-out tuned BM25 minus", "nDCG@10 [95% CI]", "Excludes 0"], rows)


def sign_changes(data: dict[str, Any]) -> list[str]:
    """Systems that lose to the baseline on one nDCG and win on the other."""
    return [
        name
        for name, system in data["systems"].items()
        if system["versus_baseline"]["ndcg10"]["mean"]
        * system["versus_baseline"]["condensed_ndcg10"]["mean"]
        < 0
    ]


def pool_note(data: dict[str, Any]) -> str:
    """Name the systems whose standing depends on how unjudged products are treated."""
    flipped = sign_changes(data)
    if not flipped:
        return "No system changes sign between the two."
    lines = []
    for name in flipped:
        judged = data["systems"][name]["summary"]["judged10"]["mean"]
        base = data["systems"][data["baseline"]]["summary"]["judged10"]["mean"]
        lines.append(
            f"- **{LABELS[name]}** changes sign between standard and condensed nDCG@10. Its top 10 is "
            f"{_f(judged)} judged, against {_f(base)} for the baseline, so part of its standard "
            "deficit comes from unjudged products scored as irrelevant."
        )
    return "\n".join(lines)


def render(data: dict[str, Any], tuning: dict[str, Any]) -> str:
    """Assemble the page."""
    worst = max(v for d in data["ranx_max_difference"].values() for v in d.values())
    return f"""# Retrieval baselines, {data["split"]} split

Generated by `scripts/make_tables.py`. Do not edit by hand.

**These are development numbers on {data["n_queries"]} queries.** The BM25 field weights were chosen on the
same queries, so rows that use tuned BM25 are optimistic here. The result that counts comes from the
primary folds after pre-registration. Ceilings and n follow [PROTOCOL.md](PROTOCOL.md) section 3.

## Systems

{main_table(data)}

Rankings are {data["run_depth"]} deep, so MRR is measured within that depth.

## Confidence intervals

95% percentile intervals over queries. Differences are paired, against tuned multi-field BM25.

{ci_table(data)}

{pool_note(data)}

## BM25 field weights

{tuning_table(tuning)}

{held_out_table(data, tuning)}

## Candidate sets

Recall here is over the whole set, so a larger set has a higher limit. The oracle columns show the best
top 10 that any re-ranker could build from the set.

{candidate_table(data)}

## Metric check

Largest per-query difference between our metrics and ranx, over all systems: {worst:.1e}.

## Published baseline

The paper plots BM25 at roughly 0.72 nDCG@10 with unstated settings. Ours over the same two fields (title
and description) is {_f(data["systems"]["bm25_title_desc"]["summary"]["ndcg10"]["mean"])} on these
{data["n_queries"]} dev queries, which are not the 480 queries in the paper. The comparison on the same
queries waits for the final run. No reproduction is claimed.
"""
