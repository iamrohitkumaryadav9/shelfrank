"""Render results/final_run.json as docs/final_results.md and feed the README."""

from collections.abc import Mapping, Sequence
from typing import Any

from shelfrank.data.audit_report import table
from shelfrank.eval.claims import GAIN, holm
from shelfrank.stats.bootstrap import bootstrap_mean

NAMES = {
    "bm25_title": "BM25, title only",
    "bm25_multi": "BM25 multi-field, tuned",
    "dense": "Dense, bge-small off the shelf",
    "dense_esci": "Dense, bge-small fine-tuned on ESCI",
    "hybrid_rrf": "Hybrid RRF (BM25 + off-the-shelf)",
    "hybrid_rrf_esci": "Hybrid RRF (BM25 + fine-tuned)",
    "ltr_full": "**LTR, full (LambdaMART)**",
}
ABLATIONS = {
    "without_G1": "minus G1 lexical match",
    "without_G2": "minus G2 semantic and fusion",
    "without_G3": "minus G3 query class",
    "without_G4": "minus G4 attribute match",
    "without_G5": "minus G5 product priors",
    "without_G6": "minus G6 text shape",
    "only_G1": "G1 only (LTR machinery without new features)",
    "pointwise": "pointwise regression, same features",
    "other_variant": "other treatment of unjudged training rows",
    "ltr_other_encoder": "fine-tuned encoder in candidates and G2",
}
COLUMNS = {
    "ndcg10": "nDCG@10",
    "condensed_ndcg10": "Condensed nDCG@10",
    "judged10": "Judged@10",
    "exact10_share": "Exact@10 / ceiling",
    "exact10": "Exact@10",
    "mrr_exact": "MRR Exact",
    "recall100": "Recall@100",
}
ALPHA = 0.05
MIN_QUERIES = 10
Ids = Sequence[int]


def ids_of(data: Mapping[str, Any], split: str) -> list[int]:
    """Query ids of a split."""
    if split == "all":
        return sorted(data["primary_ids"] + data["dev_ids"])
    return list(data[f"{split}_ids"])


def values(data: Mapping[str, Any], system: str, metric: str, ids: Ids) -> list[float]:
    """Defined per-query values."""
    rows = data["per_query"][system]
    found = [rows[str(q)][metric] for q in ids]
    return [v for v in found if v is not None]


def mean(data: Mapping[str, Any], system: str, metric: str, ids: Ids) -> float:
    """Mean over the queries where the metric is defined."""
    found = values(data, system, metric, ids)
    return sum(found) / len(found) if found else float("nan")


def _level(data: Mapping[str, Any], system: str, metric: str, ids: Ids) -> str:
    interval = bootstrap_mean(values(data, system, metric, ids))
    return f"{interval.mean:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


def _gap(cell: Mapping[str, float]) -> str:
    return f"{cell['mean']:+.3f} [{cell['low']:+.3f}, {cell['high']:+.3f}]"


def main_table(data: Mapping[str, Any], split: str) -> str:
    """Every system, with the ceiling and n."""
    ids = ids_of(data, split)
    rows = []
    for name, label in NAMES.items():
        cells = [
            _level(data, name, m, ids)
            if m in ("ndcg10", "condensed_ndcg10")
            else f"{mean(data, name, m, ids):.3f}"
            for m in COLUMNS
        ]
        rows.append([label, *cells])
    top = data["ceilings"][split]
    rows.append(["*Perfect ranker (ceiling)*", *(f"{top[m]['mean']:.3f}" for m in COLUMNS)])
    rows.append(["*n queries*", *(top[m]["n"] for m in COLUMNS)])
    return table(["System", *COLUMNS.values()], rows)


def adjusted(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Holm-adjusted claim p-values of one family, in order."""
    out = holm({str(i): r["p"]["claim"] for i, r in enumerate(rows)})
    return [out[str(i)] for i in range(len(rows))]


def claimed(row: Mapping[str, Any], p: float) -> str:
    """Verdict in words. A claim needs the pool rule and the Holm-adjusted p."""
    if row["verdict"] == GAIN and p >= ALPHA:
        return "gain before correction, not after Holm"
    return str(row["verdict"])


def comparison_table(
    data: Mapping[str, Any], split: str, family: str, names: Mapping[str, str]
) -> str:
    """One family of comparisons."""
    rows = data["comparisons"][split][family]
    ps = adjusted(rows)
    body = [
        [
            f"{names.get(r['system'], r['system']).strip('*')} vs {names.get(r['baseline'], r['baseline']).strip('*')}",
            _gap(r["ndcg10"]),
            _gap(r["condensed_ndcg10"]),
            f"{p:.4f}",
            f"**{claimed(r, p)}**",
        ]
        for r, p in zip(rows, ps, strict=True)
    ]
    return table(["Comparison", "nDCG@10", "Condensed nDCG@10", "Holm p", "Verdict"], body)


def ablation_table(data: Mapping[str, Any], split: str) -> str:
    """Every ablation against the full ranker."""
    ids = ids_of(data, split)
    rows = data["comparisons"][split]["ablation"]
    ps = adjusted(rows)
    body = [
        [
            "full model",
            _level(data, "ltr_full", "ndcg10", ids),
            _level(data, "ltr_full", "condensed_ndcg10", ids),
            "",
            "",
            "",
            "",
        ]
    ]
    for r, p in zip(rows, ps, strict=True):
        body.append(
            [
                ABLATIONS[r["system"]],
                f"{mean(data, r['system'], 'ndcg10', ids):.3f}",
                f"{mean(data, r['system'], 'condensed_ndcg10', ids):.3f}",
                _gap(r["ndcg10"]),
                _gap(r["condensed_ndcg10"]),
                f"{p:.4f}",
                claimed(r, p),
            ]
        )
    header = [
        "Configuration",
        "nDCG@10",
        "Condensed",
        "Difference to full",
        "Difference, condensed",
        "Holm p",
        "Verdict",
    ]
    return table(header, body)


def groups_in(data: Mapping[str, Any], key: str, split: str) -> dict[str, list[int]]:
    """Queries by department or class, largest first."""
    out: dict[str, list[int]] = {}
    for q in ids_of(data, split):
        name = data[key].get(str(q))
        if name:
            out.setdefault(name, []).append(q)
    return dict(sorted(out.items(), key=lambda item: (-len(item[1]), item[0])))


def breakdown_table(data: Mapping[str, Any], key: str, split: str, minimum: int) -> str:
    """Levels per group. Groups below 10 queries are marked."""
    systems = ("bm25_multi", "hybrid_rrf", "ltr_full")
    rows = []
    for name, ids in groups_in(data, key, split).items():
        if len(ids) < minimum:
            continue
        flag = " *" if len(ids) < MIN_QUERIES else ""
        gap = [
            data["per_query"]["ltr_full"][str(q)]["ndcg10"]
            - data["per_query"]["hybrid_rrf"][str(q)]["ndcg10"]
            for q in ids
        ]
        interval = bootstrap_mean(gap)
        rows.append(
            [
                name + flag,
                len(ids),
                *(f"{mean(data, s, 'ndcg10', ids):.3f}" for s in systems),
                f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]",
                *(f"{10 * mean(data, s, 'exact10', ids):.1f}" for s in ("bm25_multi", "ltr_full")),
                *(f"{mean(data, s, 'judged10', ids):.3f}" for s in ("bm25_multi", "ltr_full")),
            ]
        )
    header = ["Group", "n", "BM25 nDCG@10", "RRF nDCG@10", "LTR nDCG@10", "LTR minus RRF"]
    header += ["Exact in top 10, BM25", "Exact in top 10, LTR", "Judged@10 BM25", "Judged@10 LTR"]
    return table(header, rows)


def importance_table(data: Mapping[str, Any]) -> str:
    """SHAP and gain per group."""
    gain: dict[str, float] = {}
    for column, share in data["importance"]["ltr_full"].items():
        group = column.split("_")[0].upper()
        gain[group] = gain.get(group, 0.0) + share
    shap = data["shap_by_group"]
    total = sum(shap.values()) or 1.0
    rows = [
        [g, f"{gain.get(g, 0):.0%}", f"{shap[g]:.3f}", f"{shap[g] / total:.0%}"]
        for g in sorted(shap)
    ]
    return table(
        ["Group", "Share of split gain", "Mean absolute SHAP in top 10", "Share of absolute SHAP"],
        rows,
    )


def feature_table(data: Mapping[str, Any], n: int = 15) -> str:
    """Features with the largest mean |SHAP|."""
    ranked = sorted(data["shap_by_feature"].items(), key=lambda x: -x[1])[:n]
    gain = data["importance"]["ltr_full"]
    return table(
        ["Feature", "Mean absolute SHAP", "Share of split gain"],
        [[f"`{c}`", f"{v:.3f}", f"{gain[c]:.1%}"] for c, v in ranked],
    )


def render(data: Mapping[str, Any]) -> str:
    """Build the page."""
    fixes = "; ".join(data["bug_fixes"]) or "none"
    trees = [t for fold in data["trees"]["ltr_full"] for t in fold]
    return f"""# Final results

Generated by `scripts/make_tables.py` from `results/final_run.json`. Do not edit by hand.

Settings were frozen in `configs/frozen.yaml` (tag `preregistered-v1`, commit `{data["tagged_commit"][:7]}`) before any query
outside the dev fold was read. The run was made once. Bug-fix reruns: {fixes}. Every query is ranked by models that did not train
on it (5 folds). Intervals are 95% bootstrap intervals over queries, differences are paired. {data["rows"]:,} candidate pairs,
{data["judged_share"]:.0%} judged. Largest difference from ranx: {data["ranx_max_difference"]:.1e}.

**A gain is claimed only if it is positive under both standard and condensed nDCG@10 with the 95% CI excluding 0, and its
Holm-adjusted p is below {ALPHA}.** Otherwise the verdict says what happened.

## Primary: {data["queries"]["primary"]} queries of folds 1 to 4

{main_table(data, "primary")}

### The pre-registered comparisons

{comparison_table(data, "primary", "main", NAMES)}

### Retrieval comparisons

{comparison_table(data, "primary", "retrieval", NAMES)}

### Ablations

{ablation_table(data, "primary")}

## Which features the model uses

Exact TreeSHAP values of the out-of-fold models, over each query's top 10.

{importance_table(data)}

{feature_table(data)}

Trees per model (5 folds, 3 seeds): {min(trees)} to {max(trees)}, median {sorted(trees)[len(trees) // 2]}.

## By department

Departments come from the class of the query ([PROTOCOL.md](PROTOCOL.md) section 4). * marks fewer than {MIN_QUERIES} queries, too
few to interpret. "Exact in top 10" counts queries that have an Exact product.

{breakdown_table(data, "departments", "primary", 1)}

## By class, with at least {MIN_QUERIES} primary queries

{breakdown_table(data, "query_class", "primary", MIN_QUERIES)}

## All {data["queries"]["all"]} queries, which includes the dev fold

Settings were chosen on {data["queries"]["dev"]} of these, so this is slightly optimistic. It is not the headline.

{main_table(data, "all")}

{comparison_table(data, "all", "main", NAMES)}
"""
