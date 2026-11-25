"""docs/RESULTS.md: results in business terms, failures, and the plan for a live test."""

from collections.abc import Mapping
from typing import Any

from shelfrank.data.audit_report import table
from shelfrank.eval import final_report as fr
from shelfrank.eval.headline import earning_groups, headline, main_rows, secondary, version
from shelfrank.stats.bootstrap import bootstrap_mean

GROUP_NAMES = {
    "G1": "lexical match",
    "G2": "semantic similarity and fusion",
    "G3": "query class",
    "G4": "attribute match",
    "G5": "product priors",
    "G6": "text shape",
}


def _gap(c: Mapping[str, float]) -> str:
    return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def exact_sentences(data: Mapping[str, Any], n: int = 6) -> str:
    """Exact results in the top 10 per department, as sentences."""
    lines = []
    for name, ids in fr.groups_in(data, "departments", "primary").items():
        defined = fr.values(data, "ltr_full", "exact10", ids)
        if len(ids) < fr.MIN_QUERIES or not defined or len(lines) == n:
            continue
        ltr, rrf, bm25 = (
            10 * fr.mean(data, s, "exact10", ids) for s in ("ltr_full", "hybrid_rrf", "bm25_multi")
        )
        lines.append(
            f"- In **{name}** ({len(defined)} queries with an Exact product), {ltr:.1f} of the top 10 are exact matches, against "
            f"{rrf:.1f} for hybrid RRF and {bm25:.1f} for BM25."
        )
    ids = fr.ids_of(data, "primary")
    share = {
        s: fr.mean(data, s, "exact10_share", ids) for s in ("ltr_full", "hybrid_rrf", "bm25_multi")
    }
    lines.append(
        f"- Over all queries with an Exact product, LTR reaches {share['ltr_full']:.0%} of the achievable Exact@10, hybrid RRF "
        f"{share['hybrid_rrf']:.0%} and BM25 {share['bm25_multi']:.0%}."
    )
    return "\n".join(lines)


def leak_section(checks: Mapping[str, Any]) -> str:
    """Why the gain is believed, and what it depends on."""
    sizes = [
        [f"{n} queries", _gap(c["ndcg10"]), _gap(c["condensed_ndcg10"]), c["verdict"]]
        for n, c in checks["training_size"].items()
    ]
    folds = [
        [
            ("fold 0 (dev)" if f == "0" else f"fold {f}"),
            c["n"],
            _gap(c["ndcg10"]),
            _gap(c["condensed_ndcg10"]),
        ]
        for f, c in checks["by_fold"].items()
    ]
    thirds = [
        [
            f"{c['low']:.2f} to {c['high']:.2f}",
            c["n"],
            _gap(c["ndcg10"]),
            _gap(c["condensed_ndcg10"]),
        ]
        for c in checks["by_similarity_to_training"]
    ]
    shuffled = checks["shuffled_labels"]
    without = checks["gain_without_those"]
    strong, pair = checks["against_fine_tuned_rrf"], checks["fine_tuned_ltr_against_fine_tuned_rrf"]
    return f"""The primary gain is larger than the dev comparisons suggested, so it was checked before it was believed. These checks
came after the result and changed nothing.

- **Repeating the run** gives the same numbers (largest per-query difference {checks["reproduces_final_run"]:.1e}).
- **Labels shuffled within each query:** LTR falls to {_gap(shuffled["ndcg10"])} against hybrid RRF. Features carry no shortcut to
  the labels.
- **Training size.** The gain is there with few queries and grows:

{table(["Trained on", "LTR minus RRF, nDCG@10", "Condensed", "Verdict"], sizes)}

- **Folds.** Every fold outside dev gains more than dev:

{table(["Fold", "n", "LTR minus RRF, nDCG@10", "Condensed"], folds)}

  Dev is the weakest fold. One reason is known: BM25's field weights were tuned on it, so the BM25 and RRF baselines are
  optimistic there and not elsewhere. **The condensed difference on dev is negative**, so part of what LTR learns on other
  queries does not help on those 96.
- **Similar training queries.** Primary queries are split into thirds by their largest word overlap with a query of another
  fold:

{table(["Overlap (Jaccard)", "n", "LTR minus RRF, nDCG@10", "Condensed"], thirds)}

  Without the {checks["queries_with_similarity_at_least_half"]} queries whose overlap is 0.5 or more, the gain is {_gap(without["ndcg10"])} ({_gap(without["condensed_ndcg10"])}
  condensed). The correlation of gain and overlap is {checks["correlation_gain_similarity"]:+.2f}.
- **A stronger baseline** (not pre-registered). Hybrid RRF with the fine-tuned encoder is better than the pre-registered baseline.
  Against it, the frozen LTR is {_gap(strong["ndcg10"])} standard and {_gap(strong["condensed_ndcg10"])} condensed
  ({strong["verdict"]}). LTR with the fine-tuned encoder against it: {_gap(pair["ndcg10"])} and {_gap(pair["condensed_ndcg10"])}
  ({pair["verdict"]})."""


def attribute_section(data: Mapping[str, Any], cases: Mapping[str, Any]) -> str:
    """Queries by the number of attribute values the slot student reads."""
    counts = cases["slot_values"]
    rows = []
    for label, keep in (
        ("none", lambda n: n == 0),
        ("one", lambda n: n == 1),
        ("two or more", lambda n: n >= 2),
    ):
        ids = [q for q in fr.ids_of(data, "primary") if keep(counts[str(q)])]
        if not ids:
            continue
        gap = bootstrap_mean(
            [
                data["per_query"]["ltr_full"][str(q)]["ndcg10"]
                - data["per_query"]["hybrid_rrf"][str(q)]["ndcg10"]
                for q in ids
            ]
        )
        drop = bootstrap_mean(
            [
                data["per_query"]["without_G4"][str(q)]["ndcg10"]
                - data["per_query"]["ltr_full"][str(q)]["ndcg10"]
                for q in ids
            ]
        )
        flag = " *" if len(ids) < fr.MIN_QUERIES else ""
        rows.append(
            [
                label + flag,
                len(ids),
                *(
                    f"{fr.mean(data, s, 'ndcg10', ids):.3f}"
                    for s in ("bm25_multi", "hybrid_rrf", "ltr_full")
                ),
                f"{gap.mean:+.3f} [{gap.low:+.3f}, {gap.high:+.3f}]",
                f"{drop.mean:+.3f} [{drop.low:+.3f}, {drop.high:+.3f}]",
            ]
        )
    return table(
        [
            "Attribute values in the query",
            "n",
            "BM25",
            "RRF",
            "LTR",
            "LTR minus RRF",
            "LTR without G4 minus LTR",
        ],
        rows,
    )


def failure_section(cases: Mapping[str, Any], notes: Mapping[str, Mapping[str, str]]) -> str:
    """The queries where LTR loses most to BM25, each with a written diagnosis."""
    rows = []
    for c in cases["cases"]:
        note = notes[str(c["query_id"])]
        rows.append(
            [
                c["query"],
                f"{c['ndcg10']['bm25_multi']:.2f}",
                f"{c['ndcg10']['ltr_full']:.2f}",
                f"{c['condensed']['bm25_multi']:.2f} / {c['condensed']['ltr_full']:.2f}",
                f"{c['judged10']['ltr_full']:.1f}",
                note["cause"],
                note["diagnosis"],
                note["would_fix"],
            ]
        )
    header = [
        "Query",
        "BM25",
        "LTR",
        "Condensed BM25 / LTR",
        "LTR judged@10",
        "Cause",
        "Diagnosis",
        "What would fix it",
    ]
    return table(header, rows)


def render(
    data: Mapping[str, Any],
    checks: Mapping[str, Any],
    cases: Mapping[str, Any],
    notes: Mapping[str, Mapping[str, str]],
    similar: Mapping[str, Any],
    suggest: Mapping[str, Any],
    has_rewriting: bool,
) -> str:
    """Build the page."""
    row, _, _, _ = main_rows(data)
    causes = [notes[str(c["query_id"])]["cause"] for c in cases["cases"]]
    unjudged = sum("unjudged" in c for c in causes)
    groups = [f"{g} ({GROUP_NAMES[g]})" for g in earning_groups(data)]
    earned = " and ".join(groups) if groups else "no group on its own"
    ids = fr.ids_of(data, "primary")
    judged = {s: fr.mean(data, s, "judged10", ids) for s in ("hybrid_rrf", "ltr_full")}
    rec = similar["differences"]
    rec_words = "; ".join(
        f"{d['system']} against {d['baseline']}: {_gap(d['ndcg10'])} nDCG@10, {d['verdict']}"
        for d in rec
    )
    first = suggest["prefix_words"]["1"]
    rewriting = (
        "See [rewriting.md](rewriting.md) for gains, harm, oracle bounds, cost and latency."
        if has_rewriting
        else "Not evaluated yet."
    )
    standard, condensed = row["ndcg10"]["mean"], row["condensed_ndcg10"]["mean"]
    return f"""# Results

Generated by `scripts/make_tables.py`. Numbers come from `results/*.json`. Full tables: [final_results.md](final_results.md). The wording of
the main result was fixed for every possible outcome before the run ([README_FRAMING.md](README_FRAMING.md), version {version(data)}).

## 1. The main result

{headline(data)}

{secondary(data)}

{fr.main_table(data, "primary")}

{fr.comparison_table(data, "primary", "main", fr.NAMES)}

**How much of it could be pool bias.** The standard gain is {standard:+.3f} and the condensed gain {condensed:+.3f}. The share of judged
products in the top 10 rises from {judged["hybrid_rrf"]:.3f} to {judged["ltr_full"]:.3f}. The model was trained with unjudged candidates as
irrelevant, so it learns partly which products annotators looked at. The condensed gain, which ignores unjudged products, is the part
that this cannot explain. The smaller number is the safer one to quote.

## 2. Is it real?

{leak_section(checks)}

## 3. Which features earned it

Removing a group is a loss that survives Holm for {earned}. The other groups do not clear that bar one at a time. G4, the attribute match this
project set out to test, has an unadjusted interval below 0 and a Holm-adjusted p above 0.05: it is **not** a claimed effect.

{fr.ablation_table(data, "primary")}

Two more findings from the ablation:

- Training only on judged pairs is much worse under standard evaluation and better under condensed. Which one is right depends on how
  unjudged products are treated, so the variant was fixed on dev and the other is reported here.
- A pointwise regressor with the same features is not clearly different from LambdaMART.

## 4. In business terms

Exact@10 counts the products a shopper asked for among the first ten.

{exact_sentences(data)}

{fr.breakdown_table(data, "departments", "primary", fr.MIN_QUERIES)}

## 5. Where it fails

### Queries that name attributes

{attribute_section(data, cases)}

Counts come from the slot student (vs LLM silver labels; human verification pending). * marks fewer than {fr.MIN_QUERIES} queries.

### The {cases["n"]} primary queries where LTR loses most to BM25

In {unjudged} of them the cause includes unjudged products that count as wrong, and condensed evaluation shows a smaller loss or a gain.
The rest are real errors: numbers read without their unit, shapes and second colours that have no slot, attribute matches on the wrong
product type, and near-identical product types.

{failure_section(cases, notes)}

## 6. The other parts

- **Query understanding:** [class_dev.md](class_dev.md) (with the teacher as a ceiling on all 480 queries) and [slots_dev.md](slots_dev.md).
- **Fine-tuned encoder:** [dense_dev.md](dense_dev.md) for dev, and the retrieval table in [final_results.md](final_results.md). By the dev rule it was
  not used in the ranker. On the primary queries it is better than off-the-shelf under both evaluations. The rule was written to
  avoid choosing on noise, and here it left a gain unused.
- **Rewriting:** {rewriting}
- **Search suggestions:** MRR of the next word after one typed word is {first["mrr"]["ngram"]["mean"]:.3f}, against {first["mrr"]["frequency_only"]["mean"]:.3f} for the most
  frequent words. No popularity data exists, and the class prior did not help ([suggest.md](suggest.md)).
- **More like this:** {rec_words} ([similar.md](similar.md)).
- **Personalization:** not evaluated, because the data has no users ([PERSONALIZATION.md](PERSONALIZATION.md)).

## 7. Plan for an A/B test

**This is a plan and not a result.**

| | |
| --- | --- |
| Hypothesis | Ranking with LTR instead of hybrid RRF raises add-to-cart per search |
| Unit of randomization | Shopper, so that one person sees one ranking |
| Primary metric | Searches with an add-to-cart from the results, per search |
| Guardrails | Search exit rate, reformulation rate, p95 latency of search, zero-result rate, share of results from one class in the top 10 |
| Segments fixed in advance | Departments above, and queries with and without attributes |
| Rough sample size | For a rate of 5% and a relative change of 2%, about 16 x 0.05 x 0.95 / 0.001^2 = 760,000 searches per arm (80% power, 5% level). **The 5% and the 2% are assumptions for the arithmetic and not data.** Searches of one shopper are correlated, so the real number is larger |
| Duration | At least two full weeks, whatever the sample size, because weekdays and weekends differ |
| Roll back if | the exit rate or p95 latency rises beyond its margin with a 95% interval that excludes it, or the primary metric is negative with an interval that excludes 0 |
| Before the test | Interleave LTR and RRF on a small share of traffic, which needs far fewer searches to see a preference |

## 8. What offline nDCG cannot say

nDCG here rewards agreement with annotators who saw a product's text and not a shopper with a room, a budget and a delivery date. It
treats the 384 queries as equally frequent, when a few head queries carry most traffic. It knows nothing of price, stock, delivery or images,
and nothing of what people do after they click. It is computed on a pool that earlier systems helped to choose. A gain of this size says that
the ranking is closer to what annotators called relevant. Whether more furniture is bought is a question for a test.
"""
