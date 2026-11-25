"""Render the audit results as Markdown, so that no figure in the report is typed by hand."""

from collections.abc import Iterable, Sequence
from typing import Any


def pct(value: float) -> str:
    """Format a share as a percentage."""
    return f"{100 * value:.1f}%"


def num(value: float) -> str:
    """Format a count or quantile, dropping a meaningless decimal."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Build a Markdown table, escaping the pipe that WANDS uses as a separator."""
    cells = [[str(c).replace("|", "\\|") for c in row] for row in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells])


def _quantile_rows(stats: dict[str, dict[str, float]]) -> list[list[str]]:
    keys = ("mean", "min", "p25", "p50", "p75", "p90", "p99", "max")
    return [[name, *(num(s[k]) for k in keys)] for name, s in stats.items()]


QUANTILE_HEADER = ["", "mean", "min", "p25", "median", "p75", "p90", "p99", "max"]


def _rows_section(a: dict[str, Any]) -> str:
    r = a["rows"]
    return f"""## 1. Rows and integrity

{table(["Item", "Count"], [
    ["Products", num(r["products"])],
    ["Queries", num(r["queries"])],
    ["Judgment rows", num(r["label_rows"])],
    ["Unique (query, product) pairs", num(r["unique_pairs"])],
    ["Repeated judgment rows", num(r["repeated_rows"])],
    ["Pairs whose repeats disagree", num(r["conflicting_pairs"])],
    ["Judgments pointing at an unknown query", num(r["label_queries_missing"])],
    ["Judgments pointing at an unknown product", num(r["label_products_missing"])],
    ["Products never judged for any query", num(r["products_never_judged"])],
])}

**The published count of {num(r["label_rows"])} judgments includes repeats.** One pair appears
{r["max_repeats"]} times. Repeats would double-count a product in the ideal ranking, so each pair is
reduced to a majority vote, and a tie takes the lower grade. Query ids run to {r["query_id_max"]}
with gaps, so they must not be used as positions.

Every disagreement and its resolution, by id:

{table(["Query id", "Product id", "Labels", "Resolved to", "Rule"], [
    [c["query_id"], c["product_id"], ", ".join(c["labels"]), c["resolved"], c["rule"]]
    for c in a["conflicts"]])}"""


def _labels_section(a: dict[str, Any]) -> str:
    s, c, p = a["labels"], a["ceilings_all"], a["ceilings_primary"]
    return f"""## 2. Labels (after removing repeats)

{table(["Label", "Pairs", "Share"], [[k, num(s["counts"][k]), pct(s["shares"][k])] for k in s["counts"]])}

Judgments per query:

{table(QUANTILE_HEADER, _quantile_rows({
    "All labels": s["judgments_per_query"], "Exact": s["exact_per_query"],
    "Partial": s["partial_per_query"], "Irrelevant": s["irrelevant_per_query"]}))}

- {s["queries_without_exact"]} queries have no Exact product at all and
  {s["queries_under_10_exact"]} have fewer than 10.
- {s["queries_without_relevant"]} query has neither Exact nor Partial, so its nDCG is undefined.
- {s["queries_without_irrelevant"]} queries have no Irrelevant judgment and
  {s["queries_under_10_judged"]} have fewer than 10 judgments in total.

### Score of a perfect ranker

{table(["Metric", "Averaged over", "All folds", "n", "Primary folds", "n"], [
    ["Exact@10", "queries with an Exact product", c["exact_at_10"], c["n_exact"],
     p["exact_at_10"], p["n_exact"]],
    ["Exact@10", "all queries (not used)", c["exact_at_10_all_queries"], c["n"],
     p["exact_at_10_all_queries"], p["n"]],
    ["Recall@100, Exact or Partial", "queries with a relevant product", c["recall_at_100"],
     c["n_relevant"], p["recall_at_100"], p["n_relevant"]],
    ["Recall@100, Exact only", "queries with an Exact product", c["recall_at_100_exact_only"],
     c["n_exact"], p["recall_at_100_exact_only"], p["n_exact"]],
])}

None of these can reach 1. Recall@100 is limited because
{c["queries_over_100_relevant"]} of {c["n"]} queries have more than 100 relevant products, and Exact@10
because many queries have few Exact products. Raw values must be read against these ceilings."""


def _classes_section(a: dict[str, Any]) -> str:
    q, p = a["query_class"], a["product_class"]
    absent = ", ".join(q["absent_from_catalogue"])
    return f"""## 3. Classes

{table(["", "Query class", "Product class"], [
    ["Missing", f"{q['missing']} queries", f"{num(p['missing'])} products ({pct(p['missing_share'])})"],
    ["Distinct", q["distinct"], p["distinct_atomic"]],
    ["Appearing once", q["singletons"], p["singletons"]],
])}

- {num(p["multi_class_products"])} products list several classes separated by a pipe. Raw cells have
  {p["distinct_raw"]} distinct values, atomic classes {p["distinct_atomic"]}.
- Only {q["at_least_10"]} query classes have 10 or more queries. On the primary folds it is
  {q["at_least_10_primary"]}, covering {q["queries_in_at_least_10_primary"]} of
  {a["ceilings_primary"]["n"]} queries. A per-class breakdown is therefore mostly uninterpretable.
- {len(q["absent_from_catalogue"])} query classes never occur as a product class ({absent}),
  affecting {q["queries_in_absent"]} queries. No model trained on product names can predict them.

Results are broken down by department instead. The mapping reads the catalogue only
([PROTOCOL.md](PROTOCOL.md) section 4).

{table(["Department", "All queries", "Primary queries", "Interpretable (n >= 10)"], [
    [d["department"], d["all"], d["primary"], "yes" if d["primary"] >= 10 else "no"]
    for d in a["departments"]])}

Largest classes:

{table(["Query class", "Queries", "Product class", "Products"], [
    [*x, *y] for x, y in zip(q["top"], p["top"], strict=True)])}

Words per query:

{table(QUANTILE_HEADER, _quantile_rows({"Query": q["words_per_query"]}))}"""


def _taxonomy_section(a: dict[str, Any], chosen: int) -> str:
    grid = a["taxonomy_grid"]
    pick = next(row for row in grid if row["min_support"] == chosen)
    return f"""## 4. Classifier taxonomy

{table(["Minimum products", "K", "Products covered", "Dev queries covered", "All queries covered (disclosure only)"], [
    [f"**{r['min_support']}**" if r["min_support"] == chosen else r["min_support"], r["k"],
     pct(r["product_coverage"]), pct(r["dev_query_coverage"]), pct(r["all_query_coverage"])] for r in grid])}

**Choice: classes with at least {chosen} products, K = {pick["k"]}, plus "other".** K is derived from a
support threshold and not from ranking classes by query coverage, because reading query_class on the
test folds to choose the label set would be leakage. {chosen} products gives every class enough names
to learn from. Lowering the threshold adds many sparse classes for little coverage, and raising it to 100
loses {pct(pick["dev_query_coverage"] - grid[-1]["dev_query_coverage"])} of dev queries. Products with
no class ({pct(a["product_class"]["missing_share"])}) cap product coverage for every K."""


def _hierarchy_section(a: dict[str, Any]) -> str:
    h = a["hierarchy"]
    return f"""## 5. Category hierarchy

{table(["Depth", *h["depth"]], [["Products", *(num(v) for v in h["depth"].values())]])}

{num(h["missing"])} products have no path. Top levels: {", ".join(f"{n} ({num(c)})" for n, c in h["top_levels"])}."""


def _features_section(a: dict[str, Any]) -> str:
    f = a["features"]
    s = f["slots"]
    return f"""## 6. product_features

The format is key:value pairs separated by a pipe. It is parsed by splitting on the first colon,
because values can contain colons. Numeric values have no spaces around the colon and text values do.

- {num(f["pairs"])} usable pairs over {num(f["distinct_keys"])} distinct keys,
  {num(f["keyless"])} pairs with an empty key (mostly yes/no) and {f["malformed"]} parts with no colon.
- {pct(f["products_with_multi_value_key_share"])} of products repeat a key with different values.
  These are product families: a key lists the available options.

{table(["Slot", "Products", "Share", "Multi-valued among them", "Distinct values", "Most common"], [
    [k, num(v["products"]), pct(v["share"]), pct(v["multi_valued_share"]), num(v["distinct_values"]),
     ", ".join(v["top_values"])] for k, v in s.items()])}

**Room is almost absent** ({pct(s["room"]["share"])} of products), so a room lexicon cannot come from
product_features. **Colour has {num(s["color"]["distinct_values"])} distinct values**, so it needs
mapping to base colours. A conflict must mean that no listed option matches the query, because
{pct(s["color"]["multi_valued_share"])} of products with a colour list more than one.

{table(["Key", "Products"], [[k, num(n)] for k, n in f["top_keys"]])}"""


def _missing_section(a: dict[str, Any]) -> str:
    m, d = a["missingness"], a["duplicates"]
    return f"""## 7. Missing values, text length and duplicates

{table(["Column", "Missing", "Share"], [
    [k, num(v), pct(m["shares"][k])] for k, v in m["columns"].items()])}

Rating fields missing together: {m["rating_fields_missing_together"]}. Average rating takes
{len(m["average_rating_values"])} values in half-star steps. {m["rated_but_zero_reviews"]} rated
products have zero reviews.

{table(QUANTILE_HEADER, _quantile_rows({**a["text"], "review_count": m["review_count"]}))}

Text rows are in words, over products that have the field.

{table(["Duplicate definition", "Products"], [
    ["Same name", num(d["rows_sharing_a_name"])],
    ["Same name and description", num(d["rows_sharing_name_and_description"])],
    ["Identical except id", num(d["rows_identical_except_id"])]])}

No products are removed: none is a full copy, and each has its own judgments."""


def _bias_section(a: dict[str, Any]) -> str:
    b = a["pool_bias"]
    mix = b["label_mix"]
    return f"""## 8. Pool bias towards Wayfair's query class

The WANDS paper (section 4.3) says the second annotation step used a proprietary query classifier and
collected products of the predicted class. The effect is measurable.

{table(["Measurement", "Value"], [
    ["A random (query, product) pair is judged", pct(b["judged_share_overall"])],
    ["A product in the query's class is judged (mean over queries)", pct(b["judged_share_in_query_class_mean"])],
    ["Judged pairs where the product is in the query's class", pct(b["judged_pairs_in_query_class"])],
    ["Exact pairs where the product is in the query's class", pct(b["exact_pairs_in_query_class"])]])}

{table(["Product in query class?", *mix["match"]], [
    ["Yes", *(pct(v) for v in mix["match"].values())],
    ["No", *(pct(v) for v in mix["no_match"].values())]])}

1. Class match separates Partial from Irrelevant far more than it marks Exact.
2. About {pct(1 - b["judged_share_in_query_class_mean"])} of in-class products are unjudged and score as
   irrelevant in the primary metric, so a system can be penalised for a good product outside the pool.
3. A class feature can gain by retrieving judged products. Judged@10 and the condensed metric must
   accompany every claim about class features."""


def _folds_section(a: dict[str, Any]) -> str:
    return f"""## 9. Folds

{table(["Fold", "Queries", "Distinct classes", "Median judged", "No Exact", "Exact@10 ceiling"], [
    [f"{r['fold']} (dev)" if r["fold"] == 0 else r["fold"], r["queries"], r["distinct_classes"],
     num(r["median_judged"]), r["queries_without_exact"], r["exact_at_10_ceiling"]] for r in a["folds"]])}

Stratification by query class is only partly possible: {a["query_class"]["at_least_5"]} classes have the
five queries needed to appear in every fold. The rest are pooled. The dev fold has the lowest Exact@10
ceiling, so dev values should not be compared directly with primary values."""


def render(audit: dict[str, Any], min_support: int, commit: str) -> str:
    """Assemble the report."""
    sections = [
        "# WANDS data audit\n\n"
        "Generated by `scripts/data_audit.py` from `results/data_audit.json`. Do not edit by hand.\n"
        f"Source commit `{commit}`. Only aggregates are shown, so no test query text appears here.",
        _rows_section(audit),
        _labels_section(audit),
        _classes_section(audit),
        _taxonomy_section(audit, min_support),
        _hierarchy_section(audit),
        _features_section(audit),
        _missing_section(audit),
        _bias_section(audit),
        _folds_section(audit),
    ]
    return "\n\n".join(sections) + "\n"
