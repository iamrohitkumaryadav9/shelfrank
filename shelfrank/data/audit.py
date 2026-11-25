"""Compute the facts about WANDS that the evaluation design depends on.

Everything here is an aggregate. No query text is emitted, so the audit can cover all 480 queries
without showing test queries to whoever designs the features.
"""

from collections import Counter
from typing import Any

import pandas as pd

from shelfrank.data.product_features import parse_features, slot_of_key
from shelfrank.data.taxonomy import build_taxonomy, class_support, product_coverage, query_coverage
from shelfrank.data.wands import GAINS, split_classes, split_hierarchy
from shelfrank.eval.splits import Folds

SUPPORT_GRID = (5, 10, 20, 30, 50, 100)
SLOTS = ("color", "material", "style", "room")


def _pct(series: pd.Series) -> dict[str, float]:
    """Summarise a distribution with the quantiles used throughout the report."""
    quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "mean": round(float(series.mean()), 2),
        "min": float(series.min()),
        **{f"p{round(q * 100)}": float(v) for q, v in quantiles.items()},
        "max": float(series.max()),
    }


def _share(mask: pd.Series) -> float:
    """Mean of a boolean mask, rounded for the report."""
    return round(float(mask.mean()), 4)


def row_counts(
    queries: pd.DataFrame, products: pd.DataFrame, labels: pd.DataFrame, judged: pd.DataFrame
) -> dict[str, Any]:
    """Row counts, referential integrity and repeated judgments."""
    repeats = labels.groupby(["query_id", "product_id"])["label"].agg(["size", "nunique"])
    return {
        "queries": len(queries),
        "products": len(products),
        "label_rows": len(labels),
        "unique_pairs": len(judged),
        "repeated_rows": len(labels) - len(judged),
        "max_repeats": int(repeats["size"].max()),
        "conflicting_pairs": int((repeats["nunique"] > 1).sum()),
        "label_queries_missing": len(set(labels["query_id"]) - set(queries["query_id"])),
        "label_products_missing": len(set(labels["product_id"]) - set(products["product_id"])),
        "products_never_judged": len(set(products["product_id"]) - set(labels["product_id"])),
        "query_id_max": int(queries["query_id"].max()),
        "duplicate_query_text": int(queries["query"].str.lower().str.strip().duplicated().sum()),
    }


def grade_table(judged: pd.DataFrame, query_ids: list[int]) -> pd.DataFrame:
    """Judgments per query and label, with a zero row for any query that has none."""
    names = {grade: label for label, grade in GAINS.items()}
    table = judged.assign(label=judged["grade"].map(names))
    counts = table.groupby(["query_id", "label"]).size().unstack(fill_value=0)
    return counts.reindex(index=query_ids, columns=list(GAINS), fill_value=0)


def label_stats(judged: pd.DataFrame, query_ids: list[int]) -> dict[str, Any]:
    """Label distribution overall and per query."""
    table = grade_table(judged, query_ids)
    totals = table.sum()
    return {
        "counts": {k: int(v) for k, v in totals.items()},
        "shares": {k: round(float(v) / float(totals.sum()), 4) for k, v in totals.items()},
        "judgments_per_query": _pct(table.sum(axis=1)),
        "exact_per_query": _pct(table["Exact"]),
        "partial_per_query": _pct(table["Partial"]),
        "irrelevant_per_query": _pct(table["Irrelevant"]),
        "queries_without_exact": int((table["Exact"] == 0).sum()),
        "queries_under_10_exact": int((table["Exact"] < 10).sum()),
        "queries_without_relevant": int(((table["Exact"] + table["Partial"]) == 0).sum()),
        "queries_without_irrelevant": int((table["Irrelevant"] == 0).sum()),
        "queries_under_10_judged": int((table.sum(axis=1) < 10).sum()),
    }


def metric_ceilings(judged: pd.DataFrame, query_ids: list[int]) -> dict[str, Any]:
    """Best possible score of a perfect ranker, which bounds how each metric can be read."""
    table = grade_table(judged, query_ids)
    exact, relevant = table["Exact"], table["Exact"] + table["Partial"]
    has_exact, has_relevant = exact > 0, relevant > 0
    ratio = relevant[has_relevant].clip(upper=100) / relevant[has_relevant]
    exact_ratio = exact[has_exact].clip(upper=100) / exact[has_exact]
    return {
        "n": len(query_ids),
        "exact_at_10": round(float((exact.clip(upper=10) / 10).mean()), 4),
        "mrr_exact": _share(has_exact),
        "recall_at_100": round(float(ratio.mean()), 4),
        "recall_at_100_exact_only": round(float(exact_ratio.mean()), 4),
        "queries_over_100_relevant": int((relevant > 100).sum()),
        "queries_over_100_exact": int((exact > 100).sum()),
    }


def query_class_stats(queries: pd.DataFrame, catalogue: set[str], folds: Folds) -> dict[str, Any]:
    """Shape of the reference classes."""
    counts = queries["query_class"].value_counts()
    primary = queries[queries["query_id"].isin(folds.primary)]["query_class"].value_counts()
    absent = set(counts.index) - catalogue
    top = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[:15]
    return {
        "missing": int(queries["query_class"].isna().sum()),
        "distinct": len(counts),
        "singletons": int((counts == 1).sum()),
        "at_least_5": int((counts >= 5).sum()),
        "at_least_10": int((counts >= 10).sum()),
        "at_least_10_primary": int((primary >= 10).sum()),
        "queries_in_at_least_10_primary": int(primary[primary >= 10].sum()),
        "top": [[str(name), int(count)] for name, count in top],
        "absent_from_catalogue": sorted(absent),
        "queries_in_absent": int(queries["query_class"].isin(absent).sum()),
        "words_per_query": _pct(queries["query"].str.split().map(len)),
    }


def product_class_stats(products: pd.DataFrame) -> dict[str, Any]:
    """Size and tail of the product class vocabulary."""
    support = class_support(products["product_class"])
    n_classes = products["product_class"].map(lambda cell: len(split_classes(cell)))
    return {
        "missing": int((n_classes == 0).sum()),
        "missing_share": _share(n_classes == 0),
        "multi_class_products": int((n_classes > 1).sum()),
        "distinct_atomic": len(support),
        "distinct_raw": int(products["product_class"].nunique()),
        "singletons": int((support == 1).sum()),
        "under_10": int((support < 10).sum()),
        "under_30": int((support < 30).sum()),
        "top": [[str(name), int(count)] for name, count in support.head(15).items()],
    }


def taxonomy_grid(
    products: pd.DataFrame, queries: pd.DataFrame, folds: Folds
) -> list[dict[str, Any]]:
    """Coverage by support threshold.

    Dev-fold query coverage may inform the choice. Coverage on all queries is disclosed next to
    it and must not.
    """
    support = class_support(products["product_class"])
    dev = queries[queries["query_id"].isin(folds.dev)]["query_class"]
    rows = []
    for threshold in SUPPORT_GRID:
        kept = build_taxonomy(support, threshold)
        rows.append(
            {
                "min_support": threshold,
                "k": len(kept),
                "product_coverage": round(product_coverage(products["product_class"], kept), 4),
                "dev_query_coverage": round(query_coverage(dev, kept), 4),
                "all_query_coverage": round(query_coverage(queries["query_class"], kept), 4),
            }
        )
    return rows


def hierarchy_stats(products: pd.DataFrame) -> dict[str, Any]:
    """Depth and top level of the category path."""
    paths = products["category_hierarchy"].map(split_hierarchy)
    depth = paths.map(len)
    tops = Counter(path[0] for path in paths if path)
    return {
        "missing": int((depth == 0).sum()),
        "depth": {str(k): int(v) for k, v in depth[depth > 0].value_counts().sort_index().items()},
        "top_levels": [[name, count] for name, count in tops.most_common(10)],
    }


def feature_stats(products: pd.DataFrame) -> dict[str, Any]:
    """Format of product_features and slot coverage."""
    parsed = products["product_features"].map(parse_features)
    key_products: Counter[str] = Counter()
    pairs = keyless = malformed = multi_value = 0
    slot_values: dict[str, list[set[str]]] = {slot: [] for slot in SLOTS}
    slot_counts: dict[str, Counter[str]] = {slot: Counter() for slot in SLOTS}
    for item in parsed:
        grouped = item.by_key()
        pairs += len(item.pairs)
        keyless += len(item.keyless)
        malformed += len(item.malformed)
        multi_value += any(len(values) > 1 for values in grouped.values())
        key_products.update(grouped.keys())
        per_slot: dict[str, set[str]] = {slot: set() for slot in SLOTS}
        for key, values in grouped.items():
            slot = slot_of_key(key)
            if slot:
                per_slot[slot].update(values)
        for slot, values in per_slot.items():
            if values:
                slot_values[slot].append(values)
                slot_counts[slot].update(values)
    n = len(products)
    slots = {}
    for slot in SLOTS:
        sizes = [len(values) for values in slot_values[slot]]
        slots[slot] = {
            "products": len(sizes),
            "share": round(len(sizes) / n, 4),
            "multi_valued": sum(size > 1 for size in sizes),
            "multi_valued_share": round(sum(s > 1 for s in sizes) / len(sizes), 4) if sizes else 0,
            "distinct_values": len(slot_counts[slot]),
            "top_values": [value for value, _ in slot_counts[slot].most_common(8)],
        }
    return {
        "pairs": pairs,
        "keyless": keyless,
        "malformed": malformed,
        "distinct_keys": len(key_products),
        "products_with_multi_value_key": multi_value,
        "products_with_multi_value_key_share": round(multi_value / n, 4),
        "top_keys": [[key, count] for key, count in key_products.most_common(20)],
        "slots": slots,
    }


def missingness(products: pd.DataFrame) -> dict[str, Any]:
    """Missing values per column, and whether the rating fields go missing together."""
    rating = products[["rating_count", "average_rating", "review_count"]].isna()
    no_reviews = products["review_count"] == 0
    return {
        "columns": {c: int(products[c].isna().sum()) for c in products.columns},
        "shares": {c: _share(products[c].isna()) for c in products.columns},
        "rating_fields_missing_together": bool((rating.sum(axis=1).isin([0, 3])).all()),
        "rated_but_zero_reviews": int(no_reviews.sum()),
        "average_rating_values": sorted(
            float(v) for v in products["average_rating"].dropna().unique()
        ),
        "review_count": _pct(products["review_count"].dropna()),
    }


def text_stats(products: pd.DataFrame) -> dict[str, Any]:
    """Word counts per text field, over products that have the field."""
    fields = ("product_name", "product_description", "product_features")
    return {name: _pct(products[name].dropna().str.split().map(len)) for name in fields}


def duplicate_stats(products: pd.DataFrame) -> dict[str, Any]:
    """Repeated products under several definitions."""
    body = [c for c in products.columns if c != "product_id"]
    names = products["product_name"]
    return {
        "rows_sharing_a_name": int(names.duplicated(keep=False).sum()),
        "rows_sharing_name_and_description": int(
            products[products["product_description"].notna()]
            .duplicated(["product_name", "product_description"], keep=False)
            .sum()
        ),
        "rows_identical_except_id": int(products.duplicated(body, keep=False).sum()),
    }


def pool_bias(
    queries: pd.DataFrame, products: pd.DataFrame, judged: pd.DataFrame
) -> dict[str, Any]:
    """Measure how strongly the judgment pool follows Wayfair's own query class.

    The WANDS paper says the second annotation step collected products in the class predicted for
    the query. A class-match feature may therefore predict being judged as well as being relevant.
    """
    classes = products.set_index("product_id")["product_class"].map(lambda c: set(split_classes(c)))
    query_class = queries.set_index("query_id")["query_class"]
    members: dict[str, set[int]] = {}
    for product_id, names in classes.items():
        for name in names:
            members.setdefault(name, set()).add(int(product_id))
    match = [
        query_class[q] in classes[p]
        for q, p in zip(judged["query_id"], judged["product_id"], strict=True)
    ]
    table = judged.assign(match=match)
    names = {grade: label for label, grade in GAINS.items()}
    mix = {
        "match" if flag else "no_match": {
            names[g]: round(float((group["grade"] == g).mean()), 4)
            for g in sorted(names, reverse=True)
        }
        for flag, group in table.groupby("match")
    }
    judged_by_query = judged.groupby("query_id")["product_id"].agg(set)
    in_class = pd.Series(
        {
            q: len(members[c] & judged_by_query.get(q, set())) / len(members[c])
            for q, c in query_class.items()
            if isinstance(c, str) and c in members
        }
    )
    exact = table[table["grade"] == GAINS["Exact"]]
    return {
        "judged_share_in_query_class_mean": round(float(in_class.mean()), 4),
        "judged_share_in_query_class_median": round(float(in_class.median()), 4),
        "queries_measured": len(in_class),
        "judged_share_overall": round(len(judged) / (len(queries) * len(products)), 4),
        "judged_pairs_in_query_class": _share(table["match"]),
        "exact_pairs_in_query_class": _share(exact["match"]),
        "label_mix": mix,
    }


def fold_balance(queries: pd.DataFrame, judged: pd.DataFrame, folds: Folds) -> list[dict[str, Any]]:
    """Aggregates per fold, to show that the dev fold is not an unusual sample."""
    rows = []
    for fold in range(folds.n_folds):
        ids = folds.queries(fold)
        table = grade_table(judged, ids)
        members = queries[queries["query_id"].isin(ids)]
        rows.append(
            {
                "fold": fold,
                "queries": len(ids),
                "distinct_classes": int(members["query_class"].nunique()),
                "median_judged": float(table.sum(axis=1).median()),
                "queries_without_exact": int((table["Exact"] == 0).sum()),
                "exact_at_10_ceiling": metric_ceilings(judged, ids)["exact_at_10"],
            }
        )
    return rows


def run_audit(
    queries: pd.DataFrame,
    products: pd.DataFrame,
    labels: pd.DataFrame,
    judged: pd.DataFrame,
    folds: Folds,
) -> dict[str, Any]:
    """Assemble every section."""
    ids = [int(q) for q in queries["query_id"]]
    catalogue = set(class_support(products["product_class"]).index)
    return {
        "rows": row_counts(queries, products, labels, judged),
        "labels": label_stats(judged, ids),
        "ceilings_all": metric_ceilings(judged, ids),
        "ceilings_primary": metric_ceilings(judged, folds.primary),
        "query_class": query_class_stats(queries, catalogue, folds),
        "product_class": product_class_stats(products),
        "taxonomy_grid": taxonomy_grid(products, queries, folds),
        "hierarchy": hierarchy_stats(products),
        "features": feature_stats(products),
        "missingness": missingness(products),
        "text": text_stats(products),
        "duplicates": duplicate_stats(products),
        "pool_bias": pool_bias(queries, products, judged),
        "folds": fold_balance(queries, judged, folds),
    }
