import json

import pandas as pd
import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.audit import (
    feature_stats,
    grade_table,
    label_stats,
    metric_ceilings,
    missingness,
    pool_bias,
    row_counts,
    run_audit,
)
from shelfrank.data.audit_report import num, pct, render, table
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries
from shelfrank.eval.splits import Folds


def judged_frame(rows):
    return pd.DataFrame(rows, columns=["query_id", "product_id", "grade"])


def test_grade_table_includes_queries_without_judgments():
    out = grade_table(judged_frame([(1, 10, 2), (1, 11, 0)]), [1, 2])
    assert out.loc[1].tolist() == [1, 0, 1]
    assert out.loc[2].tolist() == [0, 0, 0]


def test_label_stats_count_hard_queries():
    rows = [(1, p, 2) for p in range(12)] + [(2, 1, 1), (3, 1, 0)]
    out = label_stats(judged_frame(rows), [1, 2, 3])
    assert out["queries_without_exact"] == 2
    assert out["queries_under_10_exact"] == 2
    assert out["queries_without_relevant"] == 1
    assert out["queries_without_irrelevant"] == 2
    assert out["counts"] == {"Exact": 12, "Partial": 1, "Irrelevant": 1}


def test_ceilings_by_hand():
    rows = [(1, p, 2) for p in range(5)] + [(2, p, 1) for p in range(200)]
    out = metric_ceilings(judged_frame(rows), [1, 2])
    assert out["exact_at_10"] == pytest.approx((0.5 + 0.0) / 2)
    assert out["mrr_exact"] == pytest.approx(0.5)
    assert out["recall_at_100"] == pytest.approx((1.0 + 0.5) / 2)
    assert out["recall_at_100_exact_only"] == pytest.approx(1.0)


def test_row_counts_find_repeats_and_conflicts():
    queries = pd.DataFrame({"query_id": [1], "query": ["bed"], "query_class": ["Beds"]})
    products = pd.DataFrame({"product_id": [10, 11]})
    labels = pd.DataFrame(
        {
            "id": range(4),
            "query_id": [1, 1, 1, 1],
            "product_id": [10, 10, 10, 99],
            "label": ["Exact", "Exact", "Partial", "Exact"],
        }
    )
    out = row_counts(queries, products, labels, judgments(labels))
    assert out["repeated_rows"] == 2 and out["max_repeats"] == 3
    assert out["conflicting_pairs"] == 1
    assert out["label_products_missing"] == 1 and out["products_never_judged"] == 1


def test_pool_bias_by_hand():
    queries = pd.DataFrame({"query_id": [1], "query": ["bed"], "query_class": ["Beds"]})
    products = pd.DataFrame(
        {"product_id": [10, 11, 12, 13], "product_class": ["Beds", "Beds|Sofas", "Rugs", None]}
    )
    out = pool_bias(queries, products, judged_frame([(1, 10, 2), (1, 12, 0), (1, 13, 1)]))
    assert out["judged_share_in_query_class_mean"] == pytest.approx(0.5)
    assert out["judged_pairs_in_query_class"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["exact_pairs_in_query_class"] == 1.0
    assert out["label_mix"]["no_match"] == {"Exact": 0.0, "Partial": 0.5, "Irrelevant": 0.5}


def test_feature_stats_count_multi_valued_slots():
    products = pd.DataFrame(
        {"product_features": ["color : red|color : blue| : yes", "color : red"]}
    )
    out = feature_stats(products)
    assert out["slots"]["color"]["multi_valued"] == 1
    assert out["slots"]["color"]["distinct_values"] == 2
    assert out["keyless"] == 1 and out["slots"]["room"]["products"] == 0


def test_missingness_detects_partial_rating_rows(sample):
    broken = sample["products"].copy()
    assert missingness(broken)["rating_fields_missing_together"] is True
    broken.loc[broken["review_count"].first_valid_index(), "review_count"] = float("nan")
    assert missingness(broken)["rating_fields_missing_together"] is False


def test_audit_runs_on_the_fixture_and_never_emits_query_text(sample):
    ids = [int(q) for q in sample["queries"]["query_id"]]
    folds = Folds({q: i % 2 for i, q in enumerate(ids)}, n_folds=2, dev_fold=0)
    audit = run_audit(
        sample["queries"], sample["products"], sample["labels"], sample["judged"], folds
    )
    text = json.dumps(audit, ensure_ascii=False)
    assert not any(q in text for q in sample["queries"]["query"])


def test_table_escapes_the_pipe():
    assert "a\\|b" in table(["x"], [["a|b"]])


@pytest.mark.parametrize(("value", "expected"), [(1234, "1,234"), (2.0, "2"), (2.53, "2.5")])
def test_num(value, expected):
    assert num(value) == expected


def test_pct():
    assert pct(0.1234) == "12.3%"


def test_committed_report_matches_committed_json(data_cfg):
    audit = json.loads((REPO_ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    report = (REPO_ROOT / "docs/data_audit.md").read_text(encoding="utf-8")
    assert report == render(audit, data_cfg["taxonomy"]["min_support"], data_cfg["wands"]["commit"])


def test_prose_claims_in_the_report_hold():
    audit = json.loads((REPO_ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    ceilings = [row["exact_at_10_ceiling"] for row in audit["folds"]]
    assert ceilings[0] == min(ceilings)
    mix = audit["pool_bias"]["label_mix"]
    irrelevant_ratio = mix["no_match"]["Irrelevant"] / mix["match"]["Irrelevant"]
    assert irrelevant_ratio > mix["match"]["Exact"] / mix["no_match"]["Exact"]
    assert audit["features"]["slots"]["room"]["share"] < 0.05
    assert audit["duplicates"]["rows_identical_except_id"] == 0


@pytest.mark.full_data
def test_committed_audit_matches_the_data(raw_dir, folds):
    labels = load_labels(raw_dir)
    audit = run_audit(
        load_queries(raw_dir), load_products(raw_dir), labels, judgments(labels), folds
    )
    saved = json.loads((REPO_ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(audit)) == saved
