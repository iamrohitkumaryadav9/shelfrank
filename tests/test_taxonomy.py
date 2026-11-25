import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.taxonomy import (
    OTHER,
    build_taxonomy,
    class_support,
    product_coverage,
    query_coverage,
)

CELLS = ["Beds", "Beds|Sofas", "Rugs", None, "Beds"]


def test_support_counts_each_class_of_a_multi_class_product():
    assert class_support(CELLS).to_dict() == {"Beds": 3, "Rugs": 1, "Sofas": 1}


def test_support_counts_a_repeated_class_once_per_product():
    assert class_support(["Beds|Beds"]).to_dict() == {"Beds": 1}


def test_support_order_breaks_ties_by_name():
    assert list(class_support(["Z", "A", "M"]).index) == ["A", "M", "Z"]


def test_threshold_is_inclusive():
    support = class_support(CELLS)
    assert build_taxonomy(support, 3) == ["Beds"]
    assert build_taxonomy(support, 4) == []


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        build_taxonomy(class_support(CELLS), 0)


def test_product_coverage_counts_any_matching_class():
    assert product_coverage(CELLS, ["Sofas"]) == pytest.approx(1 / 5)
    assert product_coverage(CELLS, ["Beds"]) == pytest.approx(3 / 5)


def test_query_coverage_treats_missing_as_uncovered():
    assert query_coverage(["Beds", float("nan"), "Lamps"], ["Beds"]) == pytest.approx(1 / 3)


def test_coverage_of_nothing_is_zero():
    assert product_coverage([], ["Beds"]) == 0.0
    assert query_coverage([], ["Beds"]) == 0.0


def test_committed_taxonomy_matches_config(data_cfg):
    saved = json.loads((REPO_ROOT / data_cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    assert saved["min_support"] == data_cfg["taxonomy"]["min_support"]
    assert len(saved["classes"]) == len(set(saved["classes"]))
    assert saved["fallback"] == OTHER and OTHER not in saved["classes"]
    assert not any("|" in name for name in saved["classes"])
