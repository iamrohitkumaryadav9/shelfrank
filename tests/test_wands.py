import pandas as pd
import pytest

from shelfrank.config import load_config
from shelfrank.data.wands import (
    GAINS,
    PRODUCT_COLUMNS,
    judgments,
    load_labels,
    load_products,
    load_queries,
    resolve_grade,
    split_classes,
    split_hierarchy,
)


def labels_frame(rows):
    return pd.DataFrame(
        [(i, q, p, label) for i, (q, p, label) in enumerate(rows)],
        columns=["id", "query_id", "product_id", "label"],
    )


def test_gains_follow_the_protocol():
    assert GAINS == {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@pytest.mark.parametrize(
    ("grades", "expected"),
    [
        ([2], 2),
        ([2, 2, 1], 2),
        ([2, 1, 1], 1),
        ([2, 1], 1),
        ([1, 0], 0),
        ([0, 0, 1], 0),
        ([2, 1, 0], 0),
    ],
)
def test_resolve_grade_majority_then_lower(grades, expected):
    assert resolve_grade(grades) == expected
    assert resolve_grade(grades[::-1]) == expected


def test_judgments_collapse_repeats():
    out = judgments(labels_frame([(1, 10, "Exact"), (1, 10, "Exact"), (1, 11, "Partial")]))
    assert out.to_dict("list") == {"query_id": [1, 1], "product_id": [10, 11], "grade": [2, 1]}


def test_judgments_keep_the_same_product_separate_per_query():
    out = judgments(labels_frame([(1, 10, "Exact"), (2, 10, "Irrelevant")]))
    assert out["grade"].tolist() == [2, 0]


def test_exactly_one_label_per_pair_after_loading(sample):
    assert not sample["judged"].duplicated(["query_id", "product_id"]).any()


@pytest.mark.full_data
def test_exactly_one_label_per_pair_on_the_full_data(raw_dir):
    labels = load_labels(raw_dir)
    judged = judgments(labels)
    assert not judged.duplicated(["query_id", "product_id"]).any()
    assert len(judged) == len(labels[["query_id", "product_id"]].drop_duplicates())
    assert set(judged["grade"]) == set(GAINS.values())


def test_judgments_do_not_depend_on_row_order():
    rows = labels_frame([(2, 5, "Exact"), (1, 9, "Partial"), (1, 9, "Irrelevant"), (1, 3, "Exact")])
    pd.testing.assert_frame_equal(judgments(rows), judgments(rows.iloc[::-1]))


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("Beds", ["Beds"]),
        ("Teen Lounge Furniture|Accent Chairs", ["Teen Lounge Furniture", "Accent Chairs"]),
        (" Beds | ", ["Beds"]),
        (float("nan"), []),
        (None, []),
        ("", []),
    ],
)
def test_split_classes(cell, expected):
    assert split_classes(cell) == expected


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("Furniture / Bedroom Furniture / Beds", ["Furniture", "Bedroom Furniture", "Beds"]),
        ("Décor / Wall/Ceiling", ["Décor", "Wall/Ceiling"]),
        (float("nan"), []),
    ],
)
def test_split_hierarchy_uses_the_spaced_slash_only(cell, expected):
    assert split_hierarchy(cell) == expected


def test_fixture_loads_with_renamed_column(sample):
    assert list(sample["products"].columns) == PRODUCT_COLUMNS
    assert len(sample["queries"]) == 5 and len(sample["products"]) == 30


def test_fixture_covers_missing_and_multi_class_products(sample):
    products = sample["products"]
    assert products["product_class"].isna().any()
    assert products["average_rating"].isna().any()
    assert products["product_class"].map(lambda c: len(split_classes(c))).max() > 1


def test_fixture_covers_every_grade_and_a_query_without_exact(sample):
    judged = sample["judged"]
    assert set(judged["grade"]) == {0, 1, 2}
    assert judged.groupby("query_id")["grade"].max().min() < 2


def test_fixture_judgments_reference_known_rows(sample):
    assert set(sample["judged"]["query_id"]) == set(sample["queries"]["query_id"])
    assert set(sample["judged"]["product_id"]) <= set(sample["products"]["product_id"])


def test_schema_change_is_rejected(tmp_path):
    (tmp_path / "query.csv").write_text("query_id\tquery\n1\tbed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        load_queries(tmp_path)


def test_unknown_label_is_rejected(tmp_path):
    (tmp_path / "label.csv").write_text(
        "id\tquery_id\tproduct_id\tlabel\n0\t1\t2\tMaybe\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Maybe"):
        load_labels(tmp_path)


def test_literal_none_text_is_not_read_as_missing(tmp_path):
    header = "\t".join(
        ["category hierarchy" if c == "category_hierarchy" else c for c in PRODUCT_COLUMNS]
    )
    row = "1\tnone\tBeds\tA / B\tnull\tx:1\t\t\t"
    (tmp_path / "product.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    loaded = load_products(tmp_path).iloc[0]
    assert loaded["product_name"] == "none" and loaded["product_description"] == "null"
    assert pd.isna(loaded["average_rating"])


def test_load_config_rejects_a_non_mapping(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bad.yaml").write_text("- a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config("bad", root=tmp_path)
