import math

import numpy as np
import pandas as pd
import pytest

from shelfrank.qu.class_student import (
    Examples,
    clean,
    evaluate_classes,
    macro_f1,
    mix,
    name_examples,
    query_examples,
    ranked_labels,
    split_seed,
    to_reference,
)

TAXONOMY = ["Sofas", "Beds"]


def products(*rows):
    return pd.DataFrame(rows, columns=["product_name", "product_class"])


def test_names_get_their_class_and_small_classes_become_other():
    out = name_examples(products(("red sofa", "Sofas"), ("odd thing", "Rare")), TAXONOMY)
    assert list(zip(out.texts, out.labels, strict=True)) == [
        ("red sofa", "Sofas"),
        ("odd thing", "other"),
    ]


def test_products_without_a_class_are_not_used():
    assert name_examples(products(("mystery", np.nan)), TAXONOMY).texts == []


def test_a_known_class_beats_other_for_mixed_products():
    out = name_examples(products(("sleeper", "Rare|Sofas")), TAXONOMY)
    assert out.labels == ["Sofas"]


@pytest.mark.parametrize("tail", [None, 1])
def test_every_product_carries_one_unit_of_weight(tail):
    out = name_examples(
        products(("red sofa bed", "Sofas|Beds"), ("big bed", "Beds")), TAXONOMY, tail
    )
    assert sum(out.weights) == pytest.approx(2.0)


def test_tail_adds_the_last_words_once():
    out = name_examples(
        products(("acme red leather sofa", "Sofas"), ("sofa", "Sofas")), TAXONOMY, 2
    )
    assert out.texts == ["acme red leather sofa", "leather sofa", "sofa"]


def test_teacher_labels_outside_the_taxonomy_are_dropped():
    out = query_examples(["red sofas", "x"], ["Sofas", None])
    assert (out.texts, out.labels) == (["red sofa"], ["Sofas"])


def test_mix_gives_queries_the_requested_share_and_keeps_the_total():
    names = Examples(["a"] * 10, ["Sofas"] * 10, [1.0] * 10)
    queries = Examples(["b"], ["Beds"], [1.0])
    out = mix(names, queries, 0.3)
    assert sum(out.weights) == pytest.approx(10)
    assert out.weights[-1] == pytest.approx(3)
    assert names.weights == [1.0] * 10


def test_seed_split_is_fixed_disjoint_and_complete():
    train, held = split_seed(2000, 0.2, 20260)
    assert (train, held) == split_seed(2000, 0.2, 20260)
    assert len(held) == 400 and not set(train) & set(held)
    assert sorted(train + held) == list(range(2000))


def test_clean_uses_the_shared_text_pipeline():
    assert clean("Grey Vanities for 54in") == "gray vanity 54 inch"


def test_ranked_labels_break_ties_by_column():
    assert ranked_labels(np.array([[0.4, 0.4, 0.2]]), ["a", "b", "c"], 2) == [["a", "b"]]


@pytest.mark.parametrize(
    ("value", "expected"), [("Sofas", "Sofas"), ("Rare", "other"), (math.nan, "other")]
)
def test_reference_outside_the_taxonomy_is_other(value, expected):
    assert to_reference(value, TAXONOMY) == expected


def test_macro_f1_uses_only_supported_classes_and_counts_them():
    truth = ["a"] * 5 + ["b"] * 4
    predicted = ["a"] * 4 + ["b"] * 5
    score, classes = macro_f1(truth, predicted, 5)
    assert classes == 1 and score == pytest.approx(8 / 9)
    assert math.isnan(macro_f1(["a"], ["a"], 5)[0])


def test_evaluate_classes_by_hand():
    truth = ["Sofas", "Beds", "Lamps", "other"]
    ranked = [["Sofas", "x", "y"], ["Sofas", "Beds", "x"], ["x", "y", "z"], ["other", "x", "y"]]
    departments = {"Sofas": "Furniture", "Beds": "Furniture", "Lamps": "Lighting"}
    out = evaluate_classes(truth, ranked, departments, {"Sofas", "other"})
    assert (out["n"], out["accuracy"], out["top3"], out["department"]) == (4, 0.5, 0.75, 0.75)
    assert out["seeded"] == {"n": 2, "accuracy": 1.0}
    assert out["names_only"] == {"n": 2, "accuracy": 0.0}


def test_unknown_labels_share_no_department_with_a_known_one():
    out = evaluate_classes(["Lamps"], [["zzz"]], {"Lamps": "Lighting"})
    assert out["department"] == 0.0


def test_linear_classifier_learns_and_returns_probabilities():
    pytest.importorskip("sklearn")
    from shelfrank.qu.class_student import LinearClassifier

    data = Examples()
    for word, label in (("sofa", "Sofas"), ("bed frame", "Beds"), ("phone case", "other")):
        for color in ("red", "blue", "green", "black"):
            data.add(f"{color} {word}", label, 1.0)
    model = LinearClassifier(ngrams=(2, 4), alpha=1e-5).fit(data)
    p = model.probabilities(["pink sofas", "a bed frame"])
    ours = model.model.predict_proba(model.vectorizer.transform(["pink sofa", "bed frame"]))
    assert p == pytest.approx(ours, abs=1e-5)
    assert p.sum(axis=1) == pytest.approx([1, 1])
    assert [r[0] for r in ranked_labels(p, model.classes, 1)] == ["Sofas", "Beds"]
    again = LinearClassifier(ngrams=(2, 4), alpha=1e-5).fit(data).probabilities(["pink sofas"])
    assert np.array_equal(again, model.probabilities(["pink sofas"]))


def test_arms_train_on_what_their_name_says():
    from shelfrank.qu.class_data import BOTH, NAMES, QUERIES, arm_examples

    frame = products(("red sofa", "Sofas"), ("oak bed", "Beds"))
    seed = query_examples(["couch"], ["Sofas"])
    assert arm_examples(NAMES, None, None, frame, TAXONOMY, seed).texts == ["red sofa", "oak bed"]
    assert arm_examples(QUERIES, None, None, frame, TAXONOMY, seed).texts == ["couch"]
    both = arm_examples(BOTH, None, 0.5, frame, TAXONOMY, seed)
    assert len(both.texts) == 3 and both.weights[-1] == pytest.approx(sum(both.weights) / 2)


def test_held_out_seed_queries_are_not_trained_on():
    from shelfrank.qu.class_data import seed_parts

    parts = seed_parts()
    assert parts is not None
    train, (held, labels) = parts
    assert len(held) == len(labels) == 400 and len(train.texts) >= 1590
    assert not {clean(q) for q in held} & set(train.texts)


def committed_class_results():
    import json

    from shelfrank.config import REPO_ROOT

    return json.loads((REPO_ROOT / "results/class_dev.json").read_text(encoding="utf-8"))


def test_committed_selection_follows_the_rule_over_every_configuration():
    data = committed_class_results()
    best = max(data["runs"], key=lambda r: r["selection"])
    assert data["selected"] == best["arm"] and data["best"][best["arm"]] == best
    for run in data["runs"]:
        mean = (run["dev"]["accuracy"] + run["esci"]["accuracy"]) / 2
        assert run["selection"] == pytest.approx(mean)


def test_committed_class_page_is_up_to_date():
    from shelfrank.config import REPO_ROOT
    from shelfrank.qu.class_report import render

    page = (REPO_ROOT / "docs/class_dev.md").read_text(encoding="utf-8")
    import json

    paths = sorted((REPO_ROOT / "results").glob("class_ceiling_*.json"))
    ceilings = {
        p.stem.removeprefix("class_ceiling_"): json.loads(p.read_text(encoding="utf-8"))
        for p in paths
    }
    assert page == render(committed_class_results(), ceilings)
