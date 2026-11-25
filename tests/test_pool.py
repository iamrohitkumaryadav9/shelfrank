import pytest

from shelfrank.config import load_config
from shelfrank.eval.pool import judged_by_class

CLASSES = {1: ["Beds"], 2: ["Beds", "Sofas"], 3: ["Rugs"], 4: []}


def test_split_by_hand():
    out = judged_by_class({9: [1, 2, 3, 4]}, {9: {1: 2, 3: 0}}, {9: "Beds"}, CLASSES)
    assert (out["in_class_n"], out["out_of_class_n"]) == (2, 2)
    assert out["in_class_judged"] == out["out_of_class_judged"] == pytest.approx(0.5)


def test_queries_without_a_class_are_left_out():
    out = judged_by_class({9: [1]}, {9: {1: 2}}, {9: float("nan")}, CLASSES)
    assert out["in_class_n"] == out["out_of_class_n"] == 0


def test_only_the_top_k_counts():
    out = judged_by_class({9: [3, 1]}, {9: {1: 2}}, {9: "Beds"}, CLASSES, k=1)
    assert (out["in_class_n"], out["out_of_class_n"]) == (0, 1)


def test_rrf_constant_is_frozen_at_the_default():
    assert load_config("retrieval")["rrf"]["k"] == 60
