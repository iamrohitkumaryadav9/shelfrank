import math

import numpy as np
import pytest

from shelfrank.bm25.fields import BM25Index
from shelfrank.features.attributes import Listed
from shelfrank.similar import sets as ss
from shelfrank.similar import systems as sy

QRELS = {1: {10: 2, 11: 2, 12: 2, 13: 1, 14: 0}, 2: {10: 2}, 3: {20: 1}}


def test_sets_hold_exact_products_and_need_two():
    assert ss.substitute_sets(QRELS) == {1: [10, 11, 12]}


def test_seed_draw_is_limited_reproducible_and_independent_of_other_sets():
    sets = {1: list(range(50)), 2: list(range(5))}
    seeds = ss.draw_seeds(sets)
    assert len([s for s in seeds if s.query_id == 1]) == 20
    assert len([s for s in seeds if s.query_id == 2]) == 5
    assert [s for s in ss.draw_seeds({1: sets[1]})] == [s for s in seeds if s.query_id == 1]


def test_seed_metrics_by_hand():
    out = ss.score_seed([99, 11, 14, 12], {11, 12}, QRELS[1])
    best = 1 + 1 / math.log2(3)
    assert out["ndcg10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / best)
    assert out["condensed_ndcg10"] == pytest.approx((1 + 1 / math.log2(4)) / best)
    assert out["recall10"] == 1.0 and out["mrr"] == 0.5 and out["recall10_ceiling"] == 1.0


def test_nothing_found_scores_zero():
    out = ss.score_seed([99], {11}, QRELS[1])
    assert out["ndcg10"] == out["mrr"] == out["recall10"] == 0.0


def test_the_seed_is_never_counted():
    seed = ss.Seed(1, 10)
    with_seed = ss.evaluate_seeds({seed: [10, 11, 12]}, {1: [10, 11, 12]}, QRELS)
    without = ss.evaluate_seeds({seed: [11, 12]}, {1: [10, 11, 12]}, QRELS)
    assert with_seed == without and with_seed[seed]["ndcg10"] == 1.0


def test_query_mean_and_seed_mean_differ_by_weighting():
    groups = {1: [1.0, 1.0, 1.0], 2: [0.0]}
    assert ss.clustered_mean(groups, False, 100).mean == 0.5
    assert ss.clustered_mean(groups, True, 100).mean == 0.75


def test_interval_resamples_queries_and_not_seeds():
    many = {q: [float(q % 2)] * 500 for q in range(10)}
    interval = ss.clustered_mean(many, True, 2000)
    assert interval.high - interval.low > 0.4  # 5,000 independent seeds would give about 0.03


def test_difference_is_paired_by_seed():
    a = {ss.Seed(1, 1): {"mrr": 0.5}, ss.Seed(2, 1): {"mrr": 0.7}}
    b = {ss.Seed(1, 1): {"mrr": 0.4}, ss.Seed(2, 1): {"mrr": 0.6}}
    assert ss.difference(a, b, "mrr", n_resamples=50).mean == pytest.approx(0.1)
    with pytest.raises(ValueError, match="same seeds"):
        ss.difference(a, {ss.Seed(1, 1): {"mrr": 0.4}}, "mrr")


IDS = np.array([10, 20, 30])


def test_rankings_leave_out_the_seed_and_respect_depth():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    assert sy.dense_rankings(vectors, {10: 0}, IDS, 2) == {10: [20, 30]}
    assert sy.dense_rankings(vectors, {10: 0, 30: 2}, IDS, 1, chunk=1) == {10: [20], 30: [20]}
    index = BM25Index([["red", "sofa"], ["red", "chair"], ["lamp"]], 1.2, 0.75)
    assert sy.bm25_rankings(index, {10: ["red", "sofa"]}, IDS, 5) == {10: [20]}


def test_fusion_per_seed():
    assert sy.fuse({1: [5, 6]}, {1: [6, 7]}, 60, 2) == {1: [6, 5]}


@pytest.mark.parametrize(
    ("a", "b", "expected"), [(["x"], ["x", "y"], 1.0), (["x"], ["y"], 0.0), ([], [], 0.0)]
)
def test_shared_class(a, b, expected):
    assert sy.shared_class(a, b) == expected


def listed(union=(), style=(), mentioned=()):
    return Listed(
        {tuple(t.split()) for t in union}, {tuple(t.split()) for t in style}, set(), set(mentioned)
    )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (listed(["dark gray"], mentioned=["color"]), listed(["gray"], mentioned=["color"]), 1.0),
        (listed(["red"], mentioned=["color"]), listed(["gray"], mentioned=["color"]), 0.0),
        (listed(["red"], mentioned=["color"]), listed([], mentioned=[]), 0.0),
        (
            listed(["red"], ["modern"], ["color", "style"]),
            listed(["red"], ["rustic"], ["color", "style"]),
            0.5,
        ),
    ],
)
def test_attribute_agreement(a, b, expected):
    assert sy.agreement(a, b) == expected == sy.agreement(b, a)


def test_prior_is_scaled_and_handles_no_reviews():
    out = sy.rating_prior(np.array([5.0, 4.0]), np.array([0.0, 10.0]))
    assert out.tolist() == [0.0, 1.0]
    assert sy.rating_prior(np.array([0.0]), np.array([0.0])).tolist() == [0.0]


def test_zero_weights_keep_the_fused_order_and_each_weight_can_move_a_product():
    runs = [[2, 3, 4], [2, 3, 4]]
    terms = {2: (0.0, 0.0, 0.0), 3: (0.0, 0.0, 0.0), 4: (1.0, 1.0, 1.0)}
    assert sy.rerank(1, [2, 3, 4], 60, runs, sy.RerankWeights(), terms) == [2, 3, 4]
    for name in ("shared_class", "attributes", "prior"):
        weights = sy.RerankWeights(**{name: 1.0})
        assert sy.rerank(1, [2, 3, 4], 60, runs, weights, terms)[0] == 4


def test_rerank_drops_the_seed_and_handles_products_in_one_run_only():
    out = sy.rerank(
        9, [9, 2, 3], 60, [[2], [3]], sy.RerankWeights(), {p: (0.0, 0.0, 0.0) for p in (9, 2, 3)}
    )
    assert out == [2, 3]


def test_lexicon_terms_map_to_one_slot():
    from shelfrank.ltr.slots_for_ranking import term_slots

    out = term_slots()
    assert out[("walnut",)] == "material_finish" and set(out.values()) == {
        "color",
        "material_finish",
    }
