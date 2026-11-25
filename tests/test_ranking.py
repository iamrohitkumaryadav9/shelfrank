import numpy as np
import pytest

from shelfrank.ranking import rank_all, reciprocal_rank_fusion, top_products, union

IDS = np.array([10, 20, 30, 40])


def test_best_first():
    assert top_products(np.array([0.1, 0.9, 0.5, 0.2]), IDS, 2) == [20, 30]


def test_ties_go_to_the_lower_id():
    assert top_products(np.array([1.0, 2.0, 1.0, 2.0]), IDS, 4) == [20, 40, 10, 30]


def test_positive_only_drops_unmatched_products():
    scores = np.array([0.0, 2.0, 0.0, 1.0])
    assert top_products(scores, IDS, 4, positive_only=True) == [20, 40]
    assert len(top_products(scores, IDS, 4)) == 4


def test_negative_scores_still_rank_for_dense_retrieval():
    assert top_products(np.array([-0.3, -0.1, -0.2, -0.4]), IDS, 2) == [20, 30]


@pytest.mark.parametrize("ids", [[2, 1], [1, 1]])
def test_unsorted_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="ascending"):
        rank_all(np.zeros((1, 2)), [1], np.array(ids), 2)


def test_query_count_must_match():
    with pytest.raises(ValueError, match="zip"):
        rank_all(np.zeros((2, 4)), [1], IDS, 2)


def test_rrf_by_hand():
    # 1: 1/61 + 1/62, 2: 1/62 + 1/61, 3: 1/63
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1]]) == [1, 2, 3]


def test_rrf_rewards_agreement():
    assert reciprocal_rank_fusion([[1, 2], [3, 2]])[0] == 2


def test_rrf_ignores_ranks_below_the_depth():
    assert 3 not in reciprocal_rank_fusion([[1, 2, 3]], depth=2)


def test_rrf_of_an_empty_ranking_keeps_the_other():
    assert reciprocal_rank_fusion([[], [5, 4]]) == [5, 4]


def test_rrf_does_not_depend_on_system_order():
    a, b = [1, 2, 3, 4], [4, 3, 9]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([b, a])


def test_union_keeps_first_seen_order_within_depth():
    assert union([[1, 2, 3], [2, 4, 5]], depth=2) == [1, 2, 4]
