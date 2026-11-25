import math
import random

import pytest

from shelfrank.eval.crosscheck import max_differences
from shelfrank.eval.harness import evaluate
from shelfrank.eval.metrics import dcg, ndcg, perfect_ranking, score_query

LOG3 = math.log2(3)


def test_dcg_by_hand():
    # 2/log2(2) + 0 + 1/log2(4)
    assert dcg([2, 0, 1], 10) == pytest.approx(2.5)


def test_dcg_cuts_at_k():
    assert dcg([1, 1, 1], 2) == pytest.approx(1 + 1 / LOG3)


def test_ndcg_by_hand():
    # ranking: 1, 2; ideal: 2, 1
    assert ndcg([1, 2], [1, 2, 0], 10) == pytest.approx((1 + 2 / LOG3) / (2 + 1 / LOG3))


def test_ndcg_uses_the_whole_pool_as_ideal():
    # One Exact retrieved, another never retrieved.
    assert ndcg([2], [2, 2], 10) == pytest.approx(2 / (2 + 2 / LOG3))


def test_ndcg_is_zero_without_relevant_products():
    assert ndcg([0, 0], [0, 0], 10) == 0.0


def test_ndcg_is_invariant_to_scaling_gains():
    ranked, pool = [0, 2, 1], [2, 2, 1, 0]
    scaled = ndcg([g * 3 for g in ranked], [g * 3 for g in pool], 10)
    assert scaled == pytest.approx(ndcg(ranked, pool, 10))


def test_score_query_by_hand():
    judged = {1: 2, 2: 1, 3: 0, 4: 2}
    out = score_query([9, 3, 1, 2], judged)
    assert out.ndcg10 == pytest.approx((2 / 2 + 1 / math.log2(5)) / (2 + 2 / LOG3 + 1 / 2))
    assert out.condensed_ndcg10 == pytest.approx((2 / LOG3 + 1 / 2) / (2 + 2 / LOG3 + 1 / 2))
    assert out.judged10 == pytest.approx(0.3)
    assert out.recall100 == pytest.approx(2 / 3)
    assert out.exact10 == pytest.approx(0.1)
    assert out.exact10_share == pytest.approx(0.5)
    assert out.mrr_exact == pytest.approx(1 / 3)
    assert out.recall100_exact == pytest.approx(0.5)


def test_exact_metrics_are_undefined_without_exact_products():
    out = score_query([1], {1: 1})
    assert out.exact10 is out.exact10_share is out.mrr_exact is out.recall100_exact is None
    assert out.ndcg10 == 1.0 and out.recall100 == 1.0


def test_recall_is_undefined_without_relevant_products():
    out = score_query([1], {1: 0})
    assert out.recall100 is None and out.ndcg10 == 0.0


def test_missed_exact_scores_zero_not_undefined():
    out = score_query([5], {1: 2})
    assert out.mrr_exact == 0.0 and out.exact10 == 0.0


def test_empty_ranking_scores_zero():
    out = score_query([], {1: 2})
    assert (out.ndcg10, out.judged10, out.recall100) == (0.0, 0.0, 0.0)


def test_share_denominator_is_capped_at_ten():
    judged = {p: 2 for p in range(50)}
    assert score_query(list(range(10)), judged).exact10_share == 1.0


def test_condensed_promotes_products_below_unjudged_ones():
    judged = {100: 2}
    ranking = [*range(20), 100]
    out = score_query(ranking, judged)
    assert out.ndcg10 == 0.0 and out.condensed_ndcg10 == 1.0


def test_recall_stops_at_one_hundred():
    judged = {p: 1 for p in range(101)}
    assert score_query(list(range(101)), judged).recall100 == pytest.approx(100 / 101)


def test_repeated_products_are_rejected():
    with pytest.raises(ValueError, match="repeat"):
        score_query([1, 1], {1: 2})


def test_perfect_ranking_attains_every_bound():
    judged = {1: 0, 2: 2, 3: 1, 4: 2}
    assert perfect_ranking(judged) == [2, 4, 3, 1]
    out = score_query(perfect_ranking(judged), judged)
    assert out.ndcg10 == out.exact10_share == out.mrr_exact == out.recall100 == 1.0


def random_case(seed, n_queries=40):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(n_queries):
        pool = rng.sample(range(500), rng.randint(1, 200))
        qrels[q] = {p: rng.choice([0, 0, 1, 1, 2]) for p in pool}
        run[q] = rng.sample(range(500), rng.randint(1, 300))
    qrels[0] = {1: 0}  # nothing relevant
    qrels[1] = {1: 1, 2: 0}  # nothing Exact
    return run, qrels


@pytest.mark.parametrize("seed", [0, 1])
def test_agrees_with_ranx_per_query(seed):
    pytest.importorskip("ranx")
    run, qrels = random_case(seed)
    worst = max_differences(run, qrels, evaluate(run, qrels))
    assert set(worst) >= {"ndcg10", "condensed_ndcg10", "mrr_exact", "exact10", "recall100"}
    assert max(worst.values()) < 1e-6


def test_crosscheck_notices_a_wrong_metric():
    pytest.importorskip("ranx")
    run, qrels = random_case(2)
    ours = evaluate(run, qrels)
    ours[5]["ndcg10"] += 1e-3
    assert max_differences(run, qrels, ours)["ndcg10"] == pytest.approx(1e-3, rel=1e-3)
