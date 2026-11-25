import json

import pandas as pd
import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.eval.access import ProtocolError, queries_for
from shelfrank.eval.harness import candidate_report, ceilings, evaluate, summarise, to_qrels
from shelfrank.eval.metrics import METRIC_NAMES
from shelfrank.eval.splits import Folds

FOLDS = Folds({1: 0, 2: 1, 3: 2}, n_folds=3, dev_fold=0)


def test_dev_is_always_available(tmp_path):
    assert queries_for("dev", FOLDS, frozen=tmp_path / "none.yaml") == [1]


@pytest.mark.parametrize("split", ["primary", "all"])
def test_other_splits_are_locked_before_freezing(split, tmp_path):
    with pytest.raises(ProtocolError):
        queries_for(split, FOLDS, frozen=tmp_path / "none.yaml")


def test_freezing_unlocks_the_other_splits(tmp_path):
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text("{}")
    assert queries_for("primary", FOLDS, frozen) == [2, 3]
    assert queries_for("all", FOLDS, frozen) == [1, 2, 3]


def test_unknown_split_is_an_error():
    with pytest.raises(ValueError, match="unknown"):
        queries_for("test", FOLDS)


def test_qrels_hold_only_requested_queries():
    judged = pd.DataFrame({"query_id": [1, 2], "product_id": [7, 8], "grade": [2, 1]})
    assert to_qrels(judged, [1, 3]) == {1: {7: 2}, 3: {}}


def test_missing_ranking_is_an_error():
    with pytest.raises(KeyError, match="1 queries"):
        evaluate({1: [7]}, {1: {7: 2}, 2: {8: 1}})


def test_extra_rankings_are_ignored():
    assert set(evaluate({1: [7], 9: [1]}, {1: {7: 2}})) == {1}


def test_summary_denominators_follow_the_protocol():
    qrels = {1: {7: 2}, 2: {8: 1}, 3: {9: 0}}
    out = summarise(evaluate({1: [7], 2: [8], 3: [9]}, qrels))
    assert out["ndcg10"] == {"mean": pytest.approx(2 / 3), "n": 3}
    assert out["recall100"] == {"mean": 1.0, "n": 2}
    assert out["exact10"] == {"mean": pytest.approx(0.1), "n": 1}


def test_ceilings_match_a_hand_case():
    out = ceilings({1: {p: 2 for p in range(5)}, 2: {p: 1 for p in range(200)}})
    assert out["exact10"] == {"mean": pytest.approx(0.5), "n": 1}
    assert out["recall100"]["mean"] == pytest.approx((1 + 0.5) / 2)


def test_ceilings_bound_any_run():
    qrels = {1: {1: 2, 2: 1, 3: 0}, 2: {4: 2, 5: 2}}
    run = summarise(evaluate({1: [3, 2, 9], 2: [8, 5]}, qrels))
    best = ceilings(qrels)
    assert all(run[m]["mean"] <= best[m]["mean"] for m in METRIC_NAMES)


def test_candidate_report_by_hand():
    qrels = {1: {1: 2, 2: 1, 3: 0, 4: 2}}
    out = candidate_report({1: [3, 1, 99]}, qrels)
    assert out["mean_size"] == 3
    assert out["recall"] == pytest.approx(1 / 3)
    assert out["recall_exact"] == pytest.approx(1 / 2)
    assert out["judged_share"] == pytest.approx(2 / 3)
    assert out["oracle_exact10_share"] == pytest.approx(0.5)


def test_candidate_oracle_is_one_when_everything_is_retrieved():
    qrels = {1: {1: 2, 2: 1}}
    assert candidate_report({1: [9, 2, 1]}, qrels)["oracle_ndcg10"] == pytest.approx(1.0)


def test_larger_candidate_sets_never_lower_the_oracle():
    qrels = {1: {1: 2, 2: 1, 3: 2}}
    small = candidate_report({1: [2]}, qrels)["oracle_ndcg10"]
    assert candidate_report({1: [2, 3]}, qrels)["oracle_ndcg10"] > small


def committed():
    path = REPO_ROOT / "results/retrieval_dev.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_dev_results_contain_dev_queries_only(folds):
    for system in committed()["systems"].values():
        assert {int(q) for q in system["per_query"]} == set(folds.dev)


def test_committed_dev_metrics_agree_with_ranx():
    worst = max(v for d in committed()["ranx_max_difference"].values() for v in d.values())
    assert worst < 1e-6


def test_committed_summaries_match_their_per_query_values():
    for system in committed()["systems"].values():
        again = summarise(system["per_query"])
        for metric, row in again.items():
            saved = system["summary"][metric]
            assert (saved["mean"], saved["n"]) == (pytest.approx(row["mean"]), row["n"])


def test_tuning_used_dev_queries_only(folds):
    saved = json.loads((REPO_ROOT / "results/bm25_tuning.json").read_text(encoding="utf-8"))
    assert {int(q) for q in saved["held_out_per_query"]} == set(folds.dev)
