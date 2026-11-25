import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.eval import final_report as fr

RESULTS = REPO_ROOT / "results/final_run.json"


def row(verdict, p):
    cell = {"mean": 0.1, "low": 0.05, "high": 0.2}
    metrics = {"ndcg10": cell, "condensed_ndcg10": cell}
    names = {"system": "ltr_full", "baseline": "hybrid_rrf"}
    return {**names, **metrics, "verdict": verdict, "p": {"claim": p}}


def test_a_claim_needs_the_pool_rule_and_holm():
    assert fr.claimed(row("gain", 0.001), 0.001) == "gain"
    assert "not after Holm" in fr.claimed(row("gain", 0.03), 0.06)
    assert fr.claimed(row("no clear difference", 0.001), 0.001) == "no clear difference"


def test_holm_is_applied_within_a_family():
    assert fr.adjusted([row("gain", 0.01), row("gain", 0.04)]) == pytest.approx([0.02, 0.04])


def test_undefined_values_are_left_out_and_not_zero():
    data = {"per_query": {"s": {"1": {"mrr_exact": None}, "2": {"mrr_exact": 0.5}}}}
    assert fr.mean(data, "s", "mrr_exact", [1, 2]) == 0.5


def test_groups_are_sorted_by_size_and_skip_missing():
    data = {"primary_ids": [1, 2, 3, 4], "departments": {"1": "A", "2": "B", "3": "B", "4": None}}
    assert list(fr.groups_in(data, "departments", "primary").items()) == [("B", [2, 3]), ("A", [1])]


needs_run = pytest.mark.skipif(not RESULTS.exists(), reason="the final run has not been made")


@pytest.fixture(scope="module")
def data():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@needs_run
def test_committed_page_is_up_to_date(data):
    page = (REPO_ROOT / "docs/final_results.md").read_text(encoding="utf-8")
    assert page == fr.render(data)


@needs_run
def test_primary_queries_are_the_384_and_every_query_is_ranked_by_every_system(data, folds):
    assert data["primary_ids"] == folds.primary and len(folds.primary) == 384
    for system in data["per_query"].values():
        assert len(system) == 480
    assert set(fr.NAMES) | set(fr.ABLATIONS) == set(data["per_query"])


@needs_run
def test_metrics_agree_with_ranx(data):
    assert data["ranx_max_difference"] < 1e-6


@needs_run
def test_dev_numbers_of_untrained_systems_equal_phase_1(data):
    old = json.loads((REPO_ROOT / "results/retrieval_dev.json").read_text(encoding="utf-8"))
    names = {
        "bm25_multi": "bm25_multi",
        "dense": "dense_bge_small",
        "hybrid_rrf": "hybrid_rrf",
    }
    for new, name in names.items():
        expected = old["systems"][name]["summary"]["ndcg10"]["mean"]
        assert fr.mean(data, new, "ndcg10", data["dev_ids"]) == pytest.approx(expected, abs=1e-9)


@needs_run
def test_every_frozen_comparison_was_made(data):
    import yaml

    frozen = yaml.safe_load((REPO_ROOT / "configs/frozen.yaml").read_text(encoding="utf-8"))
    for family in ("main", "retrieval", "ablation"):
        made = {f"{r['system']} vs {r['baseline']}" for r in data["comparisons"]["primary"][family]}
        declared = {name.split(" (")[0] for name in frozen["evaluation"]["families"][family]}
        assert made == declared
