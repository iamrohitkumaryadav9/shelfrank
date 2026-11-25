import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.eval import headline as hl


def row(baseline, verdict, p, low=0.01):
    cell = {"mean": 0.02, "low": low, "high": 0.03}
    return {
        "system": "ltr_full",
        "baseline": baseline,
        "ndcg10": cell,
        "condensed_ndcg10": cell,
        "verdict": verdict,
        "p": {"claim": p},
    }


def fake(verdict, p=0.0001, second="gain", p2=0.0001):
    per = {
        s: {"1": {"ndcg10": v}}
        for s, v in (("bm25_multi", 0.7), ("hybrid_rrf", 0.8), ("ltr_full", 0.81))
    }
    main = [row("hybrid_rrf", verdict, p), row("bm25_multi", second, p2)]
    ablation = [
        row("ltr_full", "loss", 0.001) | {"system": "without_G2"},
        row("ltr_full", "no clear difference", 0.5) | {"system": "without_G3"},
    ]
    return {
        "primary_ids": [1],
        "dev_ids": [],
        "queries": {"primary": 384},
        "per_query": per,
        "shap_by_feature": {"a": 1.0},
        "comparisons": {"primary": {"main": main, "ablation": ablation}},
    }


@pytest.mark.parametrize(
    ("verdict", "p", "kind"),
    [
        ("gain", 0.001, "A"),
        ("gain", 0.04, "B"),  # Holm doubles it
        ("no clear difference", 0.5, "B"),
        ("direction depends on unjudged handling", 0.5, "C"),
        ("loss", 0.001, "D"),
    ],
)
def test_framing_follows_the_verdict(verdict, p, kind):
    data = fake(verdict, p)
    if p == 0.04:
        data["comparisons"]["primary"]["main"][1]["p"]["claim"] = 0.04
        data["comparisons"]["primary"]["main"][0]["p"]["claim"] = 0.03
    assert hl.version(data) == kind


def test_headline_always_states_the_primary_comparison():
    for verdict in (
        "gain",
        "no clear difference",
        "direction depends on unjudged handling",
        "loss",
    ):
        text = hl.headline(fake(verdict))
        assert "hybrid RRF" in text and "384" in text


def test_version_b_does_not_claim_the_bm25_gain_when_it_does_not_hold():
    assert "Both beat tuned BM25" in hl.headline(fake("no clear difference"))
    text = hl.headline(fake("no clear difference", second="no clear difference", p2=0.5))
    assert "Both beat" not in text


def test_earning_groups_need_a_loss_after_holm():
    assert hl.earning_groups(fake("gain")) == ["G2"]


def test_framing_document_was_fixed_before_the_run_and_has_all_four_versions():
    text = (REPO_ROOT / "docs/README_FRAMING.md").read_text(encoding="utf-8")
    assert all(f"## Version {v}" in text for v in "ABCD")


@pytest.mark.skipif(not (REPO_ROOT / "results/final_run.json").exists(), reason="no final run")
def test_readme_shows_the_headline_of_the_actual_verdict():
    data = json.loads((REPO_ROOT / "results/final_run.json").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert hl.headline(data) in readme and hl.secondary(data) in readme
