import json

import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.eval.retrieval_report import (
    COLUMNS,
    LABELS,
    held_out_table,
    main_table,
    pool_note,
    render,
    sign_changes,
    tuning_table,
)


def load(name):
    return json.loads((REPO_ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return load("retrieval_dev")


def test_committed_page_is_up_to_date(data):
    page = (REPO_ROOT / "docs/retrieval_dev.md").read_text(encoding="utf-8")
    assert page == render(data, load("bm25_tuning"))


def test_every_table_row_has_every_metric(data):
    lines = main_table(data).splitlines()
    assert len(lines) == 2 + len(data["systems"]) + 2
    assert all(line.count("|") == len(COLUMNS) + 2 for line in lines)


def test_ceiling_and_n_rows_are_present(data):
    text = main_table(data)
    assert "ceiling)*" in text and "*n queries*" in text


def test_every_system_and_candidate_has_a_label(data):
    assert set(data["systems"]) | set(data["candidates"]) <= set(LABELS)


def test_no_system_beats_a_ceiling(data):
    for system in data["systems"].values():
        for metric in COLUMNS:
            limit = data["ceilings"][metric]["mean"]
            if metric != "judged10":
                assert system["summary"][metric]["mean"] <= limit + 1e-9


def test_dev_denominators_follow_the_protocol(data):
    counts = {m: data["ceilings"][m]["n"] for m in COLUMNS}
    assert counts["ndcg10"] == counts["recall100"] == 96
    assert counts["exact10"] == counts["mrr_exact"] == counts["recall100_exact"] == 74


def test_tuned_weights_in_use_are_the_committed_ones(data):
    assert data["weights"] == load("bm25_tuning")["weights"]


def test_tuning_table_shows_the_held_out_score():
    tuning = load("bm25_tuning")
    assert f"{tuning['dev_ndcg10_held_out']:.3f}" in tuning_table(tuning)


def fake(standard, condensed):
    return {
        "baseline": "bm25_multi",
        "systems": {
            name: {
                "summary": {"judged10": {"mean": 0.5}},
                "versus_baseline": {"ndcg10": {"mean": a}, "condensed_ndcg10": {"mean": b}},
            }
            for name, a, b in (("bm25_multi", 0.0, 0.0), ("dense_bge_small", standard, condensed))
        },
    }


@pytest.mark.parametrize(
    ("standard", "condensed", "flips"),
    [
        (-0.1, 0.1, True),
        (0.1, -0.1, True),
        (0.1, 0.2, False),
        (-0.1, -0.2, False),
        (0.1, 0.0, False),
    ],
)
def test_sign_changes_are_named(standard, condensed, flips):
    data = fake(standard, condensed)
    assert sign_changes(data) == (["dense_bge_small"] if flips else [])
    assert ("changes sign between" in pool_note(data) and flips) or not flips


def test_held_out_comparison_never_uses_in_sample_scores(data):
    tuning = load("bm25_tuning")
    flat = {**tuning, "held_out_per_query": dict(tuning["equal_weights_per_query"])}
    assert "| Equal weights | +0.000 [+0.000, +0.000] | no |" in held_out_table(data, flat)


def test_reported_tuned_score_matches_the_run(data):
    run = data["systems"]["bm25_multi"]["summary"]["ndcg10"]["mean"]
    assert run == pytest.approx(load("bm25_tuning")["dev_ndcg10_tuned_in_sample"], abs=1e-9)


def test_logged_configurations_stay_on_the_configured_grid():
    grid = set(load_config("retrieval")["bm25"]["tuning"]["grid"])
    for entry in load("bm25_tuning")["log"]:
        assert set(entry["weights"].values()) <= grid
        assert entry["weights"]["name"] == 1.0
