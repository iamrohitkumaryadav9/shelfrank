import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.suggest.model import (
    FrequencyBaseline,
    QueryTrie,
    SuggestModel,
    prefix_cases,
    reciprocal_rank,
)

QUERIES = ["red sofa bed", "red sofa cover", "red rug", "blue sofa bed", "lamp"]


def test_trie_counts_different_queries():
    trie = QueryTrie(QUERIES)
    assert trie.next_words([]) == {"red": 3, "blue": 1, "lamp": 1}
    assert trie.next_words(["red"]) == {"sofa": 2, "rug": 1}
    assert trie.next_words(["green"]) == {} and trie.node(["lamp"]).ends == 1


def test_exact_prefix_beats_back_off():
    model = SuggestModel(QUERIES)
    assert model.rank(["red"])[0] == "sofa"
    assert model.rank(["green", "sofa"])[:1] == ["bed"]  # unseen prefix: last word decides


def test_unseen_words_fall_back_to_the_corpus():
    assert set(SuggestModel(QUERIES).rank(["zzz"], 3)) <= {"sofa", "red", "bed"}


def test_unfinished_word_filters():
    assert SuggestModel(QUERIES).rank(["red"], typed="r") == ["rug", "red"]


def test_ties_are_alphabetical():
    assert SuggestModel(["a c", "a b"]).rank(["a"], 2) == ["b", "c"]


def test_class_prior_reorders_by_the_class_of_the_words_queries():
    queries = ["wood bed", "wood lamp"]
    plain = SuggestModel(queries)
    assert plain.rank(["wood"], 2) == ["bed", "lamp"]
    boosted = SuggestModel(queries, ["Beds", "Lamps"], lambda text: {"Lamps": 1.0}, 5.0)
    assert boosted.rank(["wood"], 2) == ["lamp", "bed"]


def test_weight_zero_equals_no_prior():
    with_prior = SuggestModel(QUERIES, ["x"] * 5, lambda text: {"x": 1.0}, 0.0)
    assert with_prior.rank(["red"]) == SuggestModel(QUERIES).rank(["red"])


@pytest.mark.parametrize(
    ("text", "first"), [("red ", "red sofa bed"), ("red s", "red sofa bed"), ("", "red sofa")]
)
def test_completions_for_the_demo(text, first):
    assert SuggestModel(QUERIES).complete(text)[0] == first


def test_frequency_baseline_ignores_the_prefix():
    baseline = FrequencyBaseline(QUERIES)
    assert baseline.rank(["red"], 2) == baseline.rank(["blue"], 2) == ["red", "sofa"]


@pytest.mark.parametrize(("truth", "expected"), [("a", 1.0), ("b", 0.5), ("z", 0.0)])
def test_reciprocal_rank(truth, expected):
    assert reciprocal_rank(["a", "b"], truth) == expected


def test_cases_need_a_following_word():
    assert prefix_cases(["red sofa bed", "lamp"], 1) == [(0, ["red"], "sofa")]
    assert prefix_cases(["red sofa bed"], 3) == []


RESULT = REPO_ROOT / "results/suggest.json"


@pytest.mark.skipif(not RESULT.exists(), reason="not evaluated yet")
def test_committed_page_is_generated_and_says_there_is_no_popularity():
    from shelfrank.suggest.report import render

    data = json.loads(RESULT.read_text(encoding="utf-8"))
    page = (REPO_ROOT / "docs/suggest.md").read_text(encoding="utf-8")
    assert page == render(data) and "There is no popularity here" in page
    best = max(data["tuning"], key=lambda r: r["mrr"])
    assert data["class_weight"] == best["class_weight"]
