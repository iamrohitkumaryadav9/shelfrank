import json
import math

import pandas as pd
import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.wands import load_products
from shelfrank.qu.guidelines import SLOT_EXAMPLES, SLOTS
from shelfrank.qu.slot_lexicon import (
    LexiconMatcher,
    build_lexicon,
    canonical,
    slot_of_term,
    span_scores,
    surface_tokens,
    wood_species,
)

LEXICON = json.loads((REPO_ROOT / "data/slot_lexicon.json").read_text(encoding="utf-8"))
MATCHER = LexiconMatcher(LEXICON)
EMPTY = {s: [] for s in SLOTS}
SPECIES = {"walnut", "oak", "cherry"}


@pytest.mark.parametrize(
    ("term", "majority", "slot"),
    [
        (("walnut",), "color", "material_finish"),
        (("white", "oak"), "color", "material_finish"),
        (("gold",), "finish", "color"),
        (("copper",), "material", "color"),
        (("brushed", "nickel"), "color", "material_finish"),
        (("matte", "black"), "color", "material_finish"),
        (("antique", "brass"), "color", "material_finish"),
        (("antique", "white"), "color", "color"),
        (("chrome",), "finish", "material_finish"),
        (("espresso",), "color", "color"),
        (("glass",), "material", "material_finish"),
    ],
)
def test_slot_follows_the_rules_and_not_the_key(term, majority, slot):
    assert slot_of_term(term, majority, SPECIES) == slot


@pytest.mark.parametrize(("query", "answer"), SLOT_EXAMPLES, ids=[q for q, _ in SLOT_EXAMPLES])
def test_matcher_reproduces_every_example_in_the_rules(query, answer):
    assert MATCHER.slots(query) == answer


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("coffee table", {}),
        ("round mirror", {}),
        ("gold lamp", {"color": ["gold"]}),
        ("gold finish lamp", {"material_finish": ["gold finish"]}),
        ("Grey Sofas", {"color": ["Grey"]}),
        ("wooden spoon rest", {"material_finish": ["wooden"]}),
        ("rustic mid-century clock", {"style": ["rustic", "mid-century"]}),
        ("navy blue vase", {"color": ["navy blue"]}),
        ("bathroom rug", {"room": ["bathroom"]}),
        ("bathroom vanity", {}),
        ("", {}),
    ],
)
def test_matcher(query, expected):
    assert MATCHER.slots(query) == EMPTY | expected


def test_no_word_is_in_two_slots():
    for query in (REPO_ROOT / "data/esci/esci_prompt_dev.txt").read_text("utf-8").splitlines():
        words = [w for spans in MATCHER.slots(query).values() for s in spans for w in s.split()]
        assert all(words.count(w) <= query.split().count(w) for w in words)


def test_spans_are_the_words_of_the_query():
    for query in (REPO_ROOT / "data/esci/esci_prompt_dev.txt").read_text("utf-8").splitlines():
        for spans in MATCHER.slots(query).values():
            assert all(span in " ".join(query.split()) for span in spans)


def test_canonical_keeps_a_known_term():
    assert canonical(("navy", "blue")) == ("navy",)
    assert canonical(("navy", "blue"), {("navy", "blue"): 1}) == ("navy", "blue")


def test_surface_tokens_point_back_to_their_word():
    words, owner, tokens = surface_tokens("Mid-Century chairs for 2")
    assert words == ["Mid-Century", "chairs", "for", "2"]
    assert tokens == ["mid", "century", "chair", "2"] and owner == [0, 0, 1, 3]


def test_wood_species_skip_non_species():
    features = ["woodspecies : oak", "woodspecies : oak", "topwoodspecies : no wood"]
    frame = pd.DataFrame({"product_features": [*features, "framewoodspecies : oak"]})
    assert wood_species(frame, 3) == {"oak"} and wood_species(frame, 4) == set()


def test_span_scores_by_hand():
    truth = [EMPTY | {"color": ["red"]}, EMPTY | {"color": ["blue"], "room": ["patio"]}]
    made = [EMPTY | {"color": ["Red"]}, EMPTY | {"color": ["navy"]}]
    out = span_scores(truth, made)
    assert out["color"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "predicted": 2,
        "support": 2,
    }
    assert out["room"]["recall"] == 0 and math.isnan(out["style"]["f1"])
    assert (out["all"]["predicted"], out["all"]["support"]) == (2, 3)


def test_lexicon_code_reads_no_query():
    code = (REPO_ROOT / "shelfrank/qu/slot_lexicon.py").read_text(encoding="utf-8")
    assert "load_queries" not in code and "esci" not in code.lower()


@pytest.mark.full_data
def test_committed_lexicon_comes_from_the_catalogue(raw_dir):
    built = json.loads(json.dumps(build_lexicon(load_products(raw_dir))))
    assert built == LEXICON
