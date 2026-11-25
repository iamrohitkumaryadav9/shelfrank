import json

import pandas as pd
import pytest

from scripts.slot_evidence import ROOM_EXAMPLES, render
from shelfrank.config import REPO_ROOT
from shelfrank.data.wands import load_products
from shelfrank.qu import slot_vocab as sv
from shelfrank.qu.guidelines import (
    CLASS_VERSION,
    SLOT_RULES,
    RulesNotFinalError,
    require_final,
)


@pytest.mark.parametrize(
    ("key", "group"),
    [
        ("finish", "finish"),
        ("hardwarefinish", "finish"),
        ("finishcolor", "finish"),
        ("legcolor", "color"),
        ("dswoodtone", "color"),
        ("woodspecies", "material"),
        ("framematerialdetails", "material"),
        ("containsrecycledmaterial", None),
        ("dsprimaryproductstyle", "style"),
        ("faucethandlestyle", None),
        ("pattern", None),
    ],
)
def test_key_group(key, group):
    assert sv.key_group(key) == group


def frame(*features):
    return pd.DataFrame({"product_features": features})


def test_value_table_counts_a_product_once_per_family():
    table = sv.value_table(frame("legcolor : red|color : red|finish : red|glossfinish : yes"))
    assert table == {"red": {"color": 1, "finish": 1}}


def test_style_terms_split_strip_audience_and_drop_non_styles():
    rows = ["style : country / farmhouse"] * 3 + ["style : youth boho"] * 3
    rows += ["style : no style"] * 5 + ["style : rare"] * 2
    out = sv.style_terms(sv.value_table(frame(*rows)))
    assert out == {"country": 3, "farmhouse": 3, "boho": 3}


def test_majority_terms_normalise_and_report_the_share():
    rows = ["finish : Brushed Nickel"] * 3 + ["color : brushed nickel"]
    out = sv.majority_terms(sv.value_table(frame(*rows)), ("color", "finish"), min_products=2)
    assert out[("brushed", "nickel")] == ("finish", 0.75, 4)


def test_majority_terms_skip_numbers_and_long_values():
    rows = ["color : 64 inch"] * 30 + ["color : a very long marketing colour name"] * 30
    assert sv.majority_terms(sv.value_table(frame(*rows)), ("color",)) == {}


PHRASES = {
    ("dining", "table"),
    ("kitchen", "island"),
    ("bathroom", "vanity"),
    ("outdoor", "dining", "set"),
    ("office", "chair"),
}


@pytest.mark.parametrize(
    ("query", "slots"),
    [
        ("dining table", []),
        ("round dining tables", []),
        ("kitchen island with stools", []),
        ("patio dining set", []),
        ("outdoor dining sets", []),
        ("bathroom rug", ["bathroom"]),
        ("patio umbrella", ["patio"]),
        ("lamp for bedroom", ["bedroom"]),
        ("dining table for kitchen", ["kitchen"]),
        ("office desk", ["office"]),
        ("living room sofa", ["living room"]),
        ("red chair", []),
        ("", []),
    ],
)
def test_room_slots(query, slots):
    assert sv.room_slots(query, PHRASES) == slots


def test_longest_room_word_wins_without_overlap():
    assert sv.room_spans("living room") == [(0, 2)]
    assert sv.room_slots("home office lamp", set()) == ["home office"]


def test_every_room_word_inside_a_product_phrase_belongs_to_it():
    assert sv.room_slots("patio dining set for balcony", PHRASES) == ["balcony"]


def test_a_phrase_that_ends_with_the_room_word_does_not_absorb_it():
    assert sv.room_slots("shed for garden", {("shed", "garden")}) == ["garden"]
    assert sv.room_slots("play kitchen", {("play", "kitchen")}) == ["kitchen"]


def test_room_word_must_come_before_the_head_noun():
    # "table dining" does not contain the phrase "dining table".
    assert sv.room_slots("table dining", PHRASES) == ["dining"]


def test_product_phrases_respect_thresholds_and_merge_patio_with_outdoor():
    products = pd.DataFrame(
        {
            "product_class": ["Patio Tables"] * 3 + ["Deck Boxes"],
            "category_hierarchy": ["Outdoor / Garden Statues"] * 2 + [None, None],
        }
    )
    out = sv.product_type_phrases(products, min_class=2, min_path=2)
    assert ("outdoor", "table") in out and ("outdoor", "statue") in out
    assert ("deck", "box") not in out


def test_match_terms_prefers_longer_terms():
    terms = {("gray",): 1, ("dark", "gray"): 1}
    assert sv.match_terms("dark grey rug and gray mat", terms) == [("dark", "gray"), ("gray",)]


def test_release_candidates_are_refused_and_final_versions_pass():
    with pytest.raises(RulesNotFinalError):
        require_final("slots-v1-rc1")
    require_final("slots-v1")
    require_final(CLASS_VERSION)


@pytest.mark.parametrize(
    ("slot", "span", "known"),
    [
        ("color", "very dark gray", True),
        ("material_finish", "dark gray", True),
        ("color", "sparkleberry", False),
        ("style", "mid century modern", True),
        ("style", "gray", False),
        ("room", "patio", True),
        ("room", "spaceship", False),
    ],
)
def test_vocabulary_is_shared_by_colour_and_material_but_not_style(slot, span, known):
    union, styles = {("gray",): 9}, {("mid", "century", "modern"): 9}
    assert sv.in_vocabulary(slot, span, union, styles) is known


@pytest.mark.parametrize("word", ["patio", "garden", "balcony", "deck", "outdoor"])
def test_outdoor_places_share_product_phrases(word):
    assert sv.room_slots(f"{word} dining set", PHRASES) == []


def load_vocab():
    return json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))


def test_values_named_in_the_rules_are_in_the_committed_vocabulary():
    union, styles, _ = sv.load_vocabulary(load_vocab())
    for value in ("chrome", "brushed nickel", "matte black", "oil rubbed bronze", "walnut", "oak"):
        assert sv.in_vocabulary("material_finish", value, union, styles)
    for value in ("navy", "light gray", "espresso", "gold", "silver"):
        assert sv.in_vocabulary("color", value, union, styles)
    for value in ("modern", "farmhouse", "mid century", "boho", "rustic", "coastal", "vintage"):
        assert sv.in_vocabulary("style", value, union, styles), value


def test_style_list_has_the_recorded_cutoff():
    vocab = load_vocab()
    assert vocab["min_products"] == 3 and min(vocab["styles"].values()) >= 3
    assert vocab["styles"] == load_evidence()["styles"]


@pytest.mark.full_data
def test_committed_vocabulary_is_rebuilt_from_the_catalogue(raw_dir):
    assert json.loads(json.dumps(sv.build_vocabulary(load_products(raw_dir)))) == load_vocab()


def test_rule_text_gives_the_same_room_examples_as_the_code():
    text = " ".join(SLOT_RULES)
    for query in ("dining table", "kitchen island", "bathroom vanity", "bathroom rug"):
        assert f"'{query}'" in text


def load_evidence():
    return json.loads((REPO_ROOT / "results/slot_evidence.json").read_text(encoding="utf-8"))


def test_committed_evidence_page_is_up_to_date():
    page = (REPO_ROOT / "docs/slot_evidence.md").read_text(encoding="utf-8")
    assert page == render(load_evidence())


def test_implemented_room_rule_reproduces_every_example():
    assert load_evidence()["room_examples"] == ROOM_EXAMPLES


def test_patterns_are_not_in_the_style_vocabulary():
    evidence = load_evidence()
    assert set(evidence["patterns"].values()) == {0}
    assert not set(evidence["patterns"]) & set(evidence["styles"])


@pytest.mark.full_data
def test_room_rule_on_the_real_catalogue(raw_dir):
    phrases = sv.product_type_phrases(load_products(raw_dir))
    assert {q: bool(sv.room_slots(q, phrases)) for q in ROOM_EXAMPLES} == ROOM_EXAMPLES
