"""The guide and practice answers must follow from the same rules the teacher got."""

import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.wands import load_queries
from shelfrank.eval.leakage import exact_matches, load_key_hashes, near_matches
from shelfrank.qu.annotator import PRACTICE, SLOT_GUIDE
from shelfrank.qu.guidelines import SLOT_EXAMPLES, SLOTS, quoted_phrases
from shelfrank.qu.slot_vocab import in_vocabulary, load_vocabulary, room_slots
from shelfrank.qu.verification import problems, unknown_styles

RAW = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
UNION, STYLES, PHRASES = load_vocabulary(RAW)
EXAMPLES = dict(SLOT_EXAMPLES)
QUOTED = set(quoted_phrases())


@pytest.mark.parametrize("slot", SLOTS)
def test_guide_has_a_rule_and_two_examples_each_way(slot):
    guide = SLOT_GUIDE[slot]
    assert "\n" not in guide.rule and len(guide.yes) == len(guide.no) == 2


@pytest.mark.parametrize("slot", SLOTS)
def test_guide_examples_come_from_the_guidelines_and_agree_with_them(slot):
    guide = SLOT_GUIDE[slot]
    for text, span in guide.yes:
        assert text in EXAMPLES or text in QUOTED
        if text in EXAMPLES:
            assert span in EXAMPLES[text][slot]
    for text, _ in guide.no:
        assert text in EXAMPLES or text in QUOTED
        if text in EXAMPLES:
            assert EXAMPLES[text][slot] == []


def test_fifteen_distinct_practice_queries_that_are_not_the_shown_examples():
    queries = [item.query for item in PRACTICE]
    assert len(queries) == len(set(queries)) == 15
    assert not set(queries) & set(EXAMPLES)


@pytest.mark.parametrize("item", PRACTICE, ids=lambda item: item.query)
def test_practice_answers_pass_the_hard_blocks(item):
    done = {}
    for slot in SLOTS:
        assert problems(item.query, done, item.answer[slot]) == []
        done[slot] = item.answer[slot]


@pytest.mark.parametrize("item", PRACTICE, ids=lambda item: item.query)
def test_practice_answers_follow_the_room_rule_and_the_vocabulary(item):
    assert item.answer["room"] == room_slots(item.query, PHRASES)
    for slot in SLOTS:
        assert all(in_vocabulary(slot, span, UNION, STYLES) for span in item.answer[slot])
    assert unknown_styles(item.answer["style"], STYLES.keys()) == []


def test_practice_covers_every_slot_and_has_queries_with_no_slot():
    assert all(sum(bool(i.answer[s]) for i in PRACTICE) >= 2 for s in SLOTS)
    assert sum(not any(i.answer.values()) for i in PRACTICE) >= 3


def test_practice_queries_are_in_no_wands_or_esci_file():
    queries = [item.query for item in PRACTICE]
    assert (
        exact_matches(queries, load_key_hashes(REPO_ROOT / "data/splits/wands_query_keys.json"))
        == []
    )
    for name in ("esci_seed_queries.txt", "esci_prompt_dev.txt"):
        lines = (REPO_ROOT / "data/esci" / name).read_text(encoding="utf-8").splitlines()
        assert near_matches(queries, lines, 0.75) == []


@pytest.mark.full_data
def test_practice_queries_resemble_no_wands_query(raw_dir, data_cfg):
    threshold = data_cfg["splits"]["near_duplicate_jaccard"]
    queries = [item.query for item in PRACTICE]
    assert near_matches(queries, load_queries(raw_dir)["query"], threshold) == []
