import pytest

from shelfrank.text.near_duplicates import jaccard, near_duplicate_groups, token_key


def test_key_ignores_case_order_punctuation_and_plurals():
    assert token_key("Blue Throw-Pillows!") == token_key("pillow throw blue")


def test_short_tokens_keep_their_s():
    assert token_key("gas bus") == frozenset({"gas", "bus"})


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ({"x"}, {"x"}, 1.0),
        ({"x", "y"}, {"y", "z"}, 1 / 3),
        ({"x"}, {"y"}, 0.0),
        (set(), set(), 0.0),
    ],
)
def test_jaccard(a, b, expected):
    assert jaccard(frozenset(a), frozenset(b)) == pytest.approx(expected)


def test_groups_same_tokens():
    assert near_duplicate_groups({1: "red chair", 2: "chairs red", 3: "lamp"}, 0.75) == [[1, 2]]


def test_threshold_is_inclusive():
    queries = {1: "a1 b1 c1", 2: "a1 b1 c1 d1"}
    assert near_duplicate_groups(queries, 0.75) == [[1, 2]]
    assert near_duplicate_groups(queries, 0.76) == []


def test_chains_merge_into_one_component():
    queries = {1: "a1 b1 c1 d1", 2: "a1 b1 c1 d1 e1", 3: "a1 b1 c1 d1 e1 f1"}
    assert jaccard(token_key(queries[1]), token_key(queries[3])) < 0.75
    assert near_duplicate_groups(queries, 0.75) == [[1, 2, 3]]


def test_blank_queries_do_not_group():
    assert near_duplicate_groups({1: "!!", 2: "??"}, 0.5) == []


def test_unrelated_queries_give_no_groups():
    assert near_duplicate_groups({1: "bed", 2: "lamp"}, 0.5) == []
