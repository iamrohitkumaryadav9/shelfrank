import pytest

from shelfrank.text.normalize import INVARIANT, IRREGULAR_PLURALS, SPELLING, normalize, tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Décor", "decor"),
        ("  Two   Words ", "two words"),
        ("27.5 '' wide", "27.5 inch wide"),
        ('54" bench', "54 inch bench"),
        ("54 in. bench", "54 inch bench"),
        ("54in bench", "54 inch bench"),
        ("12 inches", "12 inch"),
        ("3 in 1 crib", "3 in 1 crib"),
        ("chair in red", "chair in red"),
        ("2 indoor lamps", "2 indoor lamps"),
        ("5 ' x 8 '", "5 ft x 8 ft"),
        ("6 ft tree", "6 ft tree"),
        ("kid ’s desk", "kid desk"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("grey sofas", ["gray", "sofa"]),
        ("coloured centrepieces", ["colored", "centerpiece"]),
        ("mid-century", ["mid", "century"]),
        ("benches boxes dishes mattresses", ["bench", "box", "dish", "mattress"]),
        ("vanities", ["vanity"]),
        ("shelves", ["shelf"]),
        ("glass bus gas canvas", ["glass", "bus", "gas", "canvas"]),
        ("stainless", ["stainless"]),
        ("chaises tables", ["chaise", "table"]),
        ("54in", ["54", "inch"]),
        ("27.5 inch", ["27.5", "inch"]),
        ("table with a lamp", ["table", "lamp"]),
        ("!!! ---", []),
        (None, []),
        (float("nan"), []),
    ],
)
def test_tokenize(raw, expected):
    assert tokenize(raw) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("grey", "gray"),
        ("Mid Century", "mid-century"),
        ('36" vanity', "36 in vanities"),
        ("Wall Décor", "wall decor"),
        ("kids' chairs", "kid chair"),
    ],
)
def test_query_and_product_spellings_meet(a, b):
    assert tokenize(a) == tokenize(b)


def test_tokenizing_is_idempotent():
    once = tokenize("Grey Shelves, 27.5'' wide; kids' vanities & boxes")
    assert tokenize(" ".join(once)) == once


def test_distinct_product_types_are_not_merged():
    assert tokenize("bedding") != tokenize("beds")
    assert tokenize("lighting") != tokenize("lights")


def test_tables_are_clean():
    assert all(k != v for k, v in {**SPELLING, **IRREGULAR_PLURALS}.items())
    assert not INVARIANT & set(IRREGULAR_PLURALS)
    assert not set(SPELLING) & set(SPELLING.values())
