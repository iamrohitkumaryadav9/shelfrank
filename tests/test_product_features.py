import pytest

from shelfrank.data.product_features import parse_features, slot_of_key


def test_text_and_numeric_spacing_parse_the_same_way():
    assert parse_features("color : red|height:64.7").pairs == [("color", "red"), ("height", "64.7")]


def test_only_the_first_colon_splits():
    assert parse_features("ratio : 16:9").pairs == [("ratio", "16:9")]


def test_keyless_and_malformed_parts_are_kept_apart():
    parsed = parse_features(" : yes|no colon here|color : red")
    assert parsed.pairs == [("color", "red")]
    assert parsed.keyless == ["yes"]
    assert parsed.malformed == ["no colon here"]


@pytest.mark.parametrize("raw", [None, float("nan"), "", "|", " | "])
def test_empty_inputs_give_nothing(raw):
    parsed = parse_features(raw)
    assert (parsed.pairs, parsed.keyless, parsed.malformed) == ([], [], [])


def test_a_key_with_an_empty_value_is_still_a_pair():
    assert parse_features("color :").pairs == [("color", "")]


def test_by_key_keeps_every_variant_once_in_order():
    parsed = parse_features("color : caramel|size:2|color : chocolate|color : caramel")
    assert parsed.by_key() == {"color": ["caramel", "chocolate"], "size": ["2"]}


def test_parses_do_not_share_state():
    parse_features("color : red")
    assert parse_features("").pairs == []


def test_fixture_products_all_parse(sample):
    parsed = sample["products"]["product_features"].map(parse_features)
    assert parsed.map(lambda item: len(item.pairs) > 0).all()


@pytest.mark.parametrize(
    ("key", "slot"),
    [
        ("color", "color"),
        ("legcolor", "color"),
        ("isthisrugavailableindifferentcolors", None),
        ("primarymaterial", "material"),
        ("containsflameretardantmaterials", None),
        ("containsrecycledmaterial", None),
        ("style", "style"),
        ("dsprimaryproductstyle", "style"),
        ("faucethandlestyle", None),
        ("roomuse", "room"),
        ("roomdivider", None),
        ("overallwidth-sidetoside", None),
    ],
)
def test_slot_of_key(key, slot):
    assert slot_of_key(key) == slot
