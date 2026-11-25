import inspect
import math

import numpy as np
import pandas as pd
import pytest

from shelfrank.features import attributes as at
from shelfrank.features import lexical as lx
from shelfrank.features import priors, query_class, semantic, text_shape
from shelfrank.features.groups import columns_of, group_of
from shelfrank.ltr.first_stage import candidate_pairs, first_stage
from shelfrank.text.normalize import tokenize

NAMES = ["red velvet sofa", "blue lamp", "red lamp shade", "oak table 54 inch"]
IDS = np.array([10, 20, 30, 40])
TEXTS = ["red lamp", "oak table 54 in"]


@pytest.fixture
def tokens():
    names = [tokenize(n) for n in NAMES]
    empty = [[] for _ in NAMES]
    classes = [["sofa"], ["lamp"], ["lamp", "shade"], []]
    return {"name": names, "class": classes, "features": empty, "description": empty}


@pytest.fixture
def stage(tokens):
    vectors = np.eye(4, dtype=np.float32)
    queries = np.array([[0.1, 0.9, 0.5, 0.0], [0, 0, 0, 1.0]], dtype=np.float32)
    weights = {"name": 1.0, "class": 0.0}
    return first_stage([7, 3], TEXTS, IDS, tokens, weights, queries, vectors, 1.2, 0.75, 60, 4)


@pytest.fixture
def pairs(stage):
    return candidate_pairs(stage, 2)


def row(frame, pairs, query, product):
    return frame[(pairs["query_id"] == query) & (pairs["product_id"] == product)].iloc[0]


def test_candidates_are_the_union_of_both_generators(stage, pairs):
    for q in (7, 3):
        top = set(stage.runs["bm25_multi"][q][:2]) | set(stage.runs["dense"][q][:2])
        assert set(pairs[pairs["query_id"] == q]["product_id"]) == top


def test_candidate_rows_point_to_their_query_and_product(stage, pairs):
    assert (stage.product_ids[pairs["pi"]] == pairs["product_id"]).all()
    assert [stage.query_ids[i] for i in pairs["qi"]] == pairs["query_id"].tolist()
    assert pairs["query_id"].is_monotonic_increasing


def test_fields_with_weight_zero_are_scored_but_do_not_rank(stage, tokens):
    assert stage.fields["class"].any()
    assert np.array_equal(stage.tuned, stage.fields["name"])


@pytest.mark.parametrize(
    ("query", "document", "expected"),
    [(["red", "lamp"], {"red"}, 0.5), (["red", "red"], {"red"}, 1.0), ([], {"red"}, 0.0)],
)
def test_coverage(query, document, expected):
    assert lx.coverage(query, document) == expected


def test_rare_words_weigh_more():
    idf = lx.idf_table([["red", "lamp"], ["red"], ["red"]])
    assert idf["lamp"] > idf["red"]
    assert lx.weighted_coverage(["red", "lamp"], {"lamp"}, idf) > 0.5
    assert lx.weighted_coverage(["zzz"], set(), idf) == 0.0


@pytest.mark.parametrize(
    ("document", "bigrams", "phrase"),
    [("a red lamp", 1.0, True), ("lamp red", 0.0, False), ("red tall lamp", 0.0, False)],
)
def test_word_order(document, bigrams, phrase):
    words = document.split()
    assert lx.bigram_share(["red", "lamp"], words) == bigrams
    assert lx.contains_phrase(["red", "lamp"], words) is phrase


def test_single_word_query_has_no_bigram_and_empty_query_no_phrase():
    assert lx.bigram_share(["red"], ["red"]) == 0.0
    assert lx.contains_phrase([], ["red"]) is False


def test_relative_scores_handle_a_query_without_matches():
    out = lx.relative(np.array([[2.0, 1.0], [0.0, 0.0]]))
    assert out.tolist() == [[1.0, 0.5], [0.0, 0.0]]


def test_lexical_features_by_hand(pairs, stage, tokens):
    out = lx.lexical_features(pairs, stage, [tokenize(t) for t in TEXTS], tokens)
    best = row(out, pairs, 7, 30)
    assert (best["g1_cover_name"], best["g1_phrase_name"], best["g1_bigrams_name"]) == (1, 1, 1)
    assert best["g1_bm25_name_rel"] == 1.0 and best["g1_last_word_in_class"] == 1.0
    assert len(out) == len(pairs) and not out.isna().any().any()


def test_reciprocal_rank_is_zero_outside_the_run():
    frame = pd.DataFrame({"query_id": [1, 1], "product_id": [5, 6]})
    assert lx.reciprocal_ranks(frame, {1: [6]}, 60) == [0.0, 1 / 61]


def test_semantic_features_by_hand(pairs, stage):
    out = semantic.semantic_features(pairs, stage, 60)
    best = row(out, pairs, 7, 20)
    assert best["g2_cosine"] == pytest.approx(0.9) and best["g2_cosine_gap"] == 0.0
    assert row(out, pairs, 7, 30)["g2_cosine_gap"] == pytest.approx(0.4)
    position = stage.runs["bm25_multi"][7].index(20) + 1
    assert best["g2_rrf"] == pytest.approx(1 / 61 + 1 / (60 + position))


CLASSES = ["Lamps", "Sofas", "other"]
DEPARTMENTS = {"Lamps": "Lighting", "Sofas": "Furniture", "Rare Lamps": "Lighting"}


def class_table(product_classes):
    frame = pd.DataFrame({"qi": [0] * len(product_classes), "pi": range(len(product_classes))})
    p = np.array([[0.6, 0.3, 0.1]])
    return query_class.class_features(frame, p, CLASSES, product_classes, DEPARTMENTS)


def test_class_features_by_hand():
    out = class_table([["Lamps"], ["Sofas"], ["Sofas", "Lamps"]])
    assert out["g3_p_class"].tolist() == [0.6, 0.3, 0.6]
    assert out["g3_class_rank"].tolist() == [1, 2, 1]
    assert out["g3_is_top_class"].tolist() == [1, 0, 1]
    assert out["g3_p_department"].tolist() == [0.6, 0.3, 0.6]


def test_small_class_counts_as_other_but_keeps_its_department():
    out = class_table([["Rare Lamps"]]).iloc[0]
    assert out["g3_p_class"] == 0.1 and out["g3_p_department"] == 0.6


def test_product_without_a_class_is_missing_and_not_zero():
    out = class_table([[]]).iloc[0]
    assert out[["g3_p_class", "g3_class_rank"]].isna().all() and out["g3_confidence"] == 0.6


def product_frame(**changes):
    base = {
        "rating_count": [100.0, 1.0, math.nan],
        "average_rating": [4.0, 5.0, math.nan],
        "review_count": [10.0, math.nan, 0.0],
        "product_features": ["a : b|c : d", None, ""],
        "product_description": ["text", None, "x"],
    }
    return pd.DataFrame(base | changes)


def test_priors_shrink_small_samples_to_the_catalogue_mean():
    out = priors.prior_features(pd.DataFrame({"pi": [2, 0, 1]}), product_frame())
    mean = 405 / 101
    assert out["g5_shrunk_rating"].tolist() == pytest.approx(
        [mean, (400 + 10 * mean) / 110, (5 + 10 * mean) / 11]
    )
    assert out["g5_features"].tolist() == [0, 2, 0]
    assert math.isnan(out["g5_average_rating"][0]) and out["g5_log_ratings"][0] == 0.0
    assert out["g5_has_description"].tolist() == [1, 1, 0]


def test_number_match_is_missing_without_a_number(tokens):
    frame = pd.DataFrame({"qi": [0, 1, 1], "pi": [0, 3, 0]})
    out = text_shape.shape_features(frame, [tokenize(t) for t in TEXTS], tokens)
    assert math.isnan(out["g6_numbers_in_name"][0])
    assert out["g6_numbers_in_name"].tolist()[1:] == [1.0, 0.0]
    assert out["g6_query_words"].tolist() == [2, 4, 4]


SLOT_OF = {("walnut",): "material_finish", ("gray",): "color", ("dark", "gray"): "color"}


def listed(features, hierarchy=None):
    return at.listed_values(features, hierarchy, SLOT_OF)


@pytest.mark.parametrize(
    ("slot", "span", "features", "expected"),
    [
        ("color", "gray", "color : dark gray", at.MATCH),
        ("color", "dark gray", "color : gray", at.CONFLICT),
        ("color", "grey", "finish : gray", at.MATCH),
        ("color", "blue", "color : gray|color : blue", at.MATCH),
        ("color", "red", "color : gray|color : blue", at.CONFLICT),
        ("color", "red", "material : walnut", at.NOT_MENTIONED),
        ("color", "red", "color : walnut", at.NOT_MENTIONED),
        ("material_finish", "walnut", "color : walnut", at.MATCH),
        ("material_finish", "oak", "color : walnut", at.CONFLICT),
        ("style", "modern", "style : modern", at.MATCH),
        ("style", "boho", "dsprimaryproductstyle : modern", at.CONFLICT),
        ("style", "mid century", "style : mid-century modern", at.MATCH),
        ("style", "modern", "color : gray", at.NOT_MENTIONED),
        ("room", "bedroom", "roomuse : bedroom", at.MATCH),
        ("room", "patio", "roomuse : outdoor", at.MATCH),
        ("room", "bedroom", "roomuse : bathroom", at.CONFLICT),
        ("color", "red", "", at.NOT_MENTIONED),
        ("color", "yes", "legcolor : yes", at.NOT_MENTIONED),
    ],
)
def test_attribute_state(slot, span, features, expected):
    assert at.state(slot, span, listed(features)) == expected


def test_room_is_also_read_from_the_category_path():
    product = listed("", "Outdoor / Patio Furniture / Chairs")
    assert at.state("room", "garden", product) == at.MATCH
    assert at.state("room", "bedroom", product) == at.CONFLICT


def test_part_of_a_word_is_not_a_match():
    assert not at.contains(("grayish",), ("gray",))
    assert not at.contains(("dark", "gray"), ("dark", "blue"))
    assert not at.contains(("gray",), ())


def test_conflict_on_one_of_two_values_is_a_conflict():
    product = listed("color : gray")
    assert at.slot_state("color", ["gray", "red"], product) == at.CONFLICT
    assert at.slot_state("color", ["gray"], product) == at.MATCH
    assert at.slot_state("color", [], product) == at.NOT_ASKED


def test_attribute_features_by_hand():
    frame = pd.DataFrame({"qi": [0, 1], "pi": [0, 0]})
    slots = [{"color": ["gray"], "style": ["boho"]}, {}]
    out = at.attribute_features(frame, slots, {0: listed("color : gray|style : modern")})
    first = out.iloc[0]
    assert first["g4_color_match"] == 1 and first["g4_style_conflict"] == 1
    assert (first["g4_asked"], first["g4_matches"], first["g4_conflicts"]) == (2, 1, 1)
    assert first["g4_match_share"] == 0.5
    assert (out.iloc[1] == 0).all()


def test_columns_are_selected_by_group():
    columns = ["g1_a", "g2_b", "g1_c"]
    assert columns_of(columns, ["G1"]) == ["g1_a", "g1_c"]
    with pytest.raises(ValueError, match="no feature group"):
        group_of("judged_flag")
    with pytest.raises(ValueError, match="unknown"):
        columns_of(columns, ["G9"])


BUILDERS = (
    lx.lexical_features,
    semantic.semantic_features,
    query_class.class_features,
    at.attribute_features,
    priors.prior_features,
    text_shape.shape_features,
)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_no_feature_builder_can_see_judgments_or_the_reference_class(builder):
    names = " ".join(inspect.signature(builder).parameters)
    code = inspect.getsource(inspect.getmodule(builder))
    assert not any(word in names for word in ("qrel", "judg", "grade", "label", "pool"))
    for word in ("load_labels", "judgments(", 'query_class"', "qrels"):
        assert word not in code
