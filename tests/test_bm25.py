import math

import numpy as np
import pytest

from shelfrank.bm25.fields import (
    FIELDS,
    BM25Index,
    combine,
    concat_fields,
    feature_text,
    field_tokens,
    score_matrix,
)
from shelfrank.bm25.tuning import coordinate_search, held_out_scores, inner_folds

DOCS = [["red", "sofa"], ["blue", "sofa", "bed"], [], ["lamp"]]


def test_scores_match_the_lucene_formula():
    index = BM25Index(DOCS, k1=1.2, b=0.75)
    n, df = 4, 1
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    out = index.scores(["red"])
    # The empty document is stored as one blank term, so the mean length is 7 / 4.
    expected = idf * 1 / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / (7 / 4)))
    assert out[0] == pytest.approx(expected, rel=1e-4)
    assert out[1] == out[2] == out[3] == 0


def test_rarer_terms_score_higher():
    out = BM25Index(DOCS, 1.2, 0.75)
    assert out.scores(["red"])[0] > out.scores(["sofa"])[0]


def test_unknown_and_empty_queries_score_zero():
    index = BM25Index(DOCS, 1.2, 0.75)
    assert not index.scores(["unicorn"]).any()
    assert not index.scores([]).any()


def test_unknown_terms_do_not_change_known_ones():
    index = BM25Index(DOCS, 1.2, 0.75)
    np.testing.assert_allclose(index.scores(["red", "unicorn"]), index.scores(["red"]))


def test_repeated_query_terms_count_once():
    index = BM25Index(DOCS, 1.2, 0.75)
    np.testing.assert_allclose(index.scores(["sofa", "sofa"]), index.scores(["sofa"]))


def test_field_that_is_always_empty_scores_zero():
    assert BM25Index([[], []], 1.2, 0.75).scores(["red"]).tolist() == [0, 0]


def test_score_matrix_normalises_queries():
    index = BM25Index(DOCS, 1.2, 0.75)
    matrix = score_matrix(index, ["Red Sofas", "red sofa"])
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix[0], matrix[1])


def test_combine_is_a_weighted_sum():
    out = combine({"a": np.array([[1.0, 2.0]]), "b": np.array([[10.0, 0.0]])}, {"a": 2, "b": 0.5})
    assert out.tolist() == [[7.0, 4.0]]


def test_combine_does_not_modify_its_inputs():
    matrix = np.array([[1.0]])
    combine({"a": matrix}, {"a": 3})
    assert matrix.tolist() == [[1.0]]


def test_combine_rejects_unknown_fields():
    with pytest.raises(KeyError, match="title"):
        combine({"name": np.zeros((1, 1))}, {"title": 1})


def test_feature_text_keeps_informative_values_once():
    raw = "color : red|removable : yes|width:64.7|finish : red|style : mid-century| : no"
    assert feature_text(raw) == "red mid-century"


def test_field_tokens_cover_every_field_and_product(sample):
    tokens = field_tokens(sample["products"])
    assert tuple(tokens) == FIELDS
    assert all(len(column) == len(sample["products"]) for column in tokens.values())
    assert all(tokens["name"])


def test_missing_cells_become_empty_documents(sample):
    tokens = field_tokens(sample["products"])
    missing = sample["products"]["product_class"].isna().to_list()
    assert [not t for t in tokens["class"]] == missing


def test_concat_fields_joins_per_product():
    tokens = {"a": [["x"], []], "b": [["y"], ["z"]]}
    assert concat_fields(tokens, ["a", "b"]) == [["x", "y"], ["z"]]


def search(objective, min_gain=0.0005):
    return coordinate_search(["n", "a", "b"], "n", [0.0, 1.0, 2.0], objective, min_gain)


def test_search_finds_a_separable_optimum_and_keeps_the_anchor():
    out = search(lambda w: -((w["a"] - 2) ** 2) - (w["b"] - 0) ** 2 - 5 * (w["n"] != 1))
    assert out.weights == {"n": 1.0, "a": 2.0, "b": 0.0}


def test_search_ignores_gains_below_the_threshold():
    out = search(lambda w: 0.0001 * w["a"])
    assert out.weights == {"n": 1.0, "a": 1.0, "b": 1.0}


def test_search_logs_every_distinct_configuration_once():
    calls = []
    out = search(lambda w: calls.append(tuple(w.values())) or 0.0)
    assert len(calls) == len(set(calls)) == len(out.log)


def test_search_score_belongs_to_the_returned_weights():
    def objective(w):
        return w["a"] - abs(w["b"] - 2)

    out = search(objective)
    assert out.score == objective(out.weights)


def test_inner_folds_partition_deterministically():
    folds = inner_folds(range(10), 3, seed=1)
    assert sorted(q for f in folds for q in f) == list(range(10))
    assert folds == inner_folds(reversed(range(10)), 3, seed=1)


def test_held_out_queries_are_never_tuned_on():
    tuned_on = []

    def tune(ids):
        tuned_on.append(set(ids))
        return {}

    def score(_, ids):
        assert not set(ids) & tuned_on[-1]
        return {q: 1.0 for q in ids}

    out, chosen = held_out_scores(list(range(8)), 4, 0, tune, score)
    assert sorted(out) == list(range(8)) and len(chosen) == 4
