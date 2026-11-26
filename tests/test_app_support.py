import json

import numpy as np
import pandas as pd
import pytest
from app_fixtures import app_folder, engine, fake_encode, make_inputs, offline_stage  # noqa: F401

from shelfrank.app_support.catalog import Catalog
from shelfrank.app_support.char_tfidf import PlainClassifier, char_wb_ngrams
from shelfrank.app_support.engine import Engine, load_tagger
from shelfrank.app_support.sparse_bm25 import SparseBM25
from shelfrank.bm25.fields import BM25Index
from shelfrank.features.lexical import idf_table, lexical_features
from shelfrank.features.priors import prior_features
from shelfrank.features.semantic import semantic_features
from shelfrank.features.text_shape import shape_features
from shelfrank.text.normalize import tokenize

TEXTS = ["red velvet sofa", "oak table", "a", "lamp lamp lamp", "zzz qqq", "", "outdoor rug 5 ft"]


@pytest.mark.parametrize("text", [*TEXTS, "x  y", "mid-century modern"])
@pytest.mark.parametrize("ngrams", [(2, 4), (3, 5), (1, 1)])
def test_ngrams_equal_scikit_learn(text, ngrams):
    from sklearn.feature_extraction.text import TfidfVectorizer

    analyzer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngrams).build_analyzer()
    assert char_wb_ngrams(text, *ngrams) == analyzer(text)


def test_plain_classifier_equals_the_student(sample):
    inputs, student = make_inputs(sample)
    ours = inputs.classifier.probabilities(TEXTS)
    np.testing.assert_allclose(ours, student.probabilities(TEXTS), atol=1e-5)


def test_saved_classifier_keeps_the_ranking_of_classes(sample, tmp_path):
    inputs, student = make_inputs(sample)
    inputs.classifier.save(tmp_path / "c.npz")
    again = PlainClassifier.load(tmp_path / "c.npz")
    assert again.classes == student.classes
    np.testing.assert_allclose(again.probabilities(TEXTS), student.probabilities(TEXTS), atol=2e-3)


DOCUMENTS = [["red", "sofa"], ["red", "lamp", "lamp"], [], ["table"]]


@pytest.mark.parametrize("query", [["red"], ["lamp", "red", "red"], ["unknown"], []])
def test_sparse_bm25_equals_the_index(query, tmp_path):
    index = BM25Index(DOCUMENTS, 1.2, 0.75)
    SparseBM25.from_index(index).save(tmp_path / "b.npz")
    again = SparseBM25.load(tmp_path / "b.npz")
    np.testing.assert_allclose(again.scores(query), index.scores(query), atol=1e-6)


def test_sparse_bm25_of_an_empty_field(tmp_path):
    index = BM25Index([["x"], []], 1.2, 0.75)
    SparseBM25.from_index(index).save(tmp_path / "b.npz")
    assert SparseBM25.load(tmp_path / "b.npz").scores(["x"]).shape == (2,)


def test_catalog_reads_products_attributes_and_labels(app_folder, sample):
    catalog = Catalog(app_folder / "catalog.duckdb")
    products = catalog.products([0, 3, 3])
    assert set(products) == {0, 3} and products[0].row == 0
    assert catalog.products([]) == {}
    raw = sample["products"].sort_values("product_id").iloc[0]
    assert products[0].name == raw["product_name"] and products[0].product_id == raw["product_id"]
    assert catalog.attributes(int(raw["product_id"]))
    query = sample["queries"].iloc[0]
    labels = catalog.labels(query["query"])
    assert labels and set(labels.values()) <= {"Exact", "Partial", "Irrelevant"}
    assert catalog.labels("not a wands query") == {}
    assert catalog.wands_query(query["query"]) == (int(query["query_id"]), 0)


def test_filters_are_sql_and_none_means_no_filter(app_folder):
    catalog = Catalog(app_folder / "catalog.duckdb")
    assert catalog.rows_matching(None, None) is None
    name = next(c for c in catalog.column("class") if c).split("|")[0]
    rows = catalog.rows_matching(name, None)
    assert rows and all(name in catalog.products([r])[r].product_class.split("|") for r in rows)
    rated = catalog.rows_matching(None, 4.5)
    assert all(catalog.products([r])[r].average_rating >= 4.5 for r in rated)
    both = catalog.rows_matching(name, 4.5)
    assert both == rows & rated


@pytest.mark.parametrize("value", ["'; DROP TABLE products; --", "x' OR '1'='1"])
def test_user_text_is_a_parameter_and_not_sql(app_folder, value):
    catalog = Catalog(app_folder / "catalog.duckdb")
    assert catalog.rows_matching(value, None) == set() and catalog.labels(value) == {}
    assert len(catalog.column("name")) == 30


def test_unknown_column_is_refused(app_folder):
    with pytest.raises(ValueError, match="unknown column"):
        Catalog(app_folder / "catalog.duckdb").column("1; DROP TABLE products")


def test_engine_features_equal_the_evaluation_code(engine, sample):
    """The demo must not have its own version of a feature."""
    text = "oak table with red lamp"
    products, tokens, stage, pairs = offline_stage(sample, text)
    ours_stage = engine._first_stage(text, fake_encode(text))
    assert ours_stage.runs == stage.runs
    ours = engine.features(ours_stage, pairs)
    expected = pd.concat(
        [
            lexical_features(pairs, stage, [tokenize(text)], tokens),
            semantic_features(pairs, stage, 60),
            prior_features(pairs, products),
            shape_features(pairs, [tokenize(text)], tokens),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(
        ours[expected.columns].astype(float), expected.astype(float), atol=1e-5, check_dtype=False
    )


def test_given_idf_changes_nothing(sample):
    _, tokens, stage, pairs = offline_stage(sample, "red oak table")
    query = [tokenize("red oak table")]
    given = lexical_features(pairs, stage, query, tokens, idf_table(tokens["name"]))
    pd.testing.assert_frame_equal(given, lexical_features(pairs, stage, query, tokens))


@pytest.mark.parametrize("ranker", ["ltr", "rrf", "bm25"])
def test_search_returns_ranked_products_with_timing(engine, ranker):
    out = engine.search("table lamp", 5, ranker)
    assert 0 < len(out.hits) <= 5 and len({h.product.product_id for h in out.hits}) == len(out.hits)
    scores = [h.score for h in out.hits]
    assert scores == sorted(scores, reverse=True)
    assert out.timing["total"] == pytest.approx(
        sum(v for k, v in out.timing.items() if k != "total")
    )


def test_ltr_explains_by_feature_group(engine):
    hit = engine.search("table lamp", 3, "ltr", explain=True).hits[0]
    assert set(hit.contributions) == {"G1", "G2", "G3", "G4", "G5", "G6"}


@pytest.mark.parametrize("ranker", ["ltr", "rrf", "bm25"])
@pytest.mark.parametrize("query", ["zzz qqq", "!!!", "the of"])
def test_queries_without_matches_do_not_crash(engine, ranker, query):
    assert isinstance(engine.search(query, 5, ranker).hits, list)


def test_filters_apply_to_results(engine):
    hits = engine.search("table", 10, "rrf", min_rating=4.6).hits
    assert all(h.product.average_rating >= 4.6 for h in hits)
    assert engine.search("table", 10, "rrf", product_class="No Such Class").hits == []


def test_unknown_ranker_is_refused(engine):
    with pytest.raises(ValueError, match="unknown ranker"):
        engine.search("x", 5, "magic")


def test_a_wands_query_uses_the_models_of_its_own_fold_only(engine, sample):
    text = sample["queries"]["query"].iloc[0]
    assert engine.understand(text).wands_fold == 0
    assert engine._models(0) == engine.boosters[0]
    assert len(engine._models(None)) == sum(len(v) for v in engine.boosters.values())


def test_rewrites_appear_only_when_the_trigger_fires_and_are_cached(engine, sample):
    known = sample["queries"]["query"].iloc[0]
    stage = engine._first_stage(known, fake_encode(known))
    out = engine.understand(known, stage)
    assert out.rewrite_triggered and out.rewrites == ["red lamp"]
    other = engine.understand("brand new text", engine._first_stage("x y", fake_encode("x y")))
    assert other.rewrites == []
    assert engine.understand(known).rewrite_triggered is False  # no stage, no signals


def test_no_trigger_means_no_rewriting(sample, tmp_path):
    from shelfrank.app_support.build import build

    inputs, _ = make_inputs(sample, trigger=None)
    build(tmp_path, inputs)
    plain = Engine(tmp_path, fake_encode)
    out = plain.search(sample["queries"]["query"].iloc[0], 3, "rrf")
    assert plain.trigger is None and not out.understanding.rewrites


def test_given_rewrites_widen_retrieval(engine):
    assert engine.search("lamp", 5, "rrf", rewrites=["oak table"]).timing["rewrites"] >= 0


def test_similar_products_and_unknown_ids(engine, sample):
    first = int(sample["products"]["product_id"].min())
    found = engine.similar(first, 2)
    assert len(found) == 2 and first not in {p.product_id for p in found}
    assert engine.similar(-1) is None


def test_suggestions(engine):
    assert engine.suggest("red ")[0] == "red sofa bed"


def test_slot_student_survives_saving(app_folder, sample):
    inputs, _ = make_inputs(sample)
    again = load_tagger(app_folder / "slot_student.npz", app_folder / "slot_student.json")
    for text in TEXTS:
        assert again.slots(text) == inputs.tagger.slots(text)


def test_serving_code_imports_no_heavy_library():
    """The demo is deployed without torch or scikit-learn."""
    import subprocess
    import sys

    code = (
        "import sys, shelfrank.app_support.engine, shelfrank.app_support.build, "
        "shelfrank.app_support.encoder; "
        "bad = {'torch', 'sklearn', 'transformers', 'sentence_transformers'} & set(sys.modules); "
        "sys.exit(str(bad) if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_meta_is_readable_json(app_folder):
    meta = json.loads((app_folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["default_ranker"] in ("ltr", "rrf")


def test_database_is_unpacked_once_from_its_compressed_copy(app_folder, tmp_path):
    import shutil

    from shelfrank.app_support.catalog import unpack

    shutil.copyfile(app_folder / "catalog.duckdb.gz", tmp_path / "catalog.duckdb.gz")
    path = unpack(tmp_path / "catalog.duckdb")
    assert len(Catalog(path).column("name")) == 30
    before = path.stat().st_mtime_ns
    assert unpack(path).stat().st_mtime_ns == before
    assert not list(tmp_path.glob("*.partial"))


def test_committed_app_data_files_stay_below_the_git_limit():
    from shelfrank.config import REPO_ROOT

    folder = REPO_ROOT / "app_data"
    if not folder.exists():
        pytest.skip("app_data is not built")
    for path in folder.iterdir():
        if path.name != "catalog.duckdb":
            assert path.stat().st_size < 95e6, path.name  # also the limit for a release file


@pytest.mark.parametrize("ranker", ["ltr", "rrf"])
@pytest.mark.parametrize("query", ["table lamp", "oak table with red lamp", "zzz", "modern rug"])
def test_precomputed_path_ranks_exactly_like_the_plain_one(app_folder, engine, ranker, query):
    plain = Engine(app_folder, fake_encode, precomputed=False)
    a = engine.search(query, 30, ranker, explain=True)
    b = plain.search(query, 30, ranker, explain=True)
    assert [(h.product.product_id, h.score, h.contributions) for h in a.hits] == [
        (h.product.product_id, h.score, h.contributions) for h in b.hits
    ]


def test_precomputed_features_are_identical(app_folder, engine, sample):
    from shelfrank.ltr.first_stage import candidate_pairs

    plain = Engine(app_folder, fake_encode, precomputed=False)
    stage = engine._first_stage("oak table with red lamp", fake_encode("oak table with red lamp"))
    pairs = candidate_pairs(stage, 10)
    pd.testing.assert_frame_equal(engine.features(stage, pairs), plain.features(stage, pairs))


def test_token_table_round_trip(tmp_path):
    from shelfrank.app_support.product_side import TokenTable

    tokens = {f: [["a", "b"], [], ["b"]] for f in ("name", "class", "features", "description")}
    TokenTable.build(tokens).save(tmp_path / "t.npz")
    table = TokenTable.load(tmp_path / "t.npz")
    assert [table.row("name", r) for r in range(3)] == [["a", "b"], [], ["b"]]
    assert table.view("description")[0] == ["a", "b"]


def test_committed_fast_path_check_covers_every_query():
    from shelfrank.config import REPO_ROOT

    path = REPO_ROOT / "results/fast_path.json"
    if not path.exists():
        pytest.skip("not checked yet")
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body == {"queries": 480, "depth": 100, "identical": 480}


def test_cached_rewrites_are_shown_but_not_applied_unless_the_data_says_so(sample, tmp_path):
    from shelfrank.app_support.build import build

    inputs, _ = make_inputs(sample, apply_rewrites=False)
    build(tmp_path, inputs)
    off = Engine(tmp_path, fake_encode)
    query = sample["queries"]["query"].iloc[0]
    out = off.search(query, 5, "rrf")
    assert out.understanding.rewrites == ["red lamp"] and "rewrites" not in out.timing
    assert "rewrites" in off.search(query, 5, "rrf", rewrites=["red lamp"]).timing


def test_idf_table_gives_the_same_lookups_and_maximum():
    import inspect

    from shelfrank.app_support.product_side import IdfTable
    from shelfrank.features import lexical

    plain = {"a": 1.5, "b": 3.0}
    table = IdfTable(plain)
    assert table["a"] == 1.5 and table.get("z", 9) == 9 and max(table.values()) == 3.0
    for query, words in ((["a", "z"], {"a"}), (["b"], set()), (["z"], {"z"})):
        assert lexical.weighted_coverage(query, words, table) == lexical.weighted_coverage(
            query, words, plain
        )
    # The override is safe only while the feature code reads nothing but the maximum.
    source = inspect.getsource(lexical)
    assert source.count("idf.values()") == 1 and "max(idf.values(), default=1.0)" in source
