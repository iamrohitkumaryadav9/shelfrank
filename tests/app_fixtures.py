"""A tiny but complete app_data folder, built from the committed 30-product fixture."""

import numpy as np
import pandas as pd
import pytest

from shelfrank.app_support.build import Inputs, build
from shelfrank.app_support.char_tfidf import PlainClassifier
from shelfrank.app_support.engine import Engine
from shelfrank.bm25.fields import FIELDS, field_tokens
from shelfrank.data.departments import department_map
from shelfrank.data.wands import split_classes
from shelfrank.ltr.dataset import UNJUDGED_ZERO, RankingData
from shelfrank.ltr.first_stage import candidate_pairs, first_stage
from shelfrank.ltr.model import LtrConfig, Ranker
from shelfrank.qu.class_student import LinearClassifier, name_examples
from shelfrank.qu.slot_model import TaggerConfig, WordTagger

WEIGHTS = {"name": 1.0, "class": 0.0, "hierarchy": 0.5, "features": 0.1, "description": 0.1}
LEXICON = {
    "color": {"red": 50, "blue": 40},
    "material_finish": {"oak": 60},
    "style": {"modern": 30},
    "product_types": ["coffee table"],
}
DIMENSIONS = 16


def fake_encode(text):
    """A bag of hashed words: deterministic, and similar texts get similar vectors."""
    from shelfrank.text.normalize import tokenize

    vector = np.full(DIMENSIONS, 0.01, dtype=np.float32)
    for token in tokenize(text):
        vector[sum(map(ord, token)) % DIMENSIONS] += 1
    return vector / np.linalg.norm(vector)


def make_inputs(sample, **changes):
    products = sample["products"].sort_values("product_id").reset_index(drop=True)
    tokens = field_tokens(products)
    vectors = np.stack([fake_encode(n) for n in products["product_name"]])
    taxonomy = sorted({c for v in products["product_class"] for c in split_classes(v)})
    student = LinearClassifier((2, 4), 1e-4, 5000).fit(name_examples(products, taxonomy))
    tagger = WordTagger(TaggerConfig(True, 1, 10.0), LEXICON)
    tagger.fit(
        ["red rug", "oak desk", "modern lamp", "rug for bedroom", "lamp"],
        [
            {"color": ["red"]},
            {"material_finish": ["oak"]},
            {"style": ["modern"]},
            {"room": ["bedroom"]},
            {},
        ],
    )
    queries = sample["queries"]
    grades: dict[int, dict[int, int]] = {int(q): {} for q in queries["query_id"]}
    for q, p, g in sample["judged"].itertuples(index=False):
        grades[int(q)][int(p)] = int(g)
    rng = np.random.default_rng(0)
    columns = [
        "g1_bm25_tuned_rel",
        "g2_cosine",
        "g2_rrf_rank",
        "g3_p_class",
        "g4_matches",
        "g5_log_ratings",
        "g6_name_words",
    ]
    frame = pd.DataFrame(rng.normal(size=(400, len(columns))), columns=columns)
    data = RankingData(
        np.repeat(np.arange(20), 20),
        np.arange(400),
        frame,
        (frame["g2_cosine"] > 0).astype(float).to_numpy(),
    )
    config = LtrConfig(
        tuple(columns),
        UNJUDGED_ZERO,
        rounds=8,
        patience=4,
        seeds=(1, 2),
        params={"min_data_in_leaf": 5},
    )
    ranker = Ranker(config).fit(data, list(range(20)))
    texts = [b.model_to_string() for b in ranker.boosters]
    ids = [int(p) for p in products["product_id"]]
    base = {
        "products": products,
        "tokens": tokens,
        "vectors": vectors,
        "classifier": PlainClassifier.from_student(student),
        "tagger": tagger,
        "rankers": {0: texts, 1: texts},
        "columns": columns,
        "departments": department_map(products, 1),
        "slot_of": {("red",): "color", ("oak",): "material_finish"},
        "bm25_weights": WEIGHTS,
        "k1": 1.2,
        "b": 0.75,
        "rrf_k": 60,
        "run_depth": 1000,
        "candidate_depth": 10,
        "default_ranker": "rrf",
        "similar": {ids[0]: ids[1:4]},
        "queries": [
            (int(q), t, 0) for q, t in zip(queries["query_id"], queries["query"], strict=True)
        ],
        "grades": grades,
        "suggest_queries": ["red sofa bed", "red sofa cover", "oak desk"],
        "rewrites": {int(queries["query_id"].iloc[0]): ["red lamp"]},
        "trigger": {"signal": "words", "threshold": 1.0},
    }
    return Inputs(**(base | changes)), student


@pytest.fixture(scope="session")
def app_folder(sample, tmp_path_factory):
    folder = tmp_path_factory.mktemp("app_data")
    inputs, _ = make_inputs(sample)
    build(folder, inputs)
    return folder


@pytest.fixture(scope="session")
def engine(app_folder):
    return Engine(app_folder, fake_encode)


def offline_stage(sample, text):
    """The evaluation's first stage for one query of the fixture."""
    products = sample["products"].sort_values("product_id").reset_index(drop=True)
    tokens = field_tokens(products)
    assert set(FIELDS) == set(tokens)
    vectors = np.stack([fake_encode(n) for n in products["product_name"]]).astype(np.float16)
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    stage = first_stage(
        [0],
        [text],
        products["product_id"].to_numpy(),
        tokens,
        WEIGHTS,
        fake_encode(text)[None, :],
        vectors,
        1.2,
        0.75,
        60,
        1000,
    )
    return products, tokens, stage, candidate_pairs(stage, 10)
