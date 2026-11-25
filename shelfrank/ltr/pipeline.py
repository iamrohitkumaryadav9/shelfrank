"""From raw data to ranking rows: first stage, features, then grades."""

import json

import numpy as np
import pandas as pd

from shelfrank.bm25.fields import field_tokens
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.departments import department_map
from shelfrank.data.wands import (
    judgments,
    load_labels,
    load_products,
    load_queries,
    split_classes,
)
from shelfrank.dense.encoder import (
    encode_cached,
    load_sentence_transformer,
    model_identity,
    product_texts,
)
from shelfrank.eval.access import queries_for
from shelfrank.eval.harness import to_qrels
from shelfrank.eval.splits import load_folds
from shelfrank.features.attributes import attribute_features, listed_values
from shelfrank.features.lexical import lexical_features
from shelfrank.features.priors import prior_features
from shelfrank.features.query_class import class_features
from shelfrank.features.semantic import semantic_features
from shelfrank.features.text_shape import shape_features
from shelfrank.ltr.class_for_ranking import class_model
from shelfrank.ltr.dataset import RankingData, build_data
from shelfrank.ltr.first_stage import FirstStage, candidate_pairs, first_stage
from shelfrank.ltr.slots_for_ranking import term_slots
from shelfrank.qu.slot_model import load_slot_model
from shelfrank.text.normalize import tokenize


def load_everything(
    encoder: str, split: str = "dev", products: pd.DataFrame | None = None
) -> tuple[RankingData, FirstStage, dict[int, dict[int, int]], dict[int, str]]:
    """Candidates, features and grades of a split, with the named encoder in G2 and candidates."""
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    if products is None:
        products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    ids = queries_for(split, load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    texts = load_queries(raw).set_index("query_id").loc[ids, "query"].tolist()

    dense_cfg = cfg["dense"][encoder]
    identity = model_identity(dense_cfg, REPO_ROOT)
    encode = load_sentence_transformer(**identity)
    folder = REPO_ROOT / "artifacts/embeddings" / encoder
    vectors = encode_cached(product_texts(products), folder, identity, encode)
    # One at a time, as in run_retrieval.py: padding changes a vector in the last digits.
    prefix = dense_cfg["query_prefix"]
    query_vectors = np.stack([encode([prefix + t])[0] for t in texts])
    tokens = field_tokens(products)
    weights = json.loads((REPO_ROOT / "results/bm25_tuning.json").read_text())["weights"]
    stage = first_stage(
        ids,
        texts,
        products["product_id"].to_numpy(),
        tokens,
        weights,
        query_vectors,
        vectors,
        cfg["bm25"]["k1"],
        cfg["bm25"]["b"],
        cfg["rrf"]["k"],
        cfg["run_depth"],
    )
    pairs = candidate_pairs(stage, cfg["candidate_depth"])
    query_tokens = [tokenize(t) for t in texts]
    taxonomy = json.loads((REPO_ROOT / data_cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    student = class_model(products, taxonomy["classes"])
    slot_model = load_slot_model()
    slot_of = term_slots()
    listed = {
        i: listed_values(
            products["product_features"][i], products["category_hierarchy"][i], slot_of
        )
        for i in sorted(set(pairs["pi"].tolist()))
    }
    features = pd.concat(
        [
            lexical_features(pairs, stage, query_tokens, tokens),
            semantic_features(pairs, stage, cfg["rrf"]["k"]),
            class_features(
                pairs,
                student.probabilities(texts),
                student.classes,
                [split_classes(c) for c in products["product_class"]],
                department_map(products),
            ),
            attribute_features(pairs, [slot_model.slots(t) for t in texts], listed),
            prior_features(pairs, products),
            shape_features(pairs, query_tokens, tokens),
        ],
        axis=1,
    )
    # Grades are read only now, after every feature exists.
    qrels = to_qrels(judgments(load_labels(raw)), ids)
    return build_data(pairs, features, qrels), stage, qrels, dict(zip(ids, texts, strict=True))
