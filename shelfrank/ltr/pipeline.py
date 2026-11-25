"""From raw data to ranking rows: first stage, features, then grades."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
from shelfrank.features.attributes import Listed, attribute_features, listed_values
from shelfrank.features.lexical import lexical_features
from shelfrank.features.priors import prior_features
from shelfrank.features.query_class import class_features
from shelfrank.features.semantic import semantic_features
from shelfrank.features.text_shape import shape_features
from shelfrank.ltr.class_for_ranking import class_model
from shelfrank.ltr.dataset import RankingData, build_data
from shelfrank.ltr.first_stage import FirstStage, candidate_pairs, first_stage
from shelfrank.ltr.slots_for_ranking import term_slots
from shelfrank.qu.class_student import LinearClassifier
from shelfrank.qu.slot_model import SlotModel, load_slot_model
from shelfrank.qu.slot_vocab import Term
from shelfrank.text.normalize import tokenize


@dataclass
class Resources:
    """Everything that does not depend on the query."""

    products: pd.DataFrame
    tokens: dict[str, list[list[str]]]
    vectors: np.ndarray
    encode: Callable[[list[str]], np.ndarray]
    query_prefix: str
    weights: dict[str, float]
    retrieval: dict[str, Any]
    student: LinearClassifier
    slot_model: SlotModel
    slot_of: dict[Term, str]
    classes: list[list[str]]
    departments: dict[str, str]
    listed: dict[int, Listed] = field(default_factory=dict)

    def listed_for(self, rows: Sequence[int]) -> Mapping[int, Listed]:
        """Attribute values of some products, read once each."""
        for i in rows:
            if i not in self.listed:
                self.listed[i] = listed_values(
                    self.products["product_features"][i],
                    self.products["category_hierarchy"][i],
                    self.slot_of,
                )
        return self.listed


def load_resources(encoder: str, products: pd.DataFrame | None = None) -> Resources:
    """Catalogue, embeddings and the two query-understanding students."""
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    if products is None:
        products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    dense_cfg = cfg["dense"][encoder]
    identity = model_identity(dense_cfg, REPO_ROOT)
    encode = load_sentence_transformer(**identity)
    folder = REPO_ROOT / "artifacts/embeddings" / encoder
    vectors = encode_cached(product_texts(products), folder, identity, encode)
    taxonomy = json.loads((REPO_ROOT / data_cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    return Resources(
        products=products,
        tokens=field_tokens(products),
        vectors=vectors,
        encode=encode,
        query_prefix=dense_cfg["query_prefix"],
        weights=json.loads((REPO_ROOT / "results/bm25_tuning.json").read_text())["weights"],
        retrieval=cfg,
        student=class_model(products, taxonomy["classes"]),
        slot_model=load_slot_model(),
        slot_of=term_slots(),
        classes=[split_classes(c) for c in products["product_class"]],
        departments=department_map(products),
    )


def make_stage(res: Resources, ids: Sequence[int], texts: Sequence[str]) -> FirstStage:
    """First-stage scores and runs of some queries."""
    # One at a time, as in run_retrieval.py: padding changes a vector in the last digits.
    vectors = np.stack([res.encode([res.query_prefix + t])[0] for t in texts])
    cfg = res.retrieval
    return first_stage(
        ids,
        texts,
        res.products["product_id"].to_numpy(),
        res.tokens,
        res.weights,
        vectors,
        res.vectors,
        cfg["bm25"]["k1"],
        cfg["bm25"]["b"],
        cfg["rrf"]["k"],
        cfg["run_depth"],
    )


def make_features(res: Resources, stage: FirstStage, pairs: pd.DataFrame) -> pd.DataFrame:
    """Every feature of every candidate pair."""
    query_tokens = [tokenize(t) for t in stage.texts]
    rows = sorted(set(pairs["pi"].tolist()))
    return pd.concat(
        [
            lexical_features(pairs, stage, query_tokens, res.tokens),
            semantic_features(pairs, stage, res.retrieval["rrf"]["k"]),
            class_features(
                pairs,
                res.student.probabilities(stage.texts),
                res.student.classes,
                res.classes,
                res.departments,
            ),
            attribute_features(
                pairs, [res.slot_model.slots(t) for t in stage.texts], res.listed_for(rows)
            ),
            prior_features(pairs, res.products),
            shape_features(pairs, query_tokens, res.tokens),
        ],
        axis=1,
    )


def load_split(split: str) -> tuple[list[int], list[str], dict[int, dict[int, int]]]:
    """Ids, texts and grades of a split."""
    data_cfg = load_config("data")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    ids = queries_for(split, load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    texts = load_queries(raw).set_index("query_id").loc[ids, "query"].tolist()
    return ids, texts, to_qrels(judgments(load_labels(raw)), ids)


def load_everything(
    encoder: str, split: str = "dev", products: pd.DataFrame | None = None
) -> tuple[RankingData, FirstStage, dict[int, dict[int, int]], dict[int, str]]:
    """Candidates, features and grades of a split, with the named encoder in G2 and candidates."""
    res = load_resources(encoder, products)
    ids, texts, qrels = load_split(split)
    stage = make_stage(res, ids, texts)
    pairs = candidate_pairs(stage, res.retrieval["candidate_depth"])
    features = make_features(res, stage, pairs)
    # Grades are attached only now, after every feature exists.
    return build_data(pairs, features, qrels), stage, qrels, dict(zip(ids, texts, strict=True))
