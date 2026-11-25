"""Write app_data: everything the API and the demo read."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from shelfrank.app_support.char_tfidf import PlainClassifier
from shelfrank.app_support.engine import save_tagger
from shelfrank.app_support.sparse_bm25 import SparseBM25
from shelfrank.bm25.fields import FIELDS, BM25Index
from shelfrank.data.departments import departments_of
from shelfrank.data.product_features import parse_features
from shelfrank.data.wands import split_classes
from shelfrank.features.lexical import idf_table
from shelfrank.features.priors import prior_features
from shelfrank.qu.slot_model import WordTagger
from shelfrank.qu.slot_vocab import Term

GRADES = {2: "Exact", 1: "Partial", 0: "Irrelevant"}


@dataclass
class Inputs:
    """What the build needs. Products must be sorted by product_id."""

    products: pd.DataFrame
    tokens: Mapping[str, list[list[str]]]
    vectors: np.ndarray
    classifier: PlainClassifier
    tagger: WordTagger
    rankers: Mapping[int, Sequence[str]]
    columns: Sequence[str]
    departments: Mapping[str, str]
    slot_of: Mapping[Term, str]
    bm25_weights: Mapping[str, float]
    k1: float
    b: float
    rrf_k: int
    run_depth: int
    candidate_depth: int
    default_ranker: str
    similar: Mapping[int, Sequence[int]]
    queries: Sequence[tuple[int, str, int]]
    grades: Mapping[int, Mapping[int, int]]
    suggest_queries: Sequence[str]
    rewrites: Mapping[int, Sequence[str]] = field(default_factory=dict)
    trigger: Mapping[str, Any] | None = None


def write_catalog(path: Path, data: Inputs) -> None:
    """The DuckDB file: products, exploded attributes, judgments, queries, substitutes, rewrites."""
    path.unlink(missing_ok=True)
    products = data.products.reset_index(drop=True)
    rows = range(len(products))
    main = [(c if (c := split_classes(v)) else [None])[0] for v in products["product_class"]]
    frames: dict[str, pd.DataFrame] = {}
    frames["products"] = pd.DataFrame(
        {
            "row_id": rows,
            "product_id": products["product_id"].astype("int64"),
            "name": products["product_name"].fillna(""),
            "class": products["product_class"].fillna(""),
            "hierarchy": products["category_hierarchy"].fillna(""),
            "description": products["product_description"].fillna(""),
            "features": products["product_features"].fillna(""),
            "rating_count": products["rating_count"],
            "average_rating": products["average_rating"],
            "review_count": products["review_count"],
            "department": departments_of(main, dict(data.departments)),
        }
    )
    pairs = [
        (int(p), k, v)
        for p, raw in zip(products["product_id"], products["product_features"], strict=True)
        for k, v in dict.fromkeys(parse_features(raw).pairs)
    ]
    frames["attributes"] = pd.DataFrame(pairs, columns=["product_id", "key", "value"])
    frames["queries"] = pd.DataFrame(data.queries, columns=["query_id", "text", "fold"])
    labels = [(q, p, GRADES[g]) for q, judged in data.grades.items() for p, g in judged.items()]
    frames["judgments"] = pd.DataFrame(labels, columns=["query_id", "product_id", "label"])
    near = [(p, s, i) for p, found in data.similar.items() for i, s in enumerate(found, 1)]
    frames["similar_products"] = pd.DataFrame(
        near, columns=["product_id", "similar_id", "position"]
    )
    texts = [(q, t, i) for q, found in data.rewrites.items() for i, t in enumerate(found, 1)]
    frames["rewrites"] = pd.DataFrame(texts, columns=["query_id", "rewrite", "position"])
    types = {
        "rewrites": "query_id BIGINT, rewrite VARCHAR, position BIGINT",
        "similar_products": "product_id BIGINT, similar_id BIGINT, position BIGINT",
    }
    connection = duckdb.connect(str(path))
    try:
        for name, frame in frames.items():
            connection.register("frame", frame)
            if name in types:  # an empty frame has no usable types
                connection.execute(f"CREATE TABLE {name} ({types[name]})")
                if len(frame):
                    connection.execute(f"INSERT INTO {name} SELECT * FROM frame")
            else:
                connection.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
            connection.unregister("frame")
        connection.execute("CREATE INDEX product_index ON products (product_id)")
    finally:
        connection.close()


def build(folder: Path, data: Inputs) -> None:
    """Write every file."""
    folder.mkdir(parents=True, exist_ok=True)
    write_catalog(folder / "catalog.duckdb", data)
    np.save(folder / "vectors.npy", data.vectors.astype(np.float16))
    for name in FIELDS:
        index = BM25Index(data.tokens[name], data.k1, data.b)
        SparseBM25.from_index(index).save(folder / f"bm25_{name}.npz")
    data.classifier.save(folder / "class_student.npz")
    save_tagger(data.tagger, folder / "slot_student.npz", folder / "slot_student.json")
    priors = prior_features(pd.DataFrame({"pi": range(len(data.products))}), data.products)
    np.savez_compressed(
        folder / "priors.npz",
        values=priors.to_numpy(dtype=np.float64),
        columns=np.array(list(priors.columns)),
    )
    meta = {
        "departments": dict(data.departments),
        "slot_of": {" ".join(k): v for k, v in data.slot_of.items()},
        "bm25_weights": dict(data.bm25_weights),
        "rrf_k": data.rrf_k,
        "run_depth": data.run_depth,
        "candidate_depth": data.candidate_depth,
        "columns": list(data.columns),
        "default_ranker": data.default_ranker,
        "rewrite_trigger": dict(data.trigger) if data.trigger else None,
    }
    texts = {
        "meta.json": json.dumps(meta, indent=1),
        "idf.json": json.dumps(idf_table(data.tokens["name"])),
        "rankers.json": json.dumps({str(f): list(m) for f, m in data.rankers.items()}),
        "suggest_queries.txt": "\n".join(data.suggest_queries),
    }
    for name, text in texts.items():
        (folder / name).write_text(text + "\n", encoding="utf-8", newline="\n")
