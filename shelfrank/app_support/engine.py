"""Search, understanding and recommendations from the files in app_data.

The features are the same functions that the evaluation uses. A WANDS query is ranked by the models
of its own fold, which never trained on it, so what the demo shows is what was evaluated.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shelfrank.app_support.catalog import Catalog, Product
from shelfrank.app_support.char_tfidf import PlainClassifier
from shelfrank.app_support.sparse_bm25 import SparseBM25
from shelfrank.bm25.fields import FIELDS, combine, feature_text
from shelfrank.data.wands import split_classes
from shelfrank.features.attributes import attribute_features, listed_values
from shelfrank.features.groups import group_of
from shelfrank.features.lexical import lexical_features
from shelfrank.features.query_class import class_features
from shelfrank.features.semantic import semantic_features
from shelfrank.features.text_shape import shape_features
from shelfrank.ltr.first_stage import FirstStage, candidate_pairs
from shelfrank.qu.class_student import ranked_labels
from shelfrank.qu.slot_model import WordTagger
from shelfrank.ranking import rank_all, reciprocal_rank_fusion
from shelfrank.rewrite.fusion import with_rewrites
from shelfrank.rewrite.trigger import Trigger, query_signals
from shelfrank.suggest.model import SuggestModel
from shelfrank.text.normalize import tokenize

QUERY = 0
LTR, RRF, BM25 = "ltr", "rrf", "bm25"
RANKERS = (LTR, RRF, BM25)


@dataclass
class Understanding:
    """What the students read from the query."""

    classes: list[tuple[str, float]]
    slots: dict[str, list[str]]
    rewrite_triggered: bool
    rewrites: list[str]
    wands_fold: int | None


@dataclass
class Hit:
    """One result."""

    product: Product
    score: float
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class Result:
    """A ranking with timing in milliseconds per stage."""

    hits: list[Hit]
    understanding: Understanding
    timing: dict[str, float]
    candidates: int


class _Tokens:
    """Field tokens of a product, read when a feature asks for them."""

    def __init__(self, read: Callable[[int], Any]) -> None:
        self.read = read
        self.cache: dict[int, list[str]] = {}

    def __getitem__(self, row: int) -> list[str]:
        if row not in self.cache:
            self.cache[row] = tokenize(self.read(row))
        return self.cache[row]


class Engine:
    """Loaded once per process."""

    def __init__(self, folder: Path, encode: Callable[[str], np.ndarray]) -> None:
        """Read every artifact. ``encode`` is passed in, so tests need no model download."""
        self.folder, self.encode = folder, encode
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.catalog = Catalog(folder / "catalog.duckdb")
        self.product_ids = np.array(self.catalog.column("product_id"))
        self.classes = [split_classes(c) for c in self.catalog.column("class")]
        self.departments: dict[str, str] = self.meta["departments"]
        self.vectors = np.load(folder / "vectors.npy").astype(np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors /= norms  # undo the rounding of half precision
        self.bm25 = {name: SparseBM25.load(folder / f"bm25_{name}.npz") for name in FIELDS}
        self.student = PlainClassifier.load(folder / "class_student.npz")
        self.tagger = load_tagger(folder / "slot_student.npz", folder / "slot_student.json")
        self.slot_of = {tuple(k.split()): v for k, v in self.meta["slot_of"].items()}
        self.idf: dict[str, float] = json.loads((folder / "idf.json").read_text(encoding="utf-8"))
        with np.load(folder / "priors.npz", allow_pickle=False) as body:
            self.priors = pd.DataFrame(body["values"], columns=[str(c) for c in body["columns"]])
        saved = json.loads((folder / "rewrites.json").read_text(encoding="utf-8"))
        self.rewrites: dict[int, list[str]] = {int(q): t for q, t in saved.items()}
        trigger = self.meta.get("rewrite_trigger")
        self.trigger = Trigger(trigger["signal"], trigger["threshold"]) if trigger else None

    @cached_property
    def boosters(self) -> dict[int, list[Any]]:
        """Models per fold. Read on first use, because the first page does not need them."""
        import lightgbm as lgb

        body = json.loads((self.folder / "rankers.json").read_text(encoding="utf-8"))
        return {int(f): [lgb.Booster(model_str=s) for s in texts] for f, texts in body.items()}

    @cached_property
    def suggester(self) -> SuggestModel:
        """Completions from the ESCI seed queries."""
        text = (self.folder / "suggest_queries.txt").read_text(encoding="utf-8")
        return SuggestModel(text.splitlines())

    def _first_stage(self, query: str, vector: np.ndarray) -> FirstStage:
        terms = tokenize(query)
        fields = {n: self.bm25[n].scores(terms)[None, :] for n in self.meta["bm25_weights"]}
        tuned = combine(fields, self.meta["bm25_weights"])
        cosine = (vector[None, :] @ self.vectors.T).astype(np.float32)
        depth = self.meta["run_depth"]
        runs = {
            "bm25_multi": rank_all(tuned, [QUERY], self.product_ids, depth, positive_only=True),
            "dense": rank_all(cosine, [QUERY], self.product_ids, depth),
        }
        runs["hybrid_rrf"] = {
            QUERY: reciprocal_rank_fusion(
                [runs["bm25_multi"][QUERY], runs["dense"][QUERY]], self.meta["rrf_k"], depth
            )
        }
        return FirstStage([QUERY], [query], self.product_ids, fields, tuned, cosine, runs)

    def _widen(self, stage: FirstStage, texts: Sequence[str]) -> FirstStage:
        """Fuse the rankings of rewrites into the first stage."""
        parts = [self._first_stage(t, self.encode(t)) for t in texts]
        return with_rewrites(
            stage,
            {QUERY: [p.runs["bm25_multi"][QUERY] for p in parts]},
            {QUERY: [p.runs["dense"][QUERY] for p in parts]},
            self.meta["rrf_k"],
            self.meta["run_depth"],
        )

    def features(self, stage: FirstStage, pairs: pd.DataFrame) -> pd.DataFrame:
        """Every feature of the candidates of one query."""
        rows = sorted(set(pairs["pi"].tolist()))
        products = self.catalog.products(rows)
        tokens = {
            "name": _Tokens(lambda r: products[r].name),
            "class": _Tokens(lambda r: products[r].product_class),
            "features": _Tokens(lambda r: feature_text(products[r].features)),
            "description": _Tokens(lambda r: products[r].description),
        }
        query = stage.texts[0]
        query_tokens = [tokenize(query)]
        listed = {
            r: listed_values(p.features, p.hierarchy or None, self.slot_of)
            for r, p in products.items()
        }
        return pd.concat(
            [
                lexical_features(pairs, stage, query_tokens, tokens, self.idf),  # type: ignore[arg-type]
                semantic_features(pairs, stage, self.meta["rrf_k"]),
                class_features(
                    pairs,
                    self.student.probabilities([query]),
                    self.student.classes,
                    self.classes,
                    self.departments,
                ),
                attribute_features(pairs, [self.tagger.slots(query)], listed),
                self.priors.iloc[pairs["pi"].to_numpy()].reset_index(drop=True),
                shape_features(pairs, query_tokens, tokens),  # type: ignore[arg-type]
            ],
            axis=1,
        )

    def _models(self, fold: int | None) -> list[Any]:
        """Models that never saw the query's labels; all of them for a query outside WANDS."""
        if fold is not None:
            return self.boosters[fold]
        return [b for group in self.boosters.values() for b in group]

    def understand(self, query: str, stage: FirstStage | None = None) -> Understanding:
        """Class, slots, trigger and cached rewrites."""
        probabilities = self.student.probabilities([query])
        top = ranked_labels(probabilities, self.student.classes, 3)[0]
        column = {c: i for i, c in enumerate(self.student.classes)}
        classes = [(c, float(probabilities[0, column[c]])) for c in top]
        known = self.catalog.wands_query(query)
        fires = False
        if self.trigger and stage is not None:
            signals = query_signals(stage, self.student, self.tagger)  # type: ignore[arg-type]
            fires = self.trigger.fires(signals[QUERY])
        rewrites = self.rewrites.get(known[0], []) if fires and known else []
        fold = known[1] if known else None
        return Understanding(classes, self.tagger.slots(query), fires, rewrites, fold)

    def search(
        self,
        query: str,
        k: int = 10,
        ranker: str = LTR,
        product_class: str | None = None,
        min_rating: float | None = None,
        rewrites: Sequence[str] | None = None,
        explain: bool = False,
    ) -> Result:
        """Rank the catalogue. Filters are applied to the final ranking, through SQL."""
        if ranker not in RANKERS:
            raise ValueError(f"unknown ranker {ranker}")
        timing: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timing[name] = timing.get(name, 0.0) + 1000 * (now - clock)
            clock = now

        query = " ".join(query.split())
        terms = tokenize(query)
        vector = self.encode(query)
        lap("dense_encode")
        stage = self._first_stage(query, vector)
        lap("bm25_and_dense_search")
        understanding = self.understand(query, stage)
        lap("query_understanding")
        texts = understanding.rewrites if rewrites is None else list(rewrites)
        if texts and ranker != BM25:
            stage = self._widen(stage, texts)
            lap("rewrites")
        allowed = self.catalog.rows_matching(product_class, min_rating)
        contributions: dict[int, dict[str, float]] = {}
        if not terms and not len(stage.runs["bm25_multi"][QUERY]) and ranker == BM25:
            ranked: list[tuple[int, float]] = []
        elif ranker == LTR:
            pairs = candidate_pairs(stage, self.meta["candidate_depth"])
            table = self.features(stage, pairs)[self.meta["columns"]]
            lap("feature_build")
            models = self._models(understanding.wands_fold)
            scores = np.mean([m.predict(table) for m in models], axis=0)
            order = sorted(
                range(len(pairs)), key=lambda i: (-scores[i], int(pairs["product_id"][i]))
            )
            rows = pairs["pi"].tolist()
            ranked = [(rows[i], float(scores[i])) for i in order]
            lap("ltr_score")
            if explain:
                best = order[:k]
                shap = np.mean(
                    [m.predict(table.iloc[best], pred_contrib=True)[:, :-1] for m in models], axis=0
                )
                for i, values in zip(best, shap, strict=True):
                    parts: dict[str, float] = {}
                    for name, value in zip(self.meta["columns"], values, strict=True):
                        parts[group_of(name)] = parts.get(group_of(name), 0.0) + float(value)
                    contributions[rows[i]] = parts
                lap("explain")
        else:
            run = stage.runs["bm25_multi" if ranker == BM25 else "hybrid_rrf"][QUERY]
            row_of = {int(p): i for i, p in enumerate(self.product_ids)}
            ranked = [(row_of[p], 1 / (i + 1)) for i, p in enumerate(run)]
            lap("fuse")
        kept = [(r, s) for r, s in ranked if allowed is None or r in allowed][:k]
        products = self.catalog.products([r for r, _ in kept])
        hits = [Hit(products[r], s, contributions.get(r, {})) for r, s in kept]
        lap("fetch")
        timing["total"] = sum(timing.values())
        return Result(hits, understanding, timing, len(ranked))

    def similar(self, product_id: int, k: int = 10) -> list[Product] | None:
        """Substitutes of a product, or None for an unknown id."""
        if self.catalog.row_of(product_id) is None:
            return None
        ids = self.catalog.similar(product_id, k)
        rows = [self.catalog.row_of(p) for p in ids]
        found = self.catalog.products([r for r in rows if r is not None])
        return [found[r] for r in rows if r is not None]

    def suggest(self, text: str, k: int = 5) -> list[str]:
        """Completions of what has been typed."""
        return self.suggester.complete(text, k)


def save_tagger(tagger: WordTagger, arrays: Path, plain: Path) -> None:
    """Write the slot student as arrays and JSON."""
    body = tagger.to_plain()
    np.savez_compressed(arrays, coef=body["coef"], bias=body["bias"])
    rest = {k: body[k] for k in ("config", "lexicon", "vocabulary", "classes")}
    plain.write_text(json.dumps(rest) + "\n", encoding="utf-8", newline="\n")


def load_tagger(arrays: Path, plain: Path) -> WordTagger:
    """Read it back."""
    body = json.loads(plain.read_text(encoding="utf-8"))
    with np.load(arrays, allow_pickle=False) as numbers:
        body |= {"coef": numbers["coef"], "bias": numbers["bias"]}
    return WordTagger.from_plain(body)
