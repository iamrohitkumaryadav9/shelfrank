"""Tune BM25 field weights on the dev fold and record every configuration tried."""

import json

import numpy as np

from shelfrank.bm25.fields import FIELDS, BM25Index, combine, field_tokens, score_matrix
from shelfrank.bm25.tuning import Weights, coordinate_search, held_out_scores
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries
from shelfrank.eval.access import queries_for
from shelfrank.eval.harness import to_qrels
from shelfrank.eval.metrics import score_query
from shelfrank.eval.splits import load_folds
from shelfrank.ranking import top_products


def main() -> None:
    """Search, estimate the held-out score, and save both."""
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    products = load_products(raw).sort_values("product_id")
    ids = products["product_id"].to_numpy()
    dev = queries_for("dev", load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    texts = load_queries(raw).set_index("query_id").loc[dev, "query"].tolist()
    qrels = to_qrels(judgments(load_labels(raw)), dev)
    row = {q: i for i, q in enumerate(dev)}

    bm25 = cfg["bm25"]
    tokens = field_tokens(products)
    matrices = {f: score_matrix(BM25Index(tokens[f], bm25["k1"], bm25["b"]), texts) for f in FIELDS}

    def per_query(weights: Weights, query_ids: list[int]) -> dict[int, float]:
        rows = [row[q] for q in query_ids]
        scores = combine({f: m[rows] for f, m in matrices.items()}, weights)
        return {
            q: score_query(top_products(s, ids, 10, positive_only=True), qrels[q]).ndcg10
            for q, s in zip(query_ids, scores, strict=True)
        }

    tuning = bm25["tuning"]

    def search(query_ids: list[int]):  # noqa: ANN202
        def objective(weights: Weights) -> float:
            return float(np.mean(list(per_query(weights, query_ids).values())))

        return coordinate_search(
            FIELDS, tuning["anchor"], tuning["grid"], objective, tuning["min_gain"]
        )

    best = search(dev)
    held, chosen = held_out_scores(
        dev, tuning["inner_folds"], tuning["seed"], lambda q: search(q).weights, per_query
    )
    equal = per_query({f: 1.0 for f in FIELDS}, dev)
    payload = {
        "split": "dev",
        "weights": best.weights,
        "dev_ndcg10_tuned_in_sample": best.score,
        "dev_ndcg10_held_out": float(np.mean(list(held.values()))),
        "dev_ndcg10_equal_weights": float(np.mean(list(equal.values()))),
        "inner_weights": chosen,
        "held_out_per_query": held,
        "equal_weights_per_query": equal,
        "configs_tried": len(best.log),
        "log": best.log,
    }
    path = REPO_ROOT / "results" / "bm25_tuning.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print({k: v for k, v in payload.items() if not k.endswith(("log", "per_query"))})


if __name__ == "__main__":
    main()
