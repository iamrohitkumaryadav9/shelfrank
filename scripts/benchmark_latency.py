"""Per-stage latency of the serving engine, and agreement of the demo with the evaluated rankings.

Writes results/latency.json. Dev queries only for timing; agreement covers every WANDS query.
"""

import json
import platform
import time

import numpy as np

from shelfrank.app_support.encoder import QueryEncoder, fetch_model
from shelfrank.app_support.engine import BM25, LTR, RRF, Engine
from shelfrank.config import REPO_ROOT
from shelfrank.ltr.pipeline import load_split

STAGES = {
    "dense_encode": "Dense encode (fp32 ONNX)",
    "bm25_and_dense_search": "BM25 and dense search, fusion",
    "query_understanding": "Query understanding (class, slots, trigger)",
    "rewrites": "Retrieval with cached rewrites",
    "feature_build": "Feature build",
    "ltr_score": "LTR score",
    "fuse": "Rank list",
    "fetch": "Fetch products (SQL)",
    "total": "Total",
}
REPEATS = 3


def main() -> None:
    """Time and compare."""
    import psutil

    process = psutil.Process()
    start = time.perf_counter()
    folder = REPO_ROOT / "app_data"
    engine = Engine(
        folder, QueryEncoder(fetch_model(REPO_ROOT / "artifacts/onnx"), folder / "tokenizer.json")
    )
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    engine.search("warm up lamp", 10, LTR)
    first_seconds = time.perf_counter() - start
    _, texts, _ = load_split("dev")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ranker in (LTR, RRF, BM25):
        for text in texts[:10]:
            engine.search(text, 10, ranker)
        rows = [engine.search(t, 10, ranker).timing for _ in range(REPEATS) for t in texts]
        out[ranker] = {}
        for stage in STAGES:
            values = [r[stage] for r in rows if stage in r]
            if values:
                out[ranker][stage] = {
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "n": len(values),
                }
    ids, texts, _ = load_split("all")
    reference = json.loads((REPO_ROOT / "results/final_rankings.json").read_text(encoding="utf-8"))[
        "top20"
    ]
    agreement = {}
    for ranker, name in ((LTR, "ltr_full"), (RRF, "hybrid_rrf"), (BM25, "bm25_multi")):
        overlap, same = [], []
        for q, text in zip(ids, texts, strict=True):
            hits = [h.product.product_id for h in engine.search(text, 10, ranker, rewrites=[]).hits]
            expected = reference[name][str(q)][:10]
            overlap.append(len(set(hits) & set(expected)) / max(len(expected), 1))
            same.append(hits == expected)
        agreement[ranker] = {
            "mean_top10_overlap": float(np.mean(overlap)),
            "identical_top10": float(np.mean(same)),
            "n": len(ids),
        }
    payload = {
        "machine": platform.processor() or platform.machine(),
        "repeats": REPEATS,
        "queries": 96,
        "load_seconds": load_seconds,
        "first_query_seconds": first_seconds,
        "memory_mb": process.memory_info().rss / 1e6,
        "stages": out,
        "agreement_with_evaluation": agreement,
    }
    path = REPO_ROOT / "results/latency.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
