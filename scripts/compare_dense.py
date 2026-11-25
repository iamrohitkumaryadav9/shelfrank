"""Compare the fine-tuned encoder with the off-the-shelf one on dev. Writes results/dense_dev.json."""

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from shelfrank.bm25.fields import field_tokens
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries
from shelfrank.dense.encoder import (
    encode_cached,
    load_sentence_transformer,
    model_identity,
    product_texts,
)
from shelfrank.dense.serving import OnnxEncoder
from shelfrank.eval.access import queries_for
from shelfrank.eval.claims import compare
from shelfrank.eval.dense_report import METRICS, encoder_choice, serving_choice
from shelfrank.eval.harness import evaluate, summarise, to_qrels
from shelfrank.eval.splits import load_folds
from shelfrank.ltr.first_stage import first_stage
from shelfrank.ranking import rank_all
from shelfrank.stats.bootstrap import bootstrap_mean, paired_difference

MODELS = {"bge_small": "bge_small_base", "bge_small_esci": "bge_small_esci"}
REPEATS = 3


def timed(encode: Callable[[str], np.ndarray], texts: list[str]) -> tuple[np.ndarray, list[float]]:
    """Vectors of the first pass, and the time of every call after a warm-up."""
    for text in texts[:10]:
        encode(text)
    vectors, times = [], []
    for _ in range(REPEATS):
        for text in texts:
            start = time.perf_counter()
            vector = encode(text)
            times.append(1000 * (time.perf_counter() - start))
            vectors.append(vector)
    return np.stack(vectors[: len(texts)]), times


def main() -> None:
    """Measure both encoders."""
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    product_ids = products["product_id"].to_numpy()
    ids = queries_for("dev", load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    texts = load_queries(raw).set_index("query_id").loc[ids, "query"].tolist()
    qrels = to_qrels(judgments(load_labels(raw)), ids)
    tokens = field_tokens(products)
    weights = json.loads((REPO_ROOT / "results/bm25_tuning.json").read_text())["weights"]
    passages = product_texts(products)
    boot = cfg["bootstrap"]

    per_query: dict[str, Any] = {}
    serving: dict[str, Any] = {}
    for name, folder in MODELS.items():
        model = cfg["dense"][name]
        identity = model_identity(model, REPO_ROOT)
        encode = load_sentence_transformer(**identity)
        cache = REPO_ROOT / "artifacts/embeddings" / name
        vectors = encode_cached(passages, cache, identity, encode)
        prefixed = [model["query_prefix"] + t for t in texts]
        local = model["revision"] == "local"
        source = str(REPO_ROOT / model["model"]) if local else model["model"]
        options = {} if local else {"revision": model["revision"]}
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        files = REPO_ROOT / "artifacts/models" / folder
        limit = model["max_seq_length"]
        encoders: dict[str, Callable[[str], np.ndarray]] = {
            "pytorch": lambda text, run=encode: run([text])[0],
            "fp32": OnnxEncoder(files / "model_fp32.onnx", tokenizer, limit),
            "int8": OnnxEncoder(files / "model_int8.onnx", tokenizer, limit),
        }
        sizes = {k: (files / f"model_{k}.onnx").stat().st_size / 1e6 for k in ("fp32", "int8")}
        sizes["pytorch"] = sizes["fp32"]
        formats: dict[str, Any] = {}
        reference = None
        for kind, encoder in encoders.items():
            queries, times = timed(encoder, prefixed)
            if reference is None:
                reference = queries
            cosine = (queries * reference).sum(axis=1)
            run = rank_all(queries @ vectors.T, ids, product_ids, cfg["run_depth"])
            means = {m: s["mean"] for m, s in summarise(evaluate(run, qrels)).items()}
            formats[kind] = {m: means[m] for m in ("ndcg10", "condensed_ndcg10", "recall100")} | {
                "cosine_mean": float(cosine.mean()),
                "cosine_min": float(cosine.min()),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "megabytes": sizes[kind],
            }
            print(name, kind, {k: round(v, 4) for k, v in formats[kind].items()}, flush=True)
        choice = serving_choice(formats["int8"]["ndcg10"], formats["pytorch"]["ndcg10"])
        serving[name] = {"formats": formats, "choice": choice}
        assert reference is not None
        stage = first_stage(
            ids,
            texts,
            product_ids,
            tokens,
            weights,
            reference,
            vectors,
            cfg["bm25"]["k1"],
            cfg["bm25"]["b"],
            cfg["rrf"]["k"],
            cfg["run_depth"],
        )
        per_query["bm25_multi"] = evaluate(stage.runs["bm25_multi"], qrels)
        per_query[f"dense_{name}"] = evaluate(stage.runs["dense"], qrels)
        per_query[f"rrf_{name}"] = evaluate(stage.runs["hybrid_rrf"], qrels)

    systems: dict[str, Any] = {}
    for name, scores in per_query.items():
        systems[name] = {}
        for metric in METRICS:
            values = [row[metric] for row in scores.values() if row[metric] is not None]
            systems[name][metric] = asdict(bootstrap_mean(values, **boot))

    def difference(system: str, baseline: str) -> dict[str, Any]:
        row: dict[str, Any] = {"system": system, "baseline": baseline}
        for metric in METRICS:
            usable = [q for q in ids if per_query[system][q][metric] is not None]
            a = [per_query[system][q][metric] for q in usable]
            b = [per_query[baseline][q][metric] for q in usable]
            row[metric] = asdict(paired_difference(a, b, **boot))
        row["verdict"] = compare(per_query[system], per_query[baseline], **boot)["verdict"]
        return row

    differences = [
        difference("dense_bge_small_esci", "dense_bge_small"),
        difference("rrf_bge_small_esci", "rrf_bge_small"),
        difference("dense_bge_small_esci", "bm25_multi"),
        difference("rrf_bge_small_esci", "bm25_multi"),
    ]
    payload = {
        "split": "dev",
        "n_queries": len(ids),
        "latency_repeats": REPEATS,
        "systems": systems,
        "differences": differences,
        "encoder": encoder_choice(differences[0]["verdict"], differences[1]["verdict"]),
        "serving": serving,
    }
    path = REPO_ROOT / "results/dense_dev.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    for row in differences:
        print(row["system"], "-", row["baseline"], row["verdict"])
        for metric in METRICS:
            r = row[metric]
            print(
                f"   {metric:18s} {r['mean']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}] n={r['n']}"
            )
    print("G2 encoder:", payload["encoder"])


if __name__ == "__main__":
    main()
