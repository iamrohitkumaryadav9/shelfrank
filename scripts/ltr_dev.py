"""Build features, train LambdaMART and compare variants and ablations, on the dev fold only.

Writes results/ltr_dev.json and appends every configuration to results/ltr_dev_log.jsonl.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from typing import Any

import joblib
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
from shelfrank.eval.claims import compare
from shelfrank.eval.harness import evaluate, summarise, to_qrels
from shelfrank.eval.splits import load_folds
from shelfrank.features.groups import columns_of, group_of
from shelfrank.features.lexical import lexical_features
from shelfrank.features.priors import prior_features
from shelfrank.features.query_class import class_features
from shelfrank.features.semantic import semantic_features
from shelfrank.features.text_shape import shape_features
from shelfrank.ltr.ablation import ablations, alternatives, training_choices, variants
from shelfrank.ltr.cross_validation import grouped_folds, out_of_fold
from shelfrank.ltr.dataset import JUDGED_ONLY, RankingData, build_data
from shelfrank.ltr.first_stage import FirstStage, candidate_pairs, first_stage
from shelfrank.ltr.gates import allowed_groups
from shelfrank.ltr.model import LtrConfig
from shelfrank.qu.class_student import LinearClassifier, name_examples
from shelfrank.stats.bootstrap import bootstrap_mean
from shelfrank.text.normalize import tokenize

REPORTED = ("ndcg10", "condensed_ndcg10", "judged10", "exact10_share", "mrr_exact")


def class_model(products: pd.DataFrame, taxonomy: list[str]) -> LinearClassifier:
    """The names-only student chosen in results/class_dev.json, cached."""
    best = json.loads((REPO_ROOT / "results/class_dev.json").read_text(encoding="utf-8"))["best"]
    chosen = best["names"]
    low, high = chosen["ngrams"]
    name = f"class_names_tail{chosen['tail']}_n{low}{high}_a{chosen['alpha']:.0e}.joblib"
    path = REPO_ROOT / "artifacts" / name
    if path.exists():
        return joblib.load(path)
    model = LinearClassifier((low, high), chosen["alpha"], 60_000)
    model.fit(name_examples(products, taxonomy, chosen["tail"]))
    joblib.dump(model, path)
    return model


def load_everything() -> tuple[RankingData, FirstStage, dict[int, dict[int, int]], dict[int, str]]:
    """Dev queries, candidates, features and grades."""
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    ids = queries_for("dev", load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    texts = load_queries(raw).set_index("query_id").loc[ids, "query"].tolist()

    dense_cfg = cfg["dense"]["bge_small"]
    identity = model_identity(dense_cfg, REPO_ROOT)
    encode = load_sentence_transformer(**identity)
    folder = REPO_ROOT / "artifacts/embeddings/bge_small"
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
            prior_features(pairs, products),
            shape_features(pairs, query_tokens, tokens),
        ],
        axis=1,
    )
    # Grades are read only now, after every feature exists.
    qrels = to_qrels(judgments(load_labels(raw)), ids)
    return build_data(pairs, features, qrels), stage, qrels, dict(zip(ids, texts, strict=True))


def main() -> None:
    """Run every configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="why this run was made, kept in the log")
    note = parser.parse_args().note
    cfg, retrieval = load_config("ltr"), load_config("retrieval")
    groups = allowed_groups(cfg["groups"])
    data, stage, qrels, texts = load_everything()
    columns = columns_of(data.features.columns, groups)
    near = load_config("data")["splits"]["near_duplicate_jaccard"]
    folds = grouped_folds(texts, cfg["inner_folds"], cfg["seed"], near)
    boot = retrieval["bootstrap"]
    base = LtrConfig(
        tuple(columns),
        JUDGED_ONLY,
        params=cfg["params"],
        rounds=cfg["rounds"],
        patience=cfg["patience"],
        valid_share=cfg["valid_share"],
        seeds=tuple(cfg["seeds"]),
    )
    command = ["git", "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()

    per_query = {name: evaluate(run, qrels) for name, run in stage.runs.items()}
    systems: dict[str, Any] = {}

    def report(name: str, reference: str, extra: dict[str, Any]) -> None:
        summary = summarise(per_query[name])
        row: dict[str, Any] = {m: summary[m] for m in REPORTED} | extra
        for metric in ("ndcg10", "condensed_ndcg10"):
            values = [float(v[metric] or 0) for v in per_query[name].values()]
            interval = bootstrap_mean(values, **boot)
            row[metric] = {**summary[metric], "low": interval.low, "high": interval.high}
        row["reference"] = reference
        row["versus"] = {
            other: compare(per_query[name], per_query[other], **boot)
            for other in dict.fromkeys((reference, "hybrid_rrf", "bm25_multi"))
        }
        systems[name] = row
        difference = row["versus"][reference]
        print(
            f"{name:24s} {row['ndcg10']['mean']:.4f}  condensed {row['condensed_ndcg10']['mean']:.4f}"
            f"  vs {reference}: {difference['ndcg10']['mean']:+.4f} / "
            f"{difference['condensed_ndcg10']['mean']:+.4f}  {difference['verdict']}",
            flush=True,
        )

    def run(name: str, config: LtrConfig, reference: str, limit: int | None = None) -> None:
        result = out_of_fold(data, folds, config, limit)
        per_query[name] = evaluate(result.run, qrels)
        importance = {c: float(np.mean([g[c] for g in result.gain])) for c in config.columns}
        by_group: dict[str, float] = {}
        for column, share in importance.items():
            by_group[group_of(column)] = by_group.get(group_of(column), 0.0) + share
        top = sorted(importance.items(), key=lambda x: -x[1])[:10]
        extra = {"trees": result.trees, "gain_by_group": by_group, "top_features": dict(top)}
        report(name, reference, extra)
        entry = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": commit,
            "note": note,
            "name": name,
            "config": config.describe() | {"train_limit": limit},
            "ndcg10": systems[name]["ndcg10"]["mean"],
            "condensed_ndcg10": systems[name]["condensed_ndcg10"]["mean"],
        }
        log = REPO_ROOT / "results/ltr_dev_log.jsonl"
        with log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")

    for name in stage.runs:
        report(name, "hybrid_rrf", {})
    for name, config in variants(base).items():
        run(name, config, "hybrid_rrf")

    # A dev decision that both metrics share. On an exact tie judged-only stays.
    def both(name: str) -> float:
        return systems[name]["ndcg10"]["mean"] + systems[name]["condensed_ndcg10"]["mean"]

    # Stopping rule and size are chosen per variant first, so that neither variant is judged under
    # a rule that happens to suit the other.
    options = dict(variants(base))
    for variant, config in variants(base).items():
        for name, choice in training_choices(config).items():
            options[f"{variant}__{name}"] = choice
            run(f"{variant}__{name}", choice, "hybrid_rrf")
    chosen = max(options, key=both)
    full = options[chosen]
    sizes = [n for n in cfg["learning_curve"] if n < len(texts) - max(map(len, folds))]
    for size in sizes:
        run(f"train_on_{size}", full, chosen, size)
    for name, config in alternatives(full).items():
        run(name, config, chosen)
    for name, config in ablations(full, columns, groups).items():
        run(name, config, chosen)
    payload = {
        "split": "dev",
        "n_queries": len(texts),
        "rows": len(data.query_id),
        "judged_share": float(np.mean(~np.isnan(data.grade))),
        "groups": groups,
        "features": columns,
        "inner_folds": [len(f) for f in folds],
        "class_student": "names only, provisional",
        "chosen": chosen,
        "learning_curve": sizes,
        "systems": systems,
    }
    path = REPO_ROOT / "results/ltr_dev.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path.name)


if __name__ == "__main__":
    main()
