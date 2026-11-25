"""The one out-of-fold evaluation. Reads configs/frozen.yaml and tunes nothing.

Refuses unless frozen.yaml is committed and equal to the tag preregistered-v1, and refuses a second
run except for a logged bug fix.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from typing import Any

import joblib
import numpy as np

from shelfrank.bm25.fields import BM25Index, field_tokens, score_matrix
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.departments import department_map, departments_of
from shelfrank.data.wands import load_products, load_queries
from shelfrank.eval.claims import compare_with_p
from shelfrank.eval.crosscheck import max_differences
from shelfrank.eval.harness import candidate_report, ceilings, evaluate
from shelfrank.eval.splits import load_folds
from shelfrank.features.groups import columns_of, group_of
from shelfrank.ltr.dataset import VARIANTS, RankingData
from shelfrank.ltr.model import POINTWISE, LtrConfig, Ranker, to_run
from shelfrank.ltr.pipeline import load_everything
from shelfrank.ltr.selection import bug_fixes
from shelfrank.ranking import rank_all

PATH = REPO_ROOT / "results/final_run.json"
TAG = "preregistered-v1"


def check_frozen() -> str:
    """Return the tagged commit, or refuse."""

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    commit = git("rev-list", "-n", "1", TAG)
    for path in (
        "configs/frozen.yaml",
        "configs/ltr.yaml",
        "configs/retrieval.yaml",
        "configs/slot_student.yaml",
        "data/splits/wands_folds.json",
        "shelfrank/features",
    ):
        if git("diff", "--stat", TAG, "--", path):
            raise SystemExit(f"{path} differs from {TAG}")
    return commit


def config_of(frozen: dict[str, Any]) -> LtrConfig:
    """The frozen training configuration."""
    ltr = frozen["ltr"]
    return LtrConfig(
        tuple(ltr["columns"]),
        ltr["variant"],
        ltr["objective"],
        tuple(ltr["label_gain"]),
        ltr["params"],
        ltr["rounds"],
        ltr["patience"],
        ltr["valid_share"],
        ltr["stopping"],
        ltr["stopping_folds"],
        tuple(ltr["seeds"]),
    )


def main() -> None:
    """Run everything once."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug-fix", default="")
    reason = parser.parse_args().bug_fix
    commit = check_frozen()
    if PATH.exists() and not reason:
        raise SystemExit("the final run exists. A rerun needs --bug-fix with the reason.")
    frozen = load_config("frozen")
    data_cfg, retrieval = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    folds = load_folds(REPO_ROOT / data_cfg["splits"]["path"])
    products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    encoder = frozen["retrieval"]["encoder"]
    other = next(e for e in retrieval["dense"] if e != encoder)
    data, stage, qrels, texts = load_everything(encoder, "all", products)
    other_data, other_stage, _, _ = load_everything(other, "all", products)
    ids = sorted(qrels)
    title = score_matrix(
        BM25Index(field_tokens(products)["name"], retrieval["bm25"]["k1"], retrieval["bm25"]["b"]),
        [texts[q] for q in ids],
    )
    runs: dict[str, dict[int, list[int]]] = {
        "bm25_title": rank_all(
            title,
            ids,
            products["product_id"].to_numpy(),
            retrieval["run_depth"],
            positive_only=True,
        ),
        "bm25_multi": stage.runs["bm25_multi"],
        "dense": stage.runs["dense"],
        "dense_esci": other_stage.runs["dense"],
        "hybrid_rrf": stage.runs["hybrid_rrf"],
        "hybrid_rrf_esci": other_stage.runs["hybrid_rrf"],
    }
    full = config_of(frozen)
    groups = frozen["ltr"]["groups"]
    configs: dict[str, tuple[LtrConfig, RankingData]] = {"ltr_full": (full, data)}
    for group in groups:
        rest = [g for g in groups if g != group]
        configs[f"without_{group}"] = (
            replace(full, columns=tuple(columns_of(full.columns, rest))),
            data,
        )
    configs["only_G1"] = (replace(full, columns=tuple(columns_of(full.columns, ["G1"]))), data)
    configs["pointwise"] = (replace(full, objective=POINTWISE), data)
    unused = next(v for v in VARIANTS if v != full.variant)
    configs["other_variant"] = (replace(full, variant=unused), data)
    configs["ltr_other_encoder"] = (full, other_data)

    importance: dict[str, list[dict[str, float]]] = {}
    trees: dict[str, list[list[int]]] = {}
    contributions = {g: [] for g in groups}
    feature_shap: dict[str, list[float]] = {c: [] for c in full.columns}
    explained: dict[str, Any] = {}
    rankers: dict[int, list[str]] = {}
    for name, (config, source) in configs.items():
        runs[name] = {}
        importance[name], trees[name] = [], []
        for fold in range(folds.n_folds):
            held = folds.queries(fold)
            train = [q for q in ids if q not in set(held)]
            ranker = Ranker(config).fit(source, train)
            rows = source.rows(held)
            runs[name] |= to_run(source, rows, ranker.scores(source, rows))
            importance[name].append(ranker.gain_importance())
            trees[name].append(ranker.trees)
            if name == "ltr_full":
                rankers[fold] = [
                    b.model_to_string(num_iteration=b.best_iteration) for b in ranker.boosters
                ]
            if name == "ltr_full":
                for q in held:
                    part = source.rows([q])
                    shap = ranker.contributions(source, part)
                    order = {int(p): i for i, p in enumerate(source.product_id[part])}
                    top = [order[p] for p in runs[name][q][:10]]
                    magnitude = np.abs(shap[top]).mean(axis=0)
                    for c, v in zip(config.columns, magnitude, strict=True):
                        feature_shap[c].append(float(v))
                    for g in groups:
                        cols = [i for i, c in enumerate(config.columns) if group_of(c) == g]
                        contributions[g].append(
                            float(np.abs(shap[top][:, cols].sum(axis=1)).mean())
                        )
                    explained[str(q)] = {
                        str(p): {
                            g: float(
                                shap[
                                    order[p],
                                    [i for i, c in enumerate(config.columns) if group_of(c) == g],
                                ].sum()
                            )
                            for g in groups
                        }
                        for p in runs[name][q][:10]
                    }
        print(name, "done", trees[name], flush=True)

    per_query = {name: evaluate(run, qrels) for name, run in runs.items()}
    queries = load_queries(raw).set_index("query_id").loc[ids]
    mapping = department_map(products)
    departments = dict(zip(ids, departments_of(queries["query_class"], mapping), strict=True))
    classes = {q: (c if isinstance(c, str) else None) for q, c in queries["query_class"].items()}
    boot = frozen["evaluation"]["bootstrap"]
    splits = {"primary": folds.primary, "all": ids, "dev": folds.dev}

    def versus(split: str, system: str, baseline: str) -> dict[str, Any]:
        part = splits[split]
        a = {q: per_query[system][q] for q in part}
        b = {q: per_query[baseline][q] for q in part}
        return compare_with_p(a, b, **boot)

    pairs = {
        "main": [("ltr_full", "hybrid_rrf"), ("ltr_full", "bm25_multi")],
        "retrieval": [
            ("dense", "bm25_multi"),
            ("hybrid_rrf", "bm25_multi"),
            ("dense_esci", "dense"),
            ("hybrid_rrf_esci", "hybrid_rrf"),
            ("bm25_multi", "bm25_title"),
        ],
        "ablation": [(n, "ltr_full") for n in configs if n != "ltr_full"],
    }
    comparisons = {
        split: {
            family: [{"system": a, "baseline": b, **versus(split, a, b)} for a, b in items]
            for family, items in pairs.items()
        }
        for split in splits
    }
    payload = {
        "tagged_commit": commit,
        "bug_fixes": bug_fixes(PATH, reason),
        "queries": {s: len(v) for s, v in splits.items()},
        "primary_ids": folds.primary,
        "dev_ids": folds.dev,
        "rows": len(data.query_id),
        "judged_share": float(np.mean(~np.isnan(data.grade))),
        "ceilings": {s: ceilings({q: qrels[q] for q in v}) for s, v in splits.items()},
        "candidates": {
            s: candidate_report({q: runs["ltr_full"][q] for q in v}, {q: qrels[q] for q in v})
            for s, v in splits.items()
        },
        "ranx_max_difference": max(
            v for n in runs for v in max_differences(runs[n], qrels, per_query[n]).values()
        ),
        "comparisons": comparisons,
        "departments": departments,
        "query_class": classes,
        "importance": {
            n: {c: float(np.mean([f[c] for f in v])) for c in v[0]} for n, v in importance.items()
        },
        "trees": trees,
        "shap_by_group": {g: float(np.mean(v)) for g, v in contributions.items()},
        "shap_by_feature": {c: float(np.mean(v)) for c, v in feature_shap.items()},
        "per_query": per_query,
    }
    PATH.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    top = {
        n: {str(q): run[q][:20] for q in ids}
        for n, run in runs.items()
        if n in ("bm25_multi", "hybrid_rrf", "ltr_full", "dense")
    }
    target = REPO_ROOT / "results/final_rankings.json"
    target.write_text(
        json.dumps({"top20": top, "group_contributions": explained}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    joblib.dump(
        {"columns": list(full.columns), "folds": rankers},
        REPO_ROOT / "artifacts/final_rankers.joblib",
    )
    for row in comparisons["primary"]["main"]:
        print(
            row["system"],
            "-",
            row["baseline"],
            row["verdict"],
            row["ndcg10"],
            row["condensed_ndcg10"],
            row["p"],
        )


if __name__ == "__main__":
    main()
