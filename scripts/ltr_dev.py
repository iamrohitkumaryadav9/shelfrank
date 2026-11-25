"""Build features, train LambdaMART and compare variants and ablations, on the dev fold only.

Writes results/ltr_<stage>.json and appends every configuration to results/ltr_dev_log.jsonl. The
interim and the final comparison run once each; see shelfrank/ltr/selection.py.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.eval.claims import compare
from shelfrank.eval.harness import evaluate, summarise
from shelfrank.features.groups import columns_of, group_of
from shelfrank.ltr.ablation import ablations, alternatives, training_choices, variants
from shelfrank.ltr.cross_validation import grouped_folds, out_of_fold
from shelfrank.ltr.dataset import JUDGED_ONLY, RankingData
from shelfrank.ltr.gates import allowed_groups
from shelfrank.ltr.model import LtrConfig
from shelfrank.ltr.pipeline import load_everything
from shelfrank.ltr.selection import FINAL, INTERIM, bug_fixes, output_path, select
from shelfrank.stats.bootstrap import bootstrap_mean

REPORTED = ("ndcg10", "condensed_ndcg10", "judged10", "exact10_share", "mrr_exact")


def main() -> None:
    """Run every configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[INTERIM, FINAL])
    parser.add_argument("--bug-fix", default="", help="reason for a rerun, kept in the result")
    args = parser.parse_args()
    cfg, retrieval = load_config("ltr"), load_config("retrieval")
    groups = allowed_groups(cfg["groups"])
    path = output_path(args.stage, groups, args.bug_fix, REPO_ROOT)
    note = args.stage.upper() + (f", bug fix: {args.bug_fix}" if args.bug_fix else "")
    data, stage, qrels, texts = load_everything(cfg["encoder"])
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

    def run(
        name: str,
        config: LtrConfig,
        reference: str,
        limit: int | None = None,
        source: RankingData = data,
    ) -> None:
        result = out_of_fold(source, folds, config, limit)
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

    # Stopping rule and size are chosen per variant first, so that neither variant is judged under
    # a rule that happens to suit the other.
    options = dict(variants(base))
    for variant, config in variants(base).items():
        for name, choice in training_choices(config).items():
            options[f"{variant}__{name}"] = choice
            run(f"{variant}__{name}", choice, "hybrid_rrf")
    chosen = select(systems, options)
    full = options[chosen]
    sizes = [n for n in cfg["learning_curve"] if n < len(texts) - max(map(len, folds))]
    for size in sizes:
        run(f"train_on_{size}", full, chosen, size)
    for name, config in alternatives(full).items():
        run(name, config, chosen)
    for name, config in ablations(full, columns, groups).items():
        run(name, config, chosen)
    # The encoder that the rule did not choose: in candidates and features, with its own RRF.
    other = next(e for e in retrieval["dense"] if e != cfg["encoder"])
    other_data, other_stage, _, _ = load_everything(other)
    per_query["hybrid_rrf_other_encoder"] = evaluate(other_stage.runs["hybrid_rrf"], qrels)
    report("hybrid_rrf_other_encoder", "hybrid_rrf", {})
    run("other_encoder", full, chosen, source=other_data)
    own = compare(per_query["other_encoder"], per_query["hybrid_rrf_other_encoder"], **boot)
    systems["other_encoder"]["versus"]["hybrid_rrf_other_encoder"] = own
    class_results = json.loads((REPO_ROOT / "results/class_dev.json").read_text(encoding="utf-8"))
    payload = {
        "stage": args.stage,
        "bug_fixes": bug_fixes(path, args.bug_fix),
        "selection_rule": cfg["selection_rule"],
        "encoder": cfg["encoder"],
        "other_encoder": other,
        "split": "dev",
        "n_queries": len(texts),
        "rows": len(data.query_id),
        "judged_share": float(np.mean(~np.isnan(data.grade))),
        "groups": groups,
        "features": columns,
        "inner_folds": [len(f) for f in folds],
        "class_student": class_results["selected"],
        "chosen": chosen,
        "learning_curve": sizes,
        "systems": systems,
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path.name)


if __name__ == "__main__":
    main()
