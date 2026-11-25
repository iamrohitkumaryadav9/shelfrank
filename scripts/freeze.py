"""Write configs/frozen.yaml from the dev results, once. Its existence unlocks the other folds."""

import json
from typing import Any

import yaml

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.ltr.ablation import training_choices, variants
from shelfrank.ltr.dataset import JUDGED_ONLY
from shelfrank.ltr.model import LtrConfig
from shelfrank.qu.guidelines import CLASS_VERSION, SLOT_VERSION

PATH = REPO_ROOT / "configs/frozen.yaml"
POOL_RULE = (
    "A gain is claimed only if it is positive under both standard and condensed nDCG@10 with the "
    "95% CI excluding 0. Otherwise it is reported as 'direction depends on unjudged handling'."
)


def load(name: str) -> dict[str, Any]:
    """Read a results file."""
    return json.loads((REPO_ROOT / f"results/{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    """Collect every choice."""
    if PATH.exists():
        raise SystemExit("already frozen")
    final, tuning = load("ltr_final"), load("bm25_tuning")
    classes, slots, dense, similar = (
        load("class_dev"),
        load("slots_dev"),
        load("dense_dev"),
        load("similar_dev"),
    )
    cfg, retrieval = load_config("ltr"), load_config("retrieval")
    base = LtrConfig(
        tuple(final["features"]),
        JUDGED_ONLY,
        params=cfg["params"],
        rounds=cfg["rounds"],
        patience=cfg["patience"],
        valid_share=cfg["valid_share"],
        seeds=tuple(cfg["seeds"]),
    )
    options = dict(variants(base))
    for variant, config in variants(base).items():
        options |= {f"{variant}__{n}": c for n, c in training_choices(config).items()}
    chosen = options[final["chosen"]]
    student = classes["best"][classes["selected"]]
    body = {
        "date": "2026-11-26",
        "pool_rule": POOL_RULE,
        "retrieval": {
            "k1": retrieval["bm25"]["k1"],
            "b": retrieval["bm25"]["b"],
            "bm25_weights": tuning["weights"],
            "rrf_k": retrieval["rrf"]["k"],
            "run_depth": retrieval["run_depth"],
            "candidate_depth": retrieval["candidate_depth"],
            "encoder": dense["encoder"],
        },
        "ltr": {
            "option": final["chosen"],
            "selection_rule": cfg["selection_rule"],
            "groups": final["groups"],
            **chosen.describe(),
            "stopping_folds": chosen.stopping_folds,
        },
        "class_student": {
            "selection_rule": classes["selection_rule"],
            "rules": CLASS_VERSION,
            **{k: student[k] for k in ("arm", "query_share", "tail", "ngrams", "alpha")},
        },
        "slot_student": {
            "selection_rule": slots["selection_rule"],
            "rules": SLOT_VERSION,
            "kind": slots["runs"][slots["selected"]]["kind"],
            "config": slots["runs"][slots["selected"]]["config"],
            "reference": "LLM silver labels; human verification pending",
        },
        "serving": {"query_encoder_format": dense["serving"][dense["encoder"]]["choice"]},
        "similar": {"encoder": similar["encoder"], "weights": similar["weights"]},
        "evaluation": {
            "folds": "data/splits/wands_folds.json",
            "scheme": "5-fold out-of-fold; each fold is ranked by a model trained on the other four",
            "primary_queries": "folds 1 to 4",
            "bootstrap": retrieval["bootstrap"],
            "p_value": "two-sided bootstrap; for a claim, the larger of standard and condensed nDCG@10",
            "correction": "Holm within each family",
            "families": {
                "main": ["ltr_full vs hybrid_rrf (primary)", "ltr_full vs bm25_multi (secondary)"],
                "retrieval": [
                    "dense vs bm25_multi",
                    "hybrid_rrf vs bm25_multi",
                    "dense_esci vs dense",
                    "hybrid_rrf_esci vs hybrid_rrf",
                    "bm25_multi vs bm25_title",
                ],
                "ablation": [
                    *(f"without_{g} vs ltr_full" for g in final["groups"]),
                    "only_G1 vs ltr_full",
                    "pointwise vs ltr_full",
                    "other_variant vs ltr_full",
                    "ltr_other_encoder vs ltr_full",
                ],
                "similar": [
                    "rerank vs rrf (primary)",
                    "rrf vs bm25",
                    "bge_small_esci vs bge_small",
                ],
            },
        },
        "rewriting": "trigger tuned on fold 0 and frozen in configs/frozen_rewrite.yaml before its run",
    }
    text = "# Written once by scripts/freeze.py. Tag: preregistered-v1. Never edited.\n"
    PATH.write_text(
        text + yaml.safe_dump(json.loads(json.dumps(body)), sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    print(PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
