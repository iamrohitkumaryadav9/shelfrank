"""Audit WANDS, write results/data_audit.json, docs/data_audit.md and data/taxonomy.json."""

import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.audit import run_audit
from shelfrank.data.audit_report import render
from shelfrank.data.taxonomy import OTHER, build_taxonomy, class_support
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries
from shelfrank.eval.splits import load_folds


def main() -> None:
    """Run the audit end to end."""
    cfg = load_config("data")
    raw = REPO_ROOT / cfg["wands"]["raw_dir"]
    queries, products, labels = load_queries(raw), load_products(raw), load_labels(raw)
    folds = load_folds(REPO_ROOT / cfg["splits"]["path"])
    audit = run_audit(queries, products, labels, judgments(labels), folds)

    min_support = cfg["taxonomy"]["min_support"]
    classes = build_taxonomy(class_support(products["product_class"]), min_support)
    outputs = {
        "results/data_audit.json": json.dumps(audit, indent=1, ensure_ascii=False),
        "docs/data_audit.md": render(audit, min_support, cfg["wands"]["commit"]),
        cfg["taxonomy"]["path"]: json.dumps(
            {"min_support": min_support, "classes": classes, "fallback": OTHER},
            indent=1,
            ensure_ascii=False,
        ),
    }
    for relative, text in outputs.items():
        path = REPO_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {relative}")


if __name__ == "__main__":
    main()
