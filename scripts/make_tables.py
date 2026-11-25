"""Generate every table and number that appears in the README and docs, from committed results."""

import json
from typing import Any

from shelfrank.config import REPO_ROOT
from shelfrank.dense.checks_report import render as render_checks
from shelfrank.eval.dense_report import render as render_dense
from shelfrank.eval.retrieval_report import render
from shelfrank.ltr.report import render as render_ltr
from shelfrank.readme_tables import findings, limitations, replace_block


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    """Rewrite each generated file in place."""
    readme = REPO_ROOT / "README.md"
    text = replace_block(
        readme.read_text(encoding="utf-8"), "limitations", limitations(_load("data_audit"))
    )
    text = replace_block(text, "findings", findings(_load("retrieval_dev"), _load("bm25_tuning")))
    readme.write_text(text, encoding="utf-8", newline="\n")
    page = render(_load("retrieval_dev"), _load("bm25_tuning"))
    (REPO_ROOT / "docs/retrieval_dev.md").write_text(page, encoding="utf-8", newline="\n")
    page = render_ltr(_load("ltr_dev"))
    (REPO_ROOT / "docs/ltr_dev.md").write_text(page, encoding="utf-8", newline="\n")
    page = render_dense(_load("dense_dev"))
    (REPO_ROOT / "docs/dense_dev.md").write_text(page, encoding="utf-8", newline="\n")
    page = render_checks(_load("phase3_checks"), _load("finetune_esci"))
    (REPO_ROOT / "docs/phase3_checks.md").write_text(page, encoding="utf-8", newline="\n")
    print("updated README.md and the generated pages in docs")


if __name__ == "__main__":
    main()
