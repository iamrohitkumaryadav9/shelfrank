"""Generate every table and number that appears in the README and docs, from committed results."""

import json
from typing import Any

from shelfrank.config import REPO_ROOT
from shelfrank.dense.checks_report import render as render_checks
from shelfrank.eval.dense_report import render as render_dense
from shelfrank.eval.final_report import render as render_final
from shelfrank.eval.retrieval_report import render
from shelfrank.ltr.report import render as render_ltr
from shelfrank.qu.class_report import render as render_class
from shelfrank.qu.slot_report import render as render_slots
from shelfrank.readme_tables import findings, limitations, replace_block
from shelfrank.rewrite.report import render as render_rewrite
from shelfrank.similar.report import render as render_similar
from shelfrank.suggest.report import render as render_suggest


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
    for name in ("ltr_interim", "ltr_final"):
        if (REPO_ROOT / "results" / f"{name}.json").exists():
            page = render_ltr(_load(name), name)
            (REPO_ROOT / f"docs/{name}.md").write_text(page, encoding="utf-8", newline="\n")
    paths = sorted((REPO_ROOT / "results").glob("class_ceiling_*.json"))
    ceilings = {p.stem.removeprefix("class_ceiling_"): _load(p.stem) for p in paths}
    page = render_class(_load("class_dev"), ceilings)
    (REPO_ROOT / "docs/class_dev.md").write_text(page, encoding="utf-8", newline="\n")
    (REPO_ROOT / "docs/slots_dev.md").write_text(
        render_slots(_load("slots_dev")), encoding="utf-8", newline="\n"
    )
    primary = REPO_ROOT / "results/similar_primary.json"
    page = render_similar(
        _load("similar_dev"), _load("similar_primary") if primary.exists() else None
    )
    (REPO_ROOT / "docs/similar.md").write_text(page, encoding="utf-8", newline="\n")
    if (REPO_ROOT / "results/final_run.json").exists():
        page = render_final(_load("final_run"))
        (REPO_ROOT / "docs/final_results.md").write_text(page, encoding="utf-8", newline="\n")
    if (REPO_ROOT / "results/suggest.json").exists():
        page = render_suggest(_load("suggest"))
        (REPO_ROOT / "docs/suggest.md").write_text(page, encoding="utf-8", newline="\n")
    if (REPO_ROOT / "results/rewrite_dev.json").exists():
        main_split = REPO_ROOT / "results/rewrite_primary.json"
        page = render_rewrite(
            _load("rewrite_dev"), _load("rewrite_primary") if main_split.exists() else None
        )
        (REPO_ROOT / "docs/rewriting.md").write_text(page, encoding="utf-8", newline="\n")
    page = render_dense(_load("dense_dev"))
    (REPO_ROOT / "docs/dense_dev.md").write_text(page, encoding="utf-8", newline="\n")
    page = render_checks(_load("phase3_checks"), _load("finetune_esci"))
    (REPO_ROOT / "docs/phase3_checks.md").write_text(page, encoding="utf-8", newline="\n")
    print("updated README.md and the generated pages in docs")


if __name__ == "__main__":
    main()
