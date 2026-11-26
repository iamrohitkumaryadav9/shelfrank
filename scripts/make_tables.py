"""Generate every table and number that appears in the README and docs, from committed results."""

import json
from typing import Any

from shelfrank import readme_blocks as rb
from shelfrank.config import REPO_ROOT
from shelfrank.dense.checks_report import render as render_checks
from shelfrank.eval.dense_report import render as render_dense
from shelfrank.eval.final_report import render as render_final
from shelfrank.eval.results_page import render as render_results
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


def readme_blocks() -> dict[str, str]:
    """Every README block that depends on the final run."""
    final = _load("final_run")
    rewrite = REPO_ROOT / "results/rewrite_primary.json"
    ledger = (REPO_ROOT / "results/llm_spend.jsonl").read_text(encoding="utf-8").splitlines()
    ceilings = {s: _load(f"class_ceiling_{s}") for s in ("dev", "primary", "all")}
    return {
        "pitch": rb.pitch(final),
        "key_results": rb.key_results(final, _load("leak_checks")),
        "ablation": rb.ablation(final),
        "understanding": rb.understanding(_load("class_dev"), ceilings, _load("slots_all")),
        "retrieval": rb.retrieval(final, _load("dense_dev")),
        "rewriting": rb.rewriting(_load("rewrite_primary") if rewrite.exists() else None),
        "suggestions": rb.suggestions(_load("suggest")),
        "similar": rb.similar(_load("similar_primary")),
        "latency": rb.latency(_load("latency"), _load("latency_before"), _load("fast_path")),
        "main_findings": rb.findings_final(
            final,
            _load("leak_checks"),
            _load("rewrite_primary") if rewrite.exists() else None,
            _load("latency"),
        ),
        "mapping": rb.mapping(),
        "spend": rb.spend([json.loads(line) for line in ledger]),
    }


def main() -> None:
    """Rewrite each generated file in place."""
    readme = REPO_ROOT / "README.md"
    text = replace_block(
        readme.read_text(encoding="utf-8"), "limitations", limitations(_load("data_audit"))
    )
    text = replace_block(text, "findings", findings(_load("retrieval_dev"), _load("bm25_tuning")))
    for name, body in readme_blocks().items():
        text = replace_block(text, name, body)
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
    if (REPO_ROOT / "results/leak_checks.json").exists():
        notes = json.loads((REPO_ROOT / "docs/failure_diagnoses.json").read_text(encoding="utf-8"))
        page = render_results(
            _load("final_run"),
            _load("leak_checks"),
            _load("failure_cases"),
            notes,
            _load("similar_primary"),
            _load("suggest"),
            (REPO_ROOT / "results/rewrite_primary.json").exists(),
        )
        (REPO_ROOT / "docs/RESULTS.md").write_text(page, encoding="utf-8", newline="\n")
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
