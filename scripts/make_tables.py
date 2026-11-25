"""Generate every table and number that appears in the README and docs, from committed results."""

import json
from typing import Any

from shelfrank.config import REPO_ROOT
from shelfrank.eval.retrieval_report import render
from shelfrank.readme_tables import limitations, replace_block


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    """Rewrite each generated file in place."""
    readme = REPO_ROOT / "README.md"
    text = replace_block(
        readme.read_text(encoding="utf-8"), "limitations", limitations(_load("data_audit"))
    )
    readme.write_text(text, encoding="utf-8", newline="\n")
    page = render(_load("retrieval_dev"), _load("bm25_tuning"))
    (REPO_ROOT / "docs/retrieval_dev.md").write_text(page, encoding="utf-8", newline="\n")
    print("updated README.md and docs/retrieval_dev.md")


if __name__ == "__main__":
    main()
