"""Start the demo the way Streamlit Community Cloud does, in an environment with only the base dependencies.

    uv sync --locked                        # no extras: what Community Cloud installs from uv.lock
    uv run --no-sync python scripts/smoke_app.py

It downloads the release files and the query encoder when they are missing, loads the first page, runs two
searches and fails if anything heavy is importable or was imported.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "transformers", "sentence_transformers", "sklearn", "google.genai")


def heavy_installed() -> list[str]:
    """Training-only packages that should not be in the deployed environment."""
    found = []
    for name in HEAVY:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except ModuleNotFoundError:  # a missing parent package
            continue
    return found


def main() -> None:
    """Exit non-zero on any problem."""
    from streamlit.testing.v1 import AppTest

    if found := heavy_installed():
        raise SystemExit(f"heavy packages installed: {found}")
    page = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    page.run()
    if page.exception:
        raise SystemExit(f"first page failed: {page.exception}")
    for query in ("grey velvet sofa", "outdoor bar stool"):
        page.text_input(key="query").set_value(query).run()
        if page.exception:
            raise SystemExit(f"search {query!r} failed: {page.exception}")
        titles = [s.value for s in page.subheader]
        if not any("ShelfRank" in t for t in titles) or not any("BM25" in t for t in titles):
            raise SystemExit(f"search {query!r} drew no rankings: {titles}")
    loaded = sorted({h for h in HEAVY for m in sys.modules if m == h or m.startswith(h + ".")})
    if loaded:
        raise SystemExit(f"heavy packages were imported: {loaded}")
    print(
        "app smoke test passed: first page and two searches, no heavy package installed or imported"
    )


if __name__ == "__main__":
    main()
