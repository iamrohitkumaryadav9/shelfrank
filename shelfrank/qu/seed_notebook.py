"""Assemble the Colab notebook that selects ESCI seed queries.

The repository is private, so the notebook carries its own copy of the code it runs. Building it from
the package keeps that copy identical to the tested source.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EMBEDDED = (
    "shelfrank/text/normalize.py",
    "shelfrank/text/near_duplicates.py",
    "shelfrank/data/download.py",
    "shelfrank/qu/seed_selection.py",
)
TERMS = "data/class_terms.json"

INTRO = """# ShelfRank: select ESCI seed queries

CPU only, free Colab is enough. Upload this one file, choose **Runtime > Run all**, and save the three
files it downloads into `data/esci/` in the repository.

It reads the ESCI US training queries with their Exact products, keeps queries whose products look like
home goods (plus some that do not), removes anything that resembles a WANDS query, and writes query
strings only. The code cells below are copies of the repository's tested modules."""

CONFIG = """# Every path and setting is here. Nothing below needs editing.
import os, sys, time
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
DEFAULT_WORK = "/content/shelfrank_seed" if IN_COLAB else (
    "/kaggle/working/shelfrank_seed" if Path("/kaggle").exists() else "shelfrank_seed")
WORK_DIR = Path(os.environ.get("SHELFRANK_WORK", DEFAULT_WORK))
EXAMPLES = WORK_DIR / "esci" / "examples.parquet"
PRODUCTS = WORK_DIR / "esci" / "products.parquet"
WANDS_QUERIES = WORK_DIR / "wands" / "query.csv"
OUT_DIR = WORK_DIR / "out"
DOWNLOAD = os.environ.get("SHELFRANK_DOWNLOAD", "1") == "1"

SETTINGS = __SETTINGS__
SOURCES = __SOURCES__
CODE_SHA256 = __HASHES__

for folder in ("shelfrank/text", "shelfrank/data", "shelfrank/qu", "data"):
    (WORK_DIR / folder).mkdir(parents=True, exist_ok=True)
for package in ("shelfrank", "shelfrank/text", "shelfrank/data", "shelfrank/qu"):
    (WORK_DIR / package / "__init__.py").touch()
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR.resolve()))
started = time.time()
print("working in", WORK_DIR.resolve())"""

FETCH = """from shelfrank.data.download import ensure_file

targets = {"examples": EXAMPLES, "products": PRODUCTS, "wands_queries": WANDS_QUERIES}
for name, spec in SOURCES.items():
    if DOWNLOAD:
        fetched = ensure_file(spec["url"], targets[name], spec["sha256"])
        print(("downloaded" if fetched else "already have"), name, f"{time.time() - started:.0f}s")"""

RUN = """import csv, json
from collections import Counter
from shelfrank.qu import seed_selection as ss

terms = json.loads(Path("data/class_terms.json").read_text(encoding="utf-8"))
index = ss.term_index(terms)
queries = ss.read_exact_pairs(EXAMPLES, SETTINGS["locale"], SETTINGS["split"])
print(f"{len(queries):,} queries with an Exact product, {time.time() - started:.0f}s")

# WANDS queries are read by code only, to remove look-alikes. They are never printed or saved.
with WANDS_QUERIES.open(encoding="utf-8", newline="") as handle:
    protected = [row["query"] for row in csv.DictReader(handle, delimiter="\\t")]
kept, dropped = ss.filter_queries(queries, protected, SETTINGS["near_duplicate_jaccard"])
print(f"{len(kept):,} kept, dropped {dict(dropped)}")

titles = ss.add_votes(kept, PRODUCTS, SETTINGS["locale"], index)
pools = ss.assign_pools(kept, SETTINGS["min_votes"], SETTINGS["min_share"])
home = {label: ids for label, ids in pools.items() if label != ss.NO_CLASS}
print(f"{titles:,} titles read, {sum(map(len, home.values())):,} home-goods queries in "
      f"{len(home)} classes, {len(pools.get(ss.NO_CLASS, [])):,} outside, {time.time() - started:.0f}s")"""

SAVE = """import platform
import pyarrow

seed_ids, dev_ids = ss.select(
    pools, SETTINGS["n_seed"], SETTINGS["n_dev"], SETTINGS["home_share"], SETTINGS["seed"])
files = ss.write_outputs(
    OUT_DIR, [kept[i].text for i in seed_ids], [kept[i].text for i in dev_ids])
try:
    import resource
    peak_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
except ImportError:
    peak_gb = None

label_of = {i: label for label, ids in pools.items() for i in ids}
picked = Counter(label_of[i] for i in seed_ids + dev_ids)
manifest = {
    "settings": SETTINGS,
    "sources": SOURCES,
    "code_sha256": CODE_SHA256,
    "files": files,
    "queries_with_exact": len(queries),
    "dropped": dict(sorted(dropped.items())),
    "titles_read": titles,
    "classes_with_candidates": len(home),
    "home_candidates": sum(map(len, home.values())),
    "outside_candidates": len(pools.get(ss.NO_CLASS, [])),
    "picked_outside": picked.get(ss.NO_CLASS, 0),
    "picked_classes": len(set(picked) - {ss.NO_CLASS}),
    "max_picked_per_class": max((n for k, n in picked.items() if k != ss.NO_CLASS), default=0),
    "peak_memory_gb": peak_gb,
    "seconds": round(time.time() - started),
    "python": platform.python_version(),
    "pyarrow": pyarrow.__version__,
}
(OUT_DIR / "esci_seed_manifest.json").write_text(json.dumps(manifest, indent=1) + "\\n", encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "code_sha256")}, indent=1))"""

RETURN = """# Colab downloads the three files. Elsewhere, copy them from OUT_DIR.
for path in sorted(OUT_DIR.iterdir()):
    print(path, path.stat().st_size, "bytes")
try:
    from google.colab import files as colab_files
    for path in sorted(OUT_DIR.iterdir()):
        colab_files.download(str(path))
except ImportError:
    pass"""


def _cell(kind: str, source: str, number: int) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": kind,
        "id": f"cell-{number:02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    return cell


def build(root: Path, settings: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    """Build the notebook as a dictionary."""
    embedded = {name: (root / name).read_text(encoding="utf-8") for name in (*EMBEDDED, TERMS)}
    hashes = {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in embedded.items()}
    config = (
        CONFIG.replace("__SETTINGS__", repr(dict(settings)))
        .replace("__SOURCES__", json.dumps(sources, indent=1))
        .replace("__HASHES__", json.dumps(hashes, indent=1))
    )
    parts = [("markdown", INTRO), ("code", config)]
    parts += [("code", f"%%writefile {name}\n{text}") for name, text in embedded.items()]
    parts += [("code", FETCH), ("code", RUN), ("code", SAVE), ("code", RETURN)]
    return {
        "cells": [_cell(kind, text, i) for i, (kind, text) in enumerate(parts)],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def dumps(notebook: Mapping[str, Any]) -> str:
    """Serialise deterministically, so a rebuild without changes gives the same file."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
