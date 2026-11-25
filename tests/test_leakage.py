"""Protocol guards. Every phase that derives a dataset adds its file here."""

import json
import re
from collections import Counter

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.taxonomy import build_taxonomy, class_support
from shelfrank.data.wands import load_products, load_queries
from shelfrank.eval.splits import assign_folds
from shelfrank.text.near_duplicates import near_duplicate_groups

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".yaml", ".csv", ".txt", ".py"}


def test_folds_are_complete_and_equal(folds):
    assert len(folds.assignment) == 480
    assert Counter(folds.assignment.values()) == {f: 96 for f in range(5)}


def test_dev_fold_and_primary_queries_do_not_overlap(folds):
    assert len(folds.primary) == 384 and len(folds.dev) == 96
    assert not set(folds.dev) & set(folds.primary)


def test_fold_file_matches_config(data_cfg, folds):
    cfg = data_cfg["splits"]
    saved = json.loads((REPO_ROOT / cfg["path"]).read_text(encoding="utf-8"))
    assert (folds.n_folds, folds.dev_fold) == (cfg["n_folds"], cfg["dev_fold"])
    assert saved["provenance"]["seed"] == cfg["seed"]
    assert saved["provenance"]["wands_commit"] == data_cfg["wands"]["commit"]


def test_near_duplicate_queries_share_a_fold(data_cfg, folds):
    saved = json.loads((REPO_ROOT / data_cfg["splits"]["path"]).read_text(encoding="utf-8"))
    groups = saved["provenance"]["near_duplicate_groups"]
    assert groups
    assert all(len({folds.assignment[q] for q in group}) == 1 for group in groups)


def test_ci_fixture_contains_dev_queries_only(sample, folds):
    assert set(sample["queries"]["query_id"]) <= set(folds.dev)


def test_taxonomy_code_takes_no_query_input():
    code = (REPO_ROOT / "shelfrank/data/taxonomy.py").read_text(encoding="utf-8")
    body = code.split("def build_taxonomy", 1)[1].split("\ndef ", 1)[0]
    signature = body.split("->", 1)[0]
    assert "quer" not in signature


@pytest.mark.full_data
def test_committed_folds_are_reproducible(raw_dir, data_cfg, folds):
    queries = load_queries(raw_dir).set_index("query_id")
    cfg = data_cfg["splits"]
    classes = {int(q): c if isinstance(c, str) else None for q, c in queries["query_class"].items()}
    texts = {int(q): t for q, t in queries["query"].items()}
    groups = near_duplicate_groups(texts, cfg["near_duplicate_jaccard"])
    assert assign_folds(classes, groups, cfg["n_folds"], cfg["seed"]) == folds.assignment


@pytest.mark.full_data
def test_committed_taxonomy_comes_from_the_catalogue(raw_dir, data_cfg):
    saved = json.loads((REPO_ROOT / data_cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    support = class_support(load_products(raw_dir)["product_class"])
    assert saved["classes"] == build_taxonomy(support, data_cfg["taxonomy"]["min_support"])


@pytest.mark.full_data
def test_no_primary_query_text_in_committed_files(raw_dir, folds):
    """Scan for multi-word primary queries.

    Queries of one or two words are skipped because ordinary prose contains them, so this catches
    pasted queries and does not prove their absence.
    """
    queries = load_queries(raw_dir)
    primary = queries[queries["query_id"].isin(folds.primary)]["query"].str.lower().str.strip()
    needles = [q for q in primary if len(q.split()) >= 4]
    assert len(needles) > 50
    # Taxonomy and audit class names are catalogue vocabulary, not queries.
    skip = {"uv.lock", "taxonomy.json"}
    found = []
    for folder in ("configs", "data/splits", "docs", "results", "shelfrank", "scripts", "tests"):
        for path in (REPO_ROOT / folder).rglob("*"):
            in_fixture = "fixtures" in path.parts and path.name != "query.csv"
            if path.suffix not in TEXT_SUFFIXES or path.name in skip or in_fixture:
                continue
            text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower())
            found += [(path.name, q) for q in needles if re.search(rf"\b{re.escape(q)}\b", text)]
    assert found == []
