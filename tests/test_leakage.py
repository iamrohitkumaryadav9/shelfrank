"""Protocol guards. Every phase that derives a dataset adds its file here."""

import json
import re
from collections import Counter
from itertools import combinations

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.data.taxonomy import build_taxonomy, class_support
from shelfrank.data.wands import load_products, load_queries
from shelfrank.eval.leakage import exact_matches, key_hashes, load_key_hashes, near_matches
from shelfrank.eval.splits import assign_folds
from shelfrank.qu.guidelines import example_queries, quoted_phrases
from shelfrank.text.near_duplicates import jaccard, leakage_keys, near_duplicate_groups, token_key

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
    # The failure analysis shows queries. It was written after the freeze and the final run.
    assert (REPO_ROOT / "configs/frozen.yaml").exists() or not (
        REPO_ROOT / "results/failure_cases.json"
    ).exists()
    skip |= {"failure_cases.json", "failure_diagnoses.json", "RESULTS.md"}
    found = []
    for folder in ("configs", "data/splits", "docs", "results", "shelfrank", "scripts", "tests"):
        for path in (REPO_ROOT / folder).rglob("*"):
            in_fixture = "fixtures" in path.parts and path.name != "query.csv"
            if path.suffix not in TEXT_SUFFIXES or path.name in skip or in_fixture:
                continue
            text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower())
            found += [(path.name, q) for q in needles if re.search(rf"\b{re.escape(q)}\b", text)]
    assert found == []


KEYS = REPO_ROOT / "data/splits/wands_query_keys.json"
SEED_FILES = ["esci_seed_queries.txt", "esci_prompt_dev.txt"]


def seed_lines(name):
    path = REPO_ROOT / "data/esci" / name
    if not path.exists():
        pytest.skip(f"{name} has not come back from Colab yet")
    return path.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("a", "b"),
    [("Red Chairs", "chair, red"), ("bath vanity", "bath vanities"), ("grey rug", "gray rugs")],
)
def test_key_hashes_match_across_order_case_plural_and_spelling(a, b):
    assert key_hashes(a) & key_hashes(b)


def test_key_hashes_differ_for_different_queries():
    assert not key_hashes("red chair") & key_hashes("blue chair")


def test_frozen_fold_key_is_unchanged():
    # Changing this key would change which queries share a fold.
    assert token_key("bath vanities") == frozenset({"bath", "vanitie"})


def test_planted_query_is_found_by_hash_and_by_similarity():
    assert exact_matches(["lamp", "chairs red"], key_hashes("red chair")) == [1]
    assert near_matches(["a1 b1 c1 d1", "zz"], ["a1 b1 c1"], 0.75) == [0]


def test_fixture_queries_are_in_the_committed_hashes(sample):
    hashes = load_key_hashes(KEYS)
    assert exact_matches(sample["queries"]["query"], hashes) == list(range(5))


@pytest.mark.parametrize("name", SEED_FILES)
def test_seed_files_contain_no_wands_query(name):
    assert exact_matches(seed_lines(name), load_key_hashes(KEYS)) == []


def test_class_terms_and_taxonomy_come_from_no_query(data_cfg):
    code = (REPO_ROOT / "shelfrank/qu/class_terms.py").read_text(encoding="utf-8")
    assert "load_queries" not in code and "query_class" not in code


@pytest.mark.full_data
@pytest.mark.parametrize("name", SEED_FILES)
def test_seed_files_contain_no_near_duplicate(name, raw_dir, data_cfg):
    threshold = data_cfg["splits"]["near_duplicate_jaccard"]
    assert near_matches(seed_lines(name), load_queries(raw_dir)["query"], threshold) == []


@pytest.mark.full_data
def test_committed_hashes_cover_every_query(raw_dir):
    queries = load_queries(raw_dir)["query"]
    assert set().union(*map(key_hashes, queries)) == load_key_hashes(KEYS)


@pytest.mark.full_data
def test_stronger_key_finds_no_cross_fold_pair(raw_dir, data_cfg, folds):
    queries = load_queries(raw_dir).set_index("query_id")["query"].to_dict()
    threshold = data_cfg["splits"]["near_duplicate_jaccard"]
    keys = {q: leakage_keys(text) for q, text in queries.items()}
    crossing = [
        (a, b)
        for a, b in combinations(sorted(keys), 2)
        if folds.assignment[a] != folds.assignment[b]
        and any(jaccard(x, y) >= threshold for x in keys[a] for y in keys[b])
    ]
    assert crossing == []


@pytest.mark.full_data
def test_prompt_examples_resemble_no_wands_query(raw_dir, data_cfg):
    threshold = data_cfg["splits"]["near_duplicate_jaccard"]
    texts = example_queries() + quoted_phrases()
    assert near_matches(texts, load_queries(raw_dir)["query"], threshold) == []


def test_no_prompt_phrase_is_a_wands_query():
    assert len(quoted_phrases()) > 20
    assert exact_matches(example_queries() + quoted_phrases(), load_key_hashes(KEYS)) == []


def test_class_names_are_catalogue_vocabulary_and_the_overlap_is_documented(data_cfg):
    """Some class names equal a WANDS query. The classifier has to list them anyway."""
    path = REPO_ROOT / data_cfg["taxonomy"]["path"]
    classes = json.loads(path.read_text(encoding="utf-8"))["classes"]
    overlap = len(exact_matches(classes, load_key_hashes(KEYS)))
    protocol = (REPO_ROOT / "docs/PROTOCOL.md").read_text(encoding="utf-8")
    assert f"{overlap} of {len(classes)} class names" in protocol


def test_wands_label_files_hold_dev_queries_only_before_the_freeze(folds):
    if (REPO_ROOT / "configs/frozen.yaml").exists():
        pytest.skip("frozen")
    files = list((REPO_ROOT / "data/labels").glob("*wands*.jsonl"))
    assert files
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert {row["id"] for row in rows} <= set(folds.dev)


def test_label_rows_carry_their_rules_version_and_match_their_file_name():
    for path in (REPO_ROOT / "data/labels").glob("*_*-v*.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert {row["rules"] for row in rows} == {path.stem.rsplit("_", 1)[1]}
