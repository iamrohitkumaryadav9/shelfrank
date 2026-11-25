import json
import os
import random
import subprocess
import sys

import pandas as pd
import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.qu import seed_selection as ss
from shelfrank.qu.class_terms import build_class_terms, endings, name_phrases
from shelfrank.qu.seed_notebook import EMBEDDED, TERMS, build, dumps

TERM_MAP = {"Desks": ["desk"], "Lamps": ["desk lamp"], "Sofas": ["sofa", "leather sofa"]}
INDEX = ss.term_index(TERM_MAP)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Coffee & Cocktail Tables", {("coffee", "table"), ("cocktail", "table")}),
        ("Dressers & Chests", {("dresser",), ("chest",)}),
        ("Beds", {("bed",)}),
        ("Cabinet and Drawer Pulls", {("cabinet", "pull"), ("drawer", "pull")}),
        ("&", set()),
    ],
)
def test_name_phrases(name, expected):
    assert name_phrases(name) == expected


def test_endings_are_suffixes_only():
    assert endings(["red", "club", "chair"], (1, 2)) == {("chair",), ("club", "chair")}
    assert endings(["chair"], (2,)) == set()


def products(rows):
    return pd.DataFrame(rows, columns=["product_name", "product_class"])


def test_mined_terms_need_support_and_precision():
    rows = [("big club chair", "Accent Chairs")] * 10 + [("tall bar stool", "Bar Stools")] * 9
    terms = build_class_terms(products(rows), ["Accent Chairs", "Bar Stools"], min_products=10)
    assert "club chair" in terms["Accent Chairs"]
    assert "tall bar stool" not in terms["Bar Stools"]


def test_imprecise_endings_are_not_mined():
    rows = [("a wood panel", "X")] * 20 + [("b wood panel", "Y")] * 20
    terms = build_class_terms(products(rows), ["X"])
    assert "wood panel" not in terms["X"]


def test_words_before_the_product_type_are_not_terms():
    rows = [(f"polyester {i} armchair", "Accent Chairs") for i in range(40)]
    flat = build_class_terms(products(rows), ["Accent Chairs"])["Accent Chairs"]
    assert "armchair" in flat and not any("polyester" in t for t in flat)


def test_each_term_has_one_owner():
    rows = [("x sofa", "A")] * 40 + [("y sofa", "B")] * 5
    terms = build_class_terms(products(rows), ["A", "B"])
    flat = [t for items in terms.values() for t in items]
    assert len(flat) == len(set(flat)) and "sofa" in terms["A"]


def test_committed_terms_cover_the_taxonomy():
    terms = json.loads((REPO_ROOT / TERMS).read_text(encoding="utf-8"))
    taxonomy = json.loads((REPO_ROOT / "data/taxonomy.json").read_text(encoding="utf-8"))
    assert list(terms) == taxonomy["classes"]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("LED Desk Lamp for desk and office", "Lamps"),
        ("Brown Leather Sofas", "Sofas"),
        ("sofa side desk", "Sofas"),
        ("phone case", None),
        ("", None),
    ],
)
def test_match_class_prefers_longest_then_earliest(title, expected):
    assert ss.match_class(title, INDEX) == expected


@pytest.mark.parametrize(
    ("query", "usable"),
    [
        ("blue sofa", True),
        ("b07abcdefg", False),
        ("12 34", False),
        ("sofá azul", False),
        (" ".join(["word"] * 11), False),
        ("", False),
    ],
)
def test_is_usable(query, usable):
    assert ss.is_usable(query) is usable


def evidence(text, n=3, votes=None):
    return ss.QueryEvidence(text, [f"p{i}" for i in range(n)], ss.Counter(votes or {}))


def test_filter_drops_protected_and_near_protected_queries():
    queries = {
        "1": evidence("red chair"),
        "2": evidence("Chairs, red"),
        "3": evidence("big soft red wool chair"),
        "4": evidence("green lamp"),
    }
    kept, dropped = ss.filter_queries(queries, ["red chair", "big soft red chair"], 0.75)
    assert list(kept) == ["4"]
    assert dropped["resembles a protected query"] == 3


def test_filter_drops_repeats_after_normalisation():
    kept, dropped = ss.filter_queries({"1": evidence("lamps"), "2": evidence("lamp")}, [], 0.75)
    assert len(kept) == 1 and dropped["repeat"] == 1


def test_pools_need_votes_and_share():
    queries = {
        "home": evidence("a", 4, {"Sofas": 3}),
        "one_vote": evidence("b", 1, {"Sofas": 1}),
        "minor": evidence("c", 10, {"Sofas": 2}),
        "outside": evidence("d", 3),
        "thin": evidence("e", 1),
    }
    pools = ss.assign_pools(queries, min_votes=2, min_share=0.4)
    assert pools == {"Sofas": ["home"], ss.NO_CLASS: ["outside"]}


def test_spread_represents_small_classes_first():
    pools = {"big": [f"b{i}" for i in range(50)], "small": ["s0"]}
    chosen = ss.spread(pools, 10, random.Random(0))
    assert "s0" in chosen and len(chosen) == len(set(chosen)) == 10


def test_spread_stops_when_exhausted():
    assert sorted(ss.spread({"a": ["1"], "b": ["2"]}, 10, random.Random(0))) == ["1", "2"]


def pools_for_select():
    pools = {f"c{k}": [f"{k}-{i}" for i in range(30)] for k in range(10)}
    return {**pools, ss.NO_CLASS: [f"n{i}" for i in range(100)]}


def test_select_is_disjoint_sized_and_reproducible():
    first = ss.select(pools_for_select(), 80, 20, 0.85, seed=1)
    assert first == ss.select(pools_for_select(), 80, 20, 0.85, seed=1)
    seed, dev = first
    assert (len(seed), len(dev)) == (80, 20) and not set(seed) & set(dev)
    assert sum(i.startswith("n") for i in seed + dev) == 15


def test_select_refuses_to_return_a_short_sample():
    with pytest.raises(ValueError, match="only"):
        ss.select({"c": ["1", "2"], ss.NO_CLASS: ["3"]}, 80, 20, 0.85, seed=1)


def test_ambiguous_single_words_are_not_terms_but_longer_ones_are():
    rows = [("glass display case", "Display Cases")] * 40
    terms = build_class_terms(products(rows), ["Display Cases"])["Display Cases"]
    assert "case" not in terms and "display case" in terms


def test_committed_terms_do_not_match_phone_cases():
    terms = json.loads((REPO_ROOT / TERMS).read_text(encoding="utf-8"))
    index = ss.term_index(terms)
    assert ss.match_class("Acme phone case for iPhone", index) is None
    assert ss.match_class("Mid century club chair", index) == "Accent Chairs"


def test_outputs_hold_one_query_per_line(tmp_path):
    info = ss.write_outputs(tmp_path, ["a  b\tc", "d"], ["e\nf"])
    assert (tmp_path / "esci_seed_queries.txt").read_bytes() == b"a b c\nd\n"
    assert (tmp_path / "esci_prompt_dev.txt").read_bytes() == b"e f\n"
    assert info["esci_seed_queries.txt"]["lines"] == 2


def test_committed_notebook_is_up_to_date():
    cfg = load_config("esci")
    saved = (REPO_ROOT / "notebooks/esci_seed_queries.ipynb").read_text(encoding="utf-8")
    assert saved == dumps(build(REPO_ROOT, cfg["seed"], cfg["sources"]))


def test_embedded_modules_import_only_each_other_and_light_packages():
    allowed = {name[:-3].replace("/", ".") for name in EMBEDDED}
    for name in EMBEDDED:
        for line in (REPO_ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("from shelfrank"):
                assert line.split()[1] in allowed
            assert "import pandas" not in line and "import numpy" not in line


def run_notebook(work, settings):
    """Run the notebook's own cells in a fresh process, on files prepared by the test."""
    script = []
    for cell in build(REPO_ROOT, settings, load_config("esci")["sources"])["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if source.startswith("%%writefile "):
            header, body = source.split("\n", 1)
            script.append(
                f"Path({header.split()[1]!r}).write_text({body!r}, encoding='utf-8', newline='\\n')"
            )
        else:
            script.append(source)
    path = work / "cells.py"
    path.write_text("\n".join(script), encoding="utf-8")
    env = {
        **os.environ,
        "SHELFRANK_WORK": str(work / "w"),
        "SHELFRANK_DOWNLOAD": "0",
        "PYTHONUTF8": "1",
    }
    return subprocess.run(
        [sys.executable, "-P", str(path)], capture_output=True, text=True, env=env, cwd=work
    )


def write_parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=50)


def test_notebook_runs_end_to_end_on_synthetic_esci(tmp_path):
    pytest.importorskip("pyarrow")
    rows, items = [], []
    for i in range(60):
        kind = "club chair" if i < 40 else "phone case"
        for j in range(3):
            rows.append(
                (
                    i,
                    f"{kind} model{i:c}x".replace(f"{i:c}", "abcdefgh"[i % 8] * (i + 1)),
                    f"p{i}-{j}",
                    "us",
                    "E",
                    "train",
                )
            )
            items.append((f"p{i}-{j}", f"Acme {kind}", "us"))
    rows += [
        (900, "japanese desk", "j1", "jp", "E", "train"),
        (901, "held out", "t1", "us", "E", "test"),
        (902, "only irrelevant", "i1", "us", "I", "train"),
        (903, "secret wands query", "p0-0", "us", "E", "train"),
    ]
    items += [("p0-0", "ESPAÑOL sofa", "es")]
    names = ["query_id", "query", "product_id", "product_locale", "esci_label", "split"]
    work = tmp_path / "w"
    write_parquet(
        work / "esci/examples.parquet",
        dict(zip(names, map(list, zip(*rows, strict=False)), strict=True)),
    )
    names = ["product_id", "product_title", "product_locale"]
    write_parquet(
        work / "esci/products.parquet",
        dict(zip(names, map(list, zip(*items, strict=False)), strict=True)),
    )
    (work / "wands").mkdir()
    (work / "wands/query.csv").write_text(
        "query_id\tquery\tquery_class\n1\tsecret wands queries\tBeds\n", encoding="utf-8"
    )
    settings = {**load_config("esci")["seed"], "n_seed": 20, "n_dev": 5}
    run = run_notebook(tmp_path, settings)
    assert run.returncode == 0, run.stderr[-2000:]

    out = work / "out"
    manifest = json.loads((out / "esci_seed_manifest.json").read_text(encoding="utf-8"))
    seed = (out / "esci_seed_queries.txt").read_text(encoding="utf-8").splitlines()
    dev = (out / "esci_prompt_dev.txt").read_text(encoding="utf-8").splitlines()
    everything = seed + dev
    assert (len(seed), len(dev)) == (20, 5)
    assert manifest["queries_with_exact"] == 61
    assert manifest["dropped"] == {"resembles a protected query": 1}
    assert manifest["home_candidates"] == 40 and manifest["outside_candidates"] == 20
    assert sum("chair" in q for q in everything) == round(25 * 0.85)
    assert not any(w in q for q in everything for w in ("secret", "japanese", "held", "irrelevant"))
    assert "secret wands" not in run.stdout
