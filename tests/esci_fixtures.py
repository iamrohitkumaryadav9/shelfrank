"""Synthetic ESCI files shared by the fine-tuning tests."""

import json

import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.dense import esci_data as ed
from shelfrank.dense.finetune_notebook import build, commit_in

NOTEBOOK = REPO_ROOT / "notebooks/finetune_bge_esci.ipynb"
SETTINGS = {
    "locale": "us",
    "split": "train",
    "n_held_out": 4,
    "max_bullet_chars": 20,
    "near_duplicate_jaccard": 0.75,
    "seed": 1,
}
COLORS = "red blue green black white pink brown gray teal navy".split()
ITEMS = "lamp sofa rug desk shelf clock".split()


@pytest.fixture
def esci(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    examples, products = [], []
    for i, (color, item) in enumerate((c, t) for t in ITEMS for c in COLORS):
        for j, label in enumerate("EESI"):
            product = f"P{i}_{j}"
            row = {
                "query_id": i,
                "query": f"{color} {item} number{chr(97 + i % 26)}{chr(97 + i // 26)}",
                "product_id": product,
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "split": "train",
            }
            examples.append(row)
            title = f"{color} {item}" if label == "E" else "other thing"
            products.append(
                {
                    "product_id": product,
                    "product_locale": "us",
                    "product_title": title,
                    "product_brand": "Acme",
                    "product_color": None,
                    "product_bullet_point": "x" * 100,
                }
            )
    base = {"product_id": "Z", "product_locale": "us", "esci_label": "E"}
    examples += [
        base | {"query_id": 900, "query": "large one", "small_version": 0, "split": "train"},
        base | {"query_id": 901, "query": "test one", "small_version": 1, "split": "test"},
        base
        | {
            "query_id": 902,
            "query": "spanish one",
            "small_version": 1,
            "split": "train",
            "product_locale": "es",
        },
    ]
    pq.write_table(pa.Table.from_pylist(examples), tmp_path / "examples.parquet")
    pq.write_table(pa.Table.from_pylist(products), tmp_path / "products.parquet")
    return tmp_path / "examples.parquet", tmp_path / "products.parquet"


@pytest.fixture
def data(esci):
    return ed.prepare(*esci, ["red lamp numberaa"], SETTINGS)


def rebuilt():
    model = load_config("retrieval")["dense"]["bge_small"]
    committed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return build(REPO_ROOT, load_config("esci"), model, commit_in(committed))
