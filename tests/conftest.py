from pathlib import Path

import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries
from shelfrank.eval.splits import load_folds

FIXTURE = Path(__file__).parent / "fixtures" / "wands"


@pytest.fixture(scope="session")
def data_cfg():
    return load_config("data")


@pytest.fixture(scope="session")
def folds(data_cfg):
    return load_folds(REPO_ROOT / data_cfg["splits"]["path"])


@pytest.fixture(scope="session")
def sample():
    labels = load_labels(FIXTURE)
    return {
        "queries": load_queries(FIXTURE),
        "products": load_products(FIXTURE),
        "labels": labels,
        "judged": judgments(labels),
    }


@pytest.fixture(scope="session")
def raw_dir(data_cfg):
    path = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    if not all((path / name).exists() for name in data_cfg["wands"]["files"]):
        pytest.skip("WANDS is not downloaded")
    return path
