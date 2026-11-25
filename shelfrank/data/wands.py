"""Load WANDS with explicit schemas and one documented rule for repeated judgments."""

from pathlib import Path

import pandas as pd

GAINS: dict[str, int] = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
CLASS_SEPARATOR = "|"
HIERARCHY_SEPARATOR = " / "

QUERY_COLUMNS = ["query_id", "query", "query_class"]
LABEL_COLUMNS = ["id", "query_id", "product_id", "label"]
PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
]


def _read(path: Path, expected: list[str]) -> pd.DataFrame:
    """Read a tab-separated WANDS file and fail early if the schema is not the audited one.

    ``keep_default_na=False`` with an explicit empty marker stops pandas from turning real text
    such as the word "none" or "null" into a missing value.
    """
    frame = pd.read_csv(path, sep="\t", encoding="utf-8", keep_default_na=False, na_values=[""])
    frame = frame.rename(columns={"category hierarchy": "category_hierarchy"})
    if list(frame.columns) != expected:
        raise ValueError(f"{path.name}: expected columns {expected}, got {list(frame.columns)}")
    return frame


def load_queries(raw_dir: Path) -> pd.DataFrame:
    """Load the queries."""
    return _read(raw_dir / "query.csv", QUERY_COLUMNS)


def load_products(raw_dir: Path) -> pd.DataFrame:
    """Load the catalogue."""
    return _read(raw_dir / "product.csv", PRODUCT_COLUMNS)


def load_labels(raw_dir: Path) -> pd.DataFrame:
    """Load the raw judgment rows, including repeats."""
    labels = _read(raw_dir / "label.csv", LABEL_COLUMNS)
    unknown = set(labels["label"]) - set(GAINS)
    if unknown:
        raise ValueError(f"unknown relevance labels: {sorted(unknown)}")
    return labels


def resolve_grade(grades: list[int]) -> int:
    """Majority vote; a tie takes the lower grade so a disputed item is never over-credited."""
    counts = {grade: grades.count(grade) for grade in set(grades)}
    most = max(counts.values())
    return min(grade for grade, count in counts.items() if count == most)


def judgments(labels: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw rows to one graded judgment per (query, product).

    WANDS repeats some pairs. Left alone, repeats would double-count a product in ideal DCG and
    weight it twice in training.
    """
    graded = labels.assign(grade=labels["label"].map(GAINS))
    grouped = graded.groupby(["query_id", "product_id"], sort=True)["grade"]
    return grouped.agg(lambda grades: resolve_grade(list(grades))).astype(int).reset_index()


def split_classes(value: object) -> list[str]:
    """Split a product_class cell into its atomic classes; a missing cell has none."""
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(CLASS_SEPARATOR) if part.strip()]


def split_hierarchy(value: object) -> list[str]:
    """Split a category path on the spaced slash, since level names may contain a bare slash."""
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(HIERARCHY_SEPARATOR) if part.strip()]
