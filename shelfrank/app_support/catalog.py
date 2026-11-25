"""The catalogue in DuckDB. Every read of product data goes through SQL here."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

PRODUCT_COLUMNS = (
    "row_id, product_id, name, class, hierarchy, description, features, "
    "rating_count, average_rating, review_count, department"
)
SEARCH_SQL = f"SELECT {PRODUCT_COLUMNS} FROM products WHERE row_id IN (SELECT unnest(?))"
FILTER_SQL = (
    "SELECT row_id FROM products WHERE (? IS NULL OR list_contains(string_split(class, '|'), ?)) "
    "AND (? IS NULL OR average_rating >= ?)"
)
ATTRIBUTE_SQL = "SELECT key, value FROM attributes WHERE product_id = ? ORDER BY key, value"
JUDGMENT_SQL = (
    "SELECT j.product_id, j.label FROM judgments j JOIN queries q USING (query_id) WHERE q.text = ?"
)
QUERY_SQL = "SELECT query_id, fold FROM queries WHERE text = ?"
SIMILAR_SQL = (
    "SELECT s.similar_id, s.position FROM similar_products s "
    "WHERE s.product_id = ? ORDER BY s.position LIMIT ?"
)
REWRITE_SQL = "SELECT rewrite FROM rewrites WHERE query_id = ? ORDER BY position"


@dataclass(frozen=True)
class Product:
    """One catalogue row."""

    row: int
    product_id: int
    name: str
    product_class: str
    hierarchy: str
    description: str
    features: str
    rating_count: float | None
    average_rating: float | None
    review_count: float | None
    department: str


class Catalog:
    """Read-only access. A cursor per call, because the API serves requests from several threads."""

    def __init__(self, path: Path) -> None:
        """Open the file."""
        self.connection = duckdb.connect(str(path), read_only=True)

    def _rows(self, sql: str, parameters: Sequence[Any]) -> list[tuple[Any, ...]]:
        return self.connection.cursor().execute(sql, list(parameters)).fetchall()

    def products(self, rows: Sequence[int]) -> dict[int, Product]:
        """Products by row number."""
        if not len(rows):
            return {}
        found = self._rows(SEARCH_SQL, [[int(r) for r in rows]])
        return {r[0]: Product(*r) for r in found}

    def rows_matching(self, product_class: str | None, min_rating: float | None) -> set[int] | None:
        """Rows that pass the filters, or None without a filter."""
        if product_class is None and min_rating is None:
            return None
        values = [product_class, product_class, min_rating, min_rating]
        return {r[0] for r in self._rows(FILTER_SQL, values)}

    def attributes(self, product_id: int) -> list[tuple[str, str]]:
        """Listed attributes."""
        return [(k, v) for k, v in self._rows(ATTRIBUTE_SQL, [product_id])]

    def labels(self, query: str) -> dict[int, str]:
        """Human labels of a WANDS query, empty for any other query."""
        return {int(p): str(label) for p, label in self._rows(JUDGMENT_SQL, [query])}

    def wands_query(self, query: str) -> tuple[int, int] | None:
        """Id and fold of a WANDS query."""
        found = self._rows(QUERY_SQL, [query])
        return (int(found[0][0]), int(found[0][1])) if found else None

    def rewrites(self, query_id: int) -> list[str]:
        """Cached rewrites of a WANDS query."""
        return [r[0] for r in self._rows(REWRITE_SQL, [query_id])]

    def similar(self, product_id: int, k: int) -> list[int]:
        """Precomputed substitutes, best first."""
        return [int(r[0]) for r in self._rows(SIMILAR_SQL, [product_id, k])]

    def row_of(self, product_id: int) -> int | None:
        """Row number of a product id."""
        found = self._rows("SELECT row_id FROM products WHERE product_id = ?", [product_id])
        return int(found[0][0]) if found else None

    def column(self, name: str) -> list[Any]:
        """One whole column in row order, for start-up."""
        if name not in PRODUCT_COLUMNS.split(", "):
            raise ValueError(f"unknown column {name}")
        return [r[0] for r in self._rows(f"SELECT {name} FROM products ORDER BY row_id", [])]

    def classes(self) -> list[str]:
        """Classes with at least 30 products, for the filter."""
        sql = (
            "SELECT c FROM (SELECT unnest(string_split(class, '|')) AS c FROM products) "
            "WHERE c <> '' GROUP BY c HAVING count(*) >= 30 ORDER BY c"
        )
        return [r[0] for r in self._rows(sql, [])]
