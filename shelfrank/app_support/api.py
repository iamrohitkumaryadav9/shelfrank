"""HTTP service. Run: uvicorn shelfrank.app_support.api:create_app --factory"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from shelfrank.app_support.catalog import Product
from shelfrank.app_support.engine import Engine
from shelfrank.config import REPO_ROOT


class ProductOut(BaseModel):
    """A product in a response."""

    product_id: int
    name: str
    product_class: str
    average_rating: float | None
    review_count: float | None


class HitOut(ProductOut):
    """A ranked product."""

    rank: int
    score: float
    contributions: dict[str, float]


class ClassOut(BaseModel):
    """A predicted class."""

    name: str
    probability: float


class UnderstandingOut(BaseModel):
    """What was read from the query."""

    classes: list[ClassOut]
    slots: dict[str, list[str]]
    rewrite_triggered: bool
    rewrites: list[str]


class SearchOut(BaseModel):
    """Response of /search."""

    query: str
    ranker: str
    results: list[HitOut]
    understanding: UnderstandingOut
    timing_ms: dict[str, float]
    candidates: int


class SimilarOut(BaseModel):
    """Response of /similar."""

    product_id: int
    results: list[ProductOut]


class HealthOut(BaseModel):
    """Response of /health."""

    status: Literal["ok"]
    products: int
    default_ranker: str


def product_out(product: Product) -> ProductOut:
    """Copy the public fields."""
    return ProductOut(
        product_id=product.product_id,
        name=product.name,
        product_class=product.product_class or "",
        average_rating=product.average_rating,
        review_count=product.review_count,
    )


def default_engine() -> Engine:
    """Engine from app_data, with the model downloaded if needed."""
    from shelfrank.app_support.assets import ensure_assets
    from shelfrank.app_support.encoder import QueryEncoder, fetch_model

    folder = Path(os.environ.get("SHELFRANK_APP_DATA", REPO_ROOT / "app_data"))
    models = Path(os.environ.get("SHELFRANK_MODELS", REPO_ROOT / "artifacts/onnx"))
    ensure_assets(folder)
    return Engine(folder, QueryEncoder(fetch_model(models), folder / "tokenizer.json"))


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the service. Tests pass a small engine."""
    state: dict[str, Engine] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state["engine"] = engine or default_engine()
        yield

    app = FastAPI(title="ShelfRank", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> HealthOut:
        """Whether the service can answer."""
        current = state["engine"]
        ranker = current.meta["default_ranker"]
        return HealthOut(status="ok", products=len(current.product_ids), default_ranker=ranker)

    @app.get("/search")
    def search(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        ranker: Literal["ltr", "rrf", "bm25"] | None = None,
        product_class: str | None = None,
        min_rating: Annotated[float | None, Query(ge=0, le=5)] = None,
        explain: bool = False,
    ) -> SearchOut:
        """Ranked products with the query-understanding output and timing."""
        current = state["engine"]
        if not q.strip():
            raise HTTPException(422, "the query is blank")
        chosen = ranker or current.meta["default_ranker"]
        out = current.search(q, k, chosen, product_class, min_rating, explain=explain)
        results = [
            HitOut(
                **product_out(h.product).model_dump(),
                rank=i,
                score=h.score,
                contributions=h.contributions,
            )
            for i, h in enumerate(out.hits, 1)
        ]
        u = out.understanding
        return SearchOut(
            query=q,
            ranker=chosen,
            results=results,
            understanding=UnderstandingOut(
                classes=[ClassOut(name=n, probability=p) for n, p in u.classes],
                slots=u.slots,
                rewrite_triggered=u.rewrite_triggered,
                rewrites=u.rewrites,
            ),
            timing_ms=out.timing,
            candidates=out.candidates,
        )

    @app.get("/similar/{product_id}")
    def similar(product_id: int, k: Annotated[int, Query(ge=1, le=10)] = 10) -> SimilarOut:
        """Products that could replace this one."""
        found = state["engine"].similar(product_id, k)
        if found is None:
            raise HTTPException(404, "no such product")
        return SimilarOut(product_id=product_id, results=[product_out(p) for p in found])

    return app
