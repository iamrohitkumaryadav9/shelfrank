import pytest
from app_fixtures import app_folder, engine  # noqa: F401

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from shelfrank.app_support.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(engine)) as running:
        yield running


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "products": 30, "default_ranker": "rrf"}


@pytest.mark.parametrize("ranker", ["ltr", "rrf", "bm25"])
def test_search_response_has_results_understanding_and_timing(client, ranker):
    body = client.get("/search", params={"q": "table lamp", "k": 3, "ranker": ranker}).json()
    assert body["ranker"] == ranker and len(body["results"]) == 3
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert set(body["understanding"]["slots"]) == {"color", "material_finish", "style", "room"}
    assert len(body["understanding"]["classes"]) == 3 and body["timing_ms"]["total"] > 0


def test_default_ranker_comes_from_the_data(client):
    assert client.get("/search", params={"q": "lamp"}).json()["ranker"] == "rrf"


def test_explain_adds_contributions_for_ltr_only_on_request(client):
    base = {"q": "lamp", "ranker": "ltr", "k": 1}
    assert client.get("/search", params=base).json()["results"][0]["contributions"] == {}
    body = client.get("/search", params=base | {"explain": True}).json()
    assert "G2" in body["results"][0]["contributions"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 201},
        {"q": "lamp", "k": 0},
        {"q": "lamp", "k": 51},
        {"q": "lamp", "ranker": "magic"},
        {"q": "lamp", "min_rating": 9},
    ],
)
def test_bad_requests_are_refused(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_filters_reach_the_engine(client):
    body = client.get("/search", params={"q": "table", "min_rating": 4.6}).json()
    assert all(r["average_rating"] >= 4.6 for r in body["results"])


def test_similar_and_unknown_product(client, sample):
    first = int(sample["products"]["product_id"].min())
    body = client.get(f"/similar/{first}", params={"k": 2}).json()
    assert body["product_id"] == first and len(body["results"]) == 2
    assert client.get("/similar/999999999").status_code == 404
    assert client.get("/similar/notanumber").status_code == 422
