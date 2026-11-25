"""The page itself, run headless on the fixture catalogue."""

import pytest
from app_fixtures import app_folder, fake_encode  # noqa: F401

from shelfrank.config import REPO_ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def page(app_folder, monkeypatch):
    import streamlit as st

    from shelfrank.app_support import encoder

    monkeypatch.setenv("SHELFRANK_APP_DATA", str(app_folder))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(encoder, "fetch_model", lambda folder: folder / "model.onnx")
    monkeypatch.setattr(encoder, "QueryEncoder", lambda model, tokenizer: fake_encode)
    st.cache_resource.clear()
    test = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60)
    yield test
    st.cache_resource.clear()


def text_of(page):
    parts = [*page.markdown, *page.caption, *page.subheader, *page.title]
    return "\n".join(str(p.value) for p in parts)


def test_starts_without_a_key_and_shows_examples(page):
    page.run()
    assert not page.exception and "Try:" in text_of(page)


def test_search_shows_both_rankings_and_labels(page, sample):
    page.run()
    page.text_input(key="query").set_value(sample["queries"]["query"].iloc[0]).run()
    assert not page.exception
    shown = text_of(page)
    assert "BM25" in shown and "ShelfRank" in shown and "human labels" in shown
    assert "Exact" in shown or "Partial" in shown or "Irrelevant" in shown


def test_free_text_has_no_labels_and_no_key_is_needed(page):
    page.run()
    page.text_input(key="query").set_value("purple bookshelf").run()
    assert not page.exception and "no human label" in text_of(page)


def test_switching_to_ltr_explains_results(page):
    page.run()
    page.selectbox[0].set_value("ltr").run()
    page.text_input(key="query").set_value("table lamp").run()
    assert not page.exception
    assert any(e.label == "Why this ranked here" for e in page.expander)


def test_markup_in_text_is_shown_as_written():
    from shelfrank.app_support.text import plain

    assert plain("a & b") == "a \\& b" and plain("[x](http://y)") == "\\[x\\]\\(http\\://y\\)"
    assert plain("**bold** $x$ <b>") == "\\*\\*bold\\*\\* \\$x\\$ \\<b\\>"


def test_page_never_allows_html():
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source and "st.html" not in source


def test_hostile_query_does_not_break_the_page(page):
    page.run()
    page.text_input(key="query").set_value("<script>alert(1)</script> [x](y) $$ lamp").run()
    assert not page.exception
