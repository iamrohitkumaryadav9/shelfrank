"""ShelfRank demo. Run: streamlit run app.py"""

import os
from pathlib import Path

import streamlit as st

from shelfrank.app_support import catalog as sql
from shelfrank.app_support.assets import ensure_assets
from shelfrank.app_support.encoder import QueryEncoder, fetch_model
from shelfrank.app_support.engine import BM25, LTR, RRF, Engine, Hit, Result
from shelfrank.app_support.gemini import live_rewrites
from shelfrank.app_support.text import plain
from shelfrank.config import REPO_ROOT

DATA = Path(os.environ.get("SHELFRANK_APP_DATA", REPO_ROOT / "app_data"))
MODELS = Path(os.environ.get("SHELFRANK_MODELS", REPO_ROOT / "artifacts/onnx"))
NAMES = {
    LTR: "LTR (LambdaMART)",
    RRF: "Hybrid RRF (BM25 + dense)",
    BM25: "BM25",
}
GROUPS = {
    "G1": "lexical match",
    "G2": "semantic and fusion",
    "G3": "query class",
    "G4": "attributes",
    "G5": "ratings",
    "G6": "text shape",
}
BADGES = {"Exact": "green", "Partial": "orange", "Irrelevant": "red"}
EXAMPLES = ["grey velvet sofa", "round oak coffee table", "outdoor bar stool", "boho bedroom rug"]

st.set_page_config(page_title="ShelfRank", page_icon="🛋️", layout="wide")


@st.cache_resource(show_spinner="Downloading and loading the catalogue and the query encoder…")
def load_engine() -> Engine:
    """One engine per process."""
    ensure_assets(DATA)
    return Engine(DATA, QueryEncoder(fetch_model(MODELS), DATA / "tokenizer.json"))


def badge(label: str | None) -> str:
    """Human label of a result, when WANDS has one for this query."""
    if label is None:
        return ":gray[no human label]"
    return f":{BADGES[label]}[**{label}**]"


def show_hit(engine: Engine, hit: Hit, rank: int, label: str | None, side: str, query: str) -> None:
    """One result with its explanation and substitutes."""
    product = hit.product
    stars = f"★ {product.average_rating:.1f}" if product.average_rating else "no rating"
    reviews = f"({int(product.review_count)} reviews)" if product.review_count else ""
    st.markdown(f"**{rank}. {plain(product.name)}**  \n{badge(label)} · {stars} {reviews}")
    st.caption(plain(product.product_class or "no class"))
    if hit.contributions:
        with st.expander("Why this ranked here"):
            st.caption("Sum of SHAP values per feature group. Positive values move it up.")
            for group, value in sorted(hit.contributions.items(), key=lambda x: -abs(x[1])):
                st.write(f"{GROUPS[group]}: {value:+.3f}")
    with st.expander("More like this"):
        if st.button("Show substitutes", key=f"similar-{side}-{product.product_id}-{query}"):
            for other in engine.similar(product.product_id, 5) or []:
                st.write(f"- {plain(other.name)}")
            st.caption("Precomputed substitutes. Measured as substitutability, not as engagement.")


def show_column(
    engine: Engine, title: str, result: Result, labels: dict[int, str], side: str, query: str
) -> None:
    """One ranking."""
    st.subheader(title)
    judged = [labels.get(h.product.product_id) for h in result.hits]
    if labels and result.hits:
        exact = sum(j == "Exact" for j in judged)
        st.caption(f"{exact} of {len(judged)} labelled Exact · {result.timing['total']:.0f} ms")
    else:
        st.caption(f"{result.timing['total']:.0f} ms")
    if not result.hits:
        st.info("No results.")
    for rank, (hit, label) in enumerate(zip(result.hits, judged, strict=True), 1):
        show_hit(engine, hit, rank, label, side, query)


def rewrite_panel(query: str, result: Result) -> list[str] | None:
    """Rewrites, only when the trigger fired. A pasted key lives in this session's memory only."""
    u = result.understanding
    if not u.rewrite_triggered:
        return None
    with st.expander("Query rewriting (triggered: this query looks hard)", expanded=True):
        if u.rewrites:
            st.write("Cached rewrites: " + ", ".join(plain(r) for r in u.rewrites))
            st.caption(
                "Not applied by default: on held-out queries rewriting did not improve ranking. "
                "Tick the box to see what it would do."
            )
            if st.checkbox("Apply these rewrites", key=f"apply-{query}"):
                return u.rewrites
            return None
        st.write("No cached rewrite for this query.")
        key = st.text_input(
            "Gemini API key (optional; kept in this session's memory, never stored or logged)",
            type="password",
        )
        if key and st.button("Rewrite with Gemini"):
            found, problem = live_rewrites(query, key)
            if problem:
                st.warning(problem)
            else:
                st.session_state["live"] = (query, found)
        saved = st.session_state.get("live")
        if saved and saved[0] == query:
            st.write("Rewrites: " + ", ".join(plain(r) for r in saved[1]))
            return list(saved[1])
    return None


def main() -> None:
    """Draw the page."""
    engine = load_engine()
    default = engine.meta["default_ranker"]
    st.title("ShelfRank")
    st.caption(
        "Search over 42,994 products of Wayfair's public WANDS catalogue. Labels were made by "
        "people for 480 WANDS queries. Independent project, not affiliated with Wayfair."
    )
    with st.sidebar:
        st.header("Settings")
        options = [default, *(r for r in (LTR, RRF) if r != default)]
        ranker = st.selectbox(
            "Right-hand ranker",
            options,
            format_func=lambda r: NAMES[r],
            help="BM25 is on the left.",
        )
        if ranker != default:
            st.caption("Not the default. See the results for how the two compare.")
        chosen_class = st.selectbox("Class filter", ["any", *engine.catalog.classes()])
        rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5)
        st.markdown("[Results and limitations](https://github.com/iamrohitkumaryadav9/shelfrank)")
    typed = st.text_input("Search", placeholder="for example: grey velvet sofa", key="query")
    if not typed.strip():
        st.write("Try: " + " · ".join(f"`{e}`" for e in EXAMPLES))
        return
    query = " ".join(typed.split())[:200]
    suggestions = engine.suggest(query)
    if suggestions:
        st.caption(
            "Suggestions (from unique ESCI queries, no popularity): " + " · ".join(suggestions)
        )
    product_class = None if chosen_class == "any" else chosen_class
    min_rating = rating or None
    right = engine.search(query, 10, ranker, product_class, min_rating, explain=ranker == LTR)
    rewrites = rewrite_panel(query, right)
    if rewrites:
        right = engine.search(
            query, 10, ranker, product_class, min_rating, rewrites, explain=ranker == LTR
        )
    left = engine.search(query, 10, BM25, product_class, min_rating)

    u = right.understanding
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            st.markdown("**Predicted class**")
            for name, p in u.classes:
                st.write(f"{plain(name)}: {p:.0%}")
        with b:
            st.markdown("**Attributes in the query**")
            found = {s: v for s, v in u.slots.items() if v}
            if found:
                for slot, values in found.items():
                    st.write(f"{plain(slot)}: {plain(', '.join(values))}")
            else:
                st.write("none")
            st.caption("Slot model trained on LLM labels; human verification pending.")
    labels = engine.catalog.labels(query)
    if labels:
        st.caption(
            f"This is a WANDS query with {len(labels)} human labels. "
            "LTR uses models that did not train on it."
        )
    one, two = st.columns(2)
    with one:
        show_column(engine, NAMES[BM25], left, labels, "left", query)
    with two:
        show_column(engine, f"ShelfRank: {NAMES[ranker]}", right, labels, "right", query)
    with st.expander("SQL behind this page (DuckDB)"):
        st.code(sql.FILTER_SQL, language="sql")
        st.code(sql.SEARCH_SQL, language="sql")
        st.code(sql.JUDGMENT_SQL, language="sql")
        st.code(sql.SIMILAR_SQL, language="sql")
    with st.expander("Timing per stage (ms)"):
        st.json({k: round(v, 1) for k, v in right.timing.items()})


main()
