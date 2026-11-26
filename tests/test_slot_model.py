import json

import numpy as np
import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.qu import slot_model as sm
from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.slot_lexicon import LexiconMatcher

LEXICON = {
    "color": {"red": 50, "dark gray": 40},
    "material_finish": {"oak": 60},
    "style": {"modern": 30},
    "product_types": ["coffee table"],
}
TRAIN = [
    ("red rug", {"color": ["red"]}),
    ("blue rug", {"color": ["blue"]}),
    ("oak desk", {"material_finish": ["oak"]}),
    ("pine desk", {"material_finish": ["pine"]}),
    ("modern lamp", {"style": ["modern"]}),
    ("lamp for bedroom", {"room": ["bedroom"]}),
    ("desk lamp", {}),
    ("rug", {}),
]


@pytest.mark.parametrize(
    ("query", "slots", "expected"),
    [
        ("dark gray sofa", {"color": ["dark gray"]}, ["color", "color", "none"]),
        ("Red, rug", {"color": ["red"]}, ["color", "none"]),
        ("gold mirror gold frame", {"color": ["gold"]}, ["color", "none", "none", "none"]),
        ("red rug", {"color": ["purple"]}, ["none", "none"]),
        ("red rug", {"color": ["red"], "style": ["red"]}, ["color", "none"]),
        ("red rug", {}, ["none", "none"]),
    ],
)
def test_word_labels(query, slots, expected):
    assert sm.word_labels(query, slots) == expected


def test_labels_and_spans_round_trip():
    slots = sm.empty() | {"color": ["dark gray"], "material_finish": ["linen"]}
    labels = sm.word_labels("dark gray linen sofa", slots)
    assert sm.spans_from_labels("dark gray linen sofa", labels) == slots


def test_spans_at_the_end_and_split_by_other_words_are_kept():
    out = sm.spans_from_labels("red sofa blue", ["color", "none", "color"])
    assert out["color"] == ["red", "blue"]


@pytest.mark.parametrize("lexicon", [True, False])
def test_features_have_one_row_per_word_and_respect_the_switch(lexicon):
    rows = sm.word_features(
        "modern red coffee table", sm.TaggerConfig(lexicon, 1, 1.0), LexiconMatcher(LEXICON)
    )
    assert len(rows) == 4
    assert any(k.startswith("lexicon") for k in rows[0]) is lexicon
    assert "-2=modern" not in rows[2] and "-1=red" in rows[2]


@pytest.fixture(scope="module")
def tagger():
    model = sm.WordTagger(sm.TaggerConfig(True, 1, 10.0), LEXICON)
    return model.fit([q for q, _ in TRAIN], [s for _, s in TRAIN])


def test_tagger_learns_and_generalises_by_lexicon(tagger):
    assert tagger.slots("red desk")["color"] == ["red"]
    assert tagger.slots("oak lamp")["material_finish"] == ["oak"]
    assert tagger.slots("") == sm.empty()
    assert set(tagger.slots("anything at all")) == set(SLOTS)


def test_plain_form_predicts_the_same_without_scikit_learn_objects(tagger):
    body = tagger.to_plain()
    assert not any("sklearn" in type(v).__module__ for v in body.values())
    again = sm.WordTagger.from_plain(body)
    for query, _ in TRAIN:
        assert again.labels(query) == tagger.labels(query)


def test_own_scoring_equals_scikit_learn():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    config, matcher = sm.TaggerConfig(True, 2, 1.0), LexiconMatcher(LEXICON)
    rows = [r for q, _ in TRAIN for r in sm.word_features(q, config, matcher)]
    targets = [x for q, s in TRAIN for x in sm.word_labels(q, s)]
    vectorizer = DictVectorizer()
    reference = LogisticRegression(C=1.0, max_iter=2000).fit(
        vectorizer.fit_transform(rows), targets
    )
    ours = sm.WordTagger(config, LEXICON).fit([q for q, _ in TRAIN], [s for _, s in TRAIN])
    query = "modern oak desk for bedroom"
    features = vectorizer.transform(sm.word_features(query, config, matcher))
    assert ours.labels(query) == list(reference.predict(features))


def test_training_is_deterministic():
    data = [q for q, _ in TRAIN], [s for _, s in TRAIN]
    a = sm.WordTagger(sm.TaggerConfig(True, 1, 1.0), LEXICON).fit(*data)
    b = sm.WordTagger(sm.TaggerConfig(True, 1, 1.0), LEXICON).fit(*data)
    assert np.array_equal(a.coef, b.coef)


def test_labelled_slots_reads_ids(tmp_path):
    (tmp_path / "data/labels").mkdir(parents=True)
    row = {"id": 4, "label": sm.empty(), "rules": sm.SLOT_VERSION}
    path = tmp_path / f"data/labels/slots_x_{sm.SLOT_VERSION}.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert sm.labelled_slots("x", tmp_path) == {4: sm.empty()}


def test_lexicon_is_loaded_when_selected(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "configs/slot_student.yaml").write_text("kind: lexicon\n", encoding="utf-8")
    (tmp_path / "data/slot_lexicon.json").write_text(json.dumps(LEXICON), encoding="utf-8")
    assert sm.load_slot_model(tmp_path).slots("red rug")["color"] == ["red"]


def test_protocol_states_the_selection_rule_and_the_dropped_round():
    text = " ".join((REPO_ROOT / "docs/PROTOCOL.md").read_text(encoding="utf-8").split())
    assert "Round A was then dropped before the freeze, to save rater time." in text
    assert "misapplied the slot rules: product words were labelled as style" in text
    assert "could not have been blind anyway, because its tool showed the teacher's answers" in text
    assert "Round B is a blind check" in text
    assert "the mean of two exact-span F1 scores over all slots" in text
    assert "vs LLM silver labels; human verification pending" in text


def test_committed_slot_page_is_generated_and_every_score_is_marked_silver():
    from shelfrank.qu.slot_report import render

    data = json.loads((REPO_ROOT / "results/slots_dev.json").read_text(encoding="utf-8"))
    page = (REPO_ROOT / "docs/slots_dev.md").read_text(encoding="utf-8")
    assert page == render(data) and "vs LLM silver labels; human verification pending" in page


def test_committed_slot_selection_follows_the_rule():
    data = json.loads((REPO_ROOT / "results/slots_dev.json").read_text(encoding="utf-8"))
    scores = [r["selection"] for r in data["runs"]]
    assert data["selected"] == scores.index(max(scores)) and data["runs"][0]["kind"] == "lexicon"
    for run in data["runs"]:
        mean = (run["silver"]["all"]["f1"] + run["esci"]["all"]["f1"]) / 2
        assert run["selection"] == pytest.approx(mean)
