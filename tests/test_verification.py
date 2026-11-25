import importlib.util
import json
import math
import sys

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.qu.annotator import PRACTICE, SLOT_GUIDE, Attempt, practice_id, unlocked
from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.verification import (
    Session,
    VerificationLog,
    guide_lines,
    order,
    problems,
    read_spans,
    room_check,
    run_practice,
    run_session,
    summary,
    unknown_styles,
)

QUERIES = {7: "dark gray linen sofa"}
STYLES = {("modern",), ("mid", "century")}


class Person:
    """Scripted annotator that records the order of everything it sees."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []

    def ask(self, prompt):
        self.seen.append(("ask", prompt))
        return self.answers.pop(0)

    def say(self, text):
        self.seen.append(("say", text))

    def text(self):
        return "\n".join(t for _, t in self.seen)


def session(tmp_path, person, queries=QUERIES, version="v1", styles=()):
    log = VerificationLog(tmp_path / "log.jsonl")
    names = [" ".join(s) for s in styles]
    run = Session(log, version, person.ask, person.say, names, styles)
    return log, run_session(run, queries, 0)


def stages(log):
    return [e.stage for e in log.events()]


def test_labelling_shows_no_answer_and_finishes(tmp_path):
    person = Person("dark gray", "linen", "", "", "")
    log, done = session(tmp_path, person)
    assert done == 1 and stages(log) == ["label"]
    assert "Gemini" not in person.text() and "revise" not in person.text()


def test_unsure_is_recorded(tmp_path):
    log, done = session(tmp_path, Person("?"))
    assert done == 1 and stages(log) == ["unsure"]


def test_labelling_code_has_no_way_to_receive_the_teachers_answers():
    """No reveal in either round: the session holds nothing the answers could arrive through."""
    assert not [f for f in Session.__dataclass_fields__ if "teacher" in f]
    source = (REPO_ROOT / "shelfrank/qu/verification.py").read_text(encoding="utf-8")
    labelling = source.split("def summary(")[0]
    for word in ("reveal", "Gemini", "revise", "load_teacher"):
        assert word not in labelling
    script = (REPO_ROOT / "scripts/verify_slots.py").read_text(encoding="utf-8")
    assert script.count("load_teacher(") == 2 and "REVEALS" not in script


def test_words_outside_the_query_are_blocked_until_corrected(tmp_path):
    person = Person("purple", "y", "dark gray", "linen", "", "", "")
    log, _ = session(tmp_path, person)
    assert "BLOCKED: not in the query, word for word: purple" in person.text()
    assert log.events()[0].slots["color"] == ["dark gray"]


@pytest.mark.parametrize(
    ("span", "allowed"),
    [
        ("dark gray", True),
        ("Dark Gray", True),
        ("gray dark", False),
        ("dark linen", False),
        ("gra", False),
        ("grey", False),
        ("dark-gray", False),
    ],
)
def test_a_span_must_appear_verbatim(span, allowed):
    assert (problems("dark gray linen sofa", {}, [span]) == []) is allowed


def test_a_word_cannot_be_in_two_slots(tmp_path):
    person = Person("linen", "linen", "", "", "", "")
    log, _ = session(tmp_path, person)
    assert "already in color: linen" in person.text()
    assert log.events()[0].slots["material_finish"] == []


@pytest.mark.parametrize("second", ["gray", "GRAY", "gray linen", "dark gray linen"])
def test_sharing_one_word_of_a_phrase_is_also_blocked(second):
    used = {"color": ["dark gray"]}
    assert any("already in color" in p for p in problems("dark gray linen sofa", used, [second]))


def test_same_word_twice_in_the_query_still_belongs_to_one_slot():
    assert problems("gold mirror gold frame", {"color": ["gold"]}, ["gold frame"])


def test_neighbouring_words_in_different_slots_are_allowed():
    assert problems("dark gray linen sofa", {"color": ["dark gray"]}, ["linen"]) == []


def test_every_prompt_shows_its_rule_and_two_examples_each_way(tmp_path):
    person = Person("", "", "", "", "")
    session(tmp_path, person)
    asked = [i for i, (kind, _) in enumerate(person.seen) if kind == "ask"]
    for slot, at in zip(SLOTS, asked, strict=False):
        shown = person.seen[at - 2][1]
        assert SLOT_GUIDE[slot].rule in shown
        assert shown.count("yes:") == 2 and shown.count("no :") == 2


def test_style_prompt_lists_every_catalogue_style_and_the_others_do_not():
    raw = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
    names = list(raw["styles"])
    shown = " ".join(guide_lines("style", names).split())
    assert len(names) == 57 and "catalogue styles (57)" in shown
    assert all(name in shown for name in names)
    assert all("catalogue styles" not in guide_lines(s, names) for s in SLOTS if s != "style")


def test_unknown_style_warns_and_can_be_retyped(tmp_path):
    person = Person("", "", "cosy", "", "", "", "")
    log, _ = session(tmp_path, person, {1: "cosy modern rug"}, styles=STYLES)
    assert "WARNING: not a catalogue style: cosy" in person.text()
    assert log.events()[0].slots["style"] == []


def test_unknown_style_can_be_kept_because_the_warning_is_soft(tmp_path):
    person = Person("", "", "cosy", "y", "", "")
    log, _ = session(tmp_path, person, {1: "cosy modern rug"}, styles=STYLES)
    assert log.events()[0].slots["style"] == ["cosy"]


def test_known_style_and_spelling_do_not_warn():
    assert unknown_styles(["Modern", "mid-century"], STYLES) == []
    assert unknown_styles(["modern rug"], STYLES) == ["modern rug"]


def test_redo_during_and_after_labelling_stores_only_the_final_label(tmp_path):
    person = Person("gray", "!", "gray", "", "", "", "!", "dark gray", "", "", "", "")
    log, _ = session(tmp_path, person)
    assert stages(log) == ["label"]
    assert log.events()[0].slots["color"] == ["dark gray"]


@pytest.mark.parametrize("answers", [("dark gray", "q"), ("dark gray", "linen", "", "", "q")])
def test_quit_before_saving_stores_nothing(tmp_path, answers):
    log, done = session(tmp_path, Person(*answers))
    assert done == 0 and log.events() == []


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_ctrl_c_or_closed_input_keeps_what_was_saved(tmp_path, error):
    person = Person("dark gray", "linen", "", "", "")
    queries = {7: "dark gray linen sofa", 8: "dark gray linen chair"}
    answer = person.ask

    def ask(prompt):
        if not person.answers:
            raise error
        return answer(prompt)

    log = VerificationLog(tmp_path / "log.jsonl")
    done = run_session(Session(log, "v1", ask, person.say), queries, 0)
    assert done == 1 and stages(log) == ["label"] and "Saved" in person.text()


def test_finished_queries_are_not_asked_again(tmp_path):
    session(tmp_path, Person("dark gray", "linen", "", "", ""))
    log, done = session(tmp_path, Person())
    assert done == 1 and stages(log) == ["label"]


def test_labels_under_different_rules_are_not_mixed(tmp_path):
    session(tmp_path, Person("?"))
    with pytest.raises(ValueError, match="other rules"):
        session(tmp_path, Person(), version="v2")


def test_log_holds_ids_and_no_query_text(tmp_path):
    log, _ = session(tmp_path, Person("", "", "", "", ""))
    assert "sofa" not in log.path.read_text(encoding="utf-8")


def test_order_is_shuffled_and_stable():
    ids = list(range(50))
    assert order(ids, 1) == order(ids[::-1], 1) != ids


@pytest.mark.parametrize(
    ("answer", "spans"),
    [("", []), ("red", ["red"]), (" red ,, blue,red ", ["red", "blue"])],
)
def test_read_spans(answer, spans):
    assert read_spans(answer) == spans


def answers_for(item):
    return [", ".join(item.answer[s]) for s in SLOTS]


def practise(wrong=0):
    answers = []
    for i, item in enumerate(PRACTICE):
        answers += ["?"] if i < wrong else answers_for(item)
    person = Person(*answers)
    return run_practice(PRACTICE, "v1", person.ask, person.say), person


def test_perfect_practice_scores_all_and_unlocks():
    attempt, _ = practise()
    assert (attempt.correct, attempt.total) == (15, 15) and attempt.unlocks("v1")


@pytest.mark.parametrize(("wrong", "unlocks"), [(2, True), (3, False)])
def test_thirteen_of_fifteen_is_the_pass_mark(wrong, unlocks):
    attempt, _ = practise(wrong)
    assert attempt.unlocks("v1") is unlocks


def test_feedback_shows_the_answer_and_the_rule_after_each_query_only():
    person = Person("q")
    run_practice(PRACTICE, "v1", person.ask, person.say)
    assert PRACTICE[0].why not in person.text()
    _, person = practise(1)
    text = person.text()
    assert "WRONG" in text and all(item.why in text for item in PRACTICE)
    assert text.index(PRACTICE[0].why) < text.index(PRACTICE[1].query)


def test_stopped_practice_is_not_an_attempt():
    person = Person("q")
    assert run_practice(PRACTICE, "v1", person.ask, person.say) is None


def write_attempt(path, **changes):
    base = {"correct": 15, "total": 15, "version": "v1", "practice": practice_id(), "time": "t"}
    path.write_text(json.dumps(base | changes) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"correct": 12}, False),
        ({"version": "v0"}, False),
        ({"practice": "other"}, False),
        ({"total": 13, "correct": 13}, False),
    ],
)
def test_unlock_needs_a_pass_under_the_current_rules_and_questions(tmp_path, changes, expected):
    assert unlocked(write_attempt(tmp_path / "p.jsonl", **changes), "v1") is expected


def test_no_practice_means_locked(tmp_path):
    assert unlocked(tmp_path / "none.jsonl", "v1") is False
    assert not Attempt(0, 15, "v1", practice_id(), "t").unlocks("v1")


@pytest.fixture
def script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "verify_slots_script", REPO_ROOT / "scripts/verify_slots.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PRACTICE_LOG", tmp_path / "p.jsonl")
    monkeypatch.setattr(module, "FOLDER", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_slots.py", "--round", "A"])
    return module


def test_round_a_is_locked_until_practice_is_passed(script, capsys):
    assert script.main() == 1 and "locked" in capsys.readouterr().out
    write_attempt(script.PRACTICE_LOG, correct=12, version=script.SLOT_VERSION)
    assert script.main() == 1 and "locked" in capsys.readouterr().out


def test_round_a_never_opens_the_teacher_file_while_labelling(script, monkeypatch):
    import pandas as pd

    write_attempt(script.PRACTICE_LOG, version=script.SLOT_VERSION)
    used = {}
    monkeypatch.setattr(script, "load_teacher", lambda path: pytest.fail("teacher file opened"))
    monkeypatch.setattr(script, "queries_for", lambda *a: (7,))
    frame = pd.DataFrame({"query_id": [7], "query": ["red rug"]})
    monkeypatch.setattr(script, "load_queries", lambda path: frame)
    monkeypatch.setattr(script, "run_session", lambda s, q, seed: used.update(s=s, q=q) or 0)
    assert script.main() == 0
    assert len(used["s"].style_names) == 57


def test_summary_is_refused_before_every_query_is_finished(script, monkeypatch, capsys):
    import pandas as pd

    write_attempt(script.PRACTICE_LOG, version=script.SLOT_VERSION)
    monkeypatch.setattr(sys, "argv", ["verify_slots.py", "--round", "A", "--summary"])
    monkeypatch.setattr(script, "load_teacher", lambda path: pytest.fail("teacher file opened"))
    monkeypatch.setattr(script, "queries_for", lambda *a: (7,))
    frame = pd.DataFrame({"query_id": [7], "query": ["red rug"]})
    monkeypatch.setattr(script, "load_queries", lambda path: frame)
    assert script.main() == 1 and "0 of 1 finished" in capsys.readouterr().out


def event(query_id, stage, color=None):
    slots = None if stage == "unsure" else {s: [] for s in SLOTS} | {"color": color or []}
    return {"query_id": query_id, "stage": stage, "slots": slots, "time": "t", "version": "v1"}


def write_log(tmp_path, rows):
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return VerificationLog(path)


def test_summary_scores_the_first_label_and_counts_the_rest(tmp_path):
    rows = [
        event(1, "label", ["red"]),
        event(2, "label", ["blue"]),
        event(3, "unsure"),
    ]
    empty = {s: [] for s in SLOTS}
    teacher = {1: {**empty, "color": ["Red"]}, 2: {**empty, "color": ["navy"]}, 3: empty}
    out = summary(write_log(tmp_path, rows), teacher)
    per_slot = out.pop("per_slot")
    assert out == {
        "n": 2,
        "excluded": [],
        "unsure": 1,
        "agree_all_slots": 1,
        "agreement": 0.5,
    }
    assert per_slot["color"] == {"agreement": 0.5, "support": 2, "teacher_spans": 2, "f1": 0.5}
    assert per_slot["room"]["agreement"] == 1.0 and math.isnan(per_slot["room"]["f1"])


def test_excluded_queries_change_n_and_agreement_everywhere(tmp_path):
    log = write_log(tmp_path, [event(1, "label", ["red"]), event(2, "label", ["blue"])])
    empty = {s: [] for s in SLOTS}
    teacher = {1: {**empty, "color": ["red"]}, 2: empty}
    out = summary(log, teacher, {2})
    assert (out["n"], out["excluded"], out["agreement"]) == (1, [2], 1.0)
    assert out["per_slot"]["color"]["support"] == 1
    assert room_check(log, {1: "bathroom rug", 2: "bathroom mat"}, set(), {2})["cases"] == 1


def test_room_check_compares_labels_with_the_rule_on_room_queries_only(tmp_path):
    queries = {1: "dining table", 2: "bathroom rug", 3: "red chair", 4: "bedroom lamp"}
    rows = [event(q, "label") for q in (1, 2, 3, 4)]
    rows[3]["slots"]["room"] = ["Bedroom"]
    out = room_check(write_log(tmp_path, rows), queries, {("dining", "table")})
    assert out == {"cases": 3, "disagreements": 1, "query_ids": [2]}
