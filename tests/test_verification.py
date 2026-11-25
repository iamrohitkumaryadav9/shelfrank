import json

import pytest

from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.verification import (
    VerificationLog,
    order,
    read_spans,
    run_session,
    summary,
)

QUERIES = {7: "dark gray linen sofa"}
TEACHER = {7: {"color": ["gray"], "material": ["linen"], "style": [], "room": []}}


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


def session(tmp_path, person, queries=QUERIES, teacher=TEACHER, version="v1"):
    log = VerificationLog(tmp_path / "log.jsonl")
    done = run_session(queries, teacher.__getitem__, log, version, 0, person.ask, person.say)
    return log, done


def stages(log):
    return [e.stage for e in log.events()]


def test_blind_label_is_stored_before_the_teacher_is_looked_up(tmp_path):
    log = VerificationLog(tmp_path / "log.jsonl")

    def teacher(query_id):
        assert stages(log) == ["blind"]
        return TEACHER[query_id]

    person = Person("dark gray", "linen", "", "", "", "")
    run_session(QUERIES, teacher, log, "v1", 0, person.ask, person.say)
    assert stages(log) == ["blind", "kept"]


def test_teacher_answer_is_not_shown_before_the_last_blind_answer(tmp_path):
    person = Person("dark gray", "linen", "", "", "", "")
    session(tmp_path, person)
    shown = [i for i, (_, t) in enumerate(person.seen) if "Gemini" in t]
    asked = [i for i, (k, t) in enumerate(person.seen) if k == "ask" and "save?" in t]
    assert shown and min(shown) > max(asked)


def test_revision_is_kept_beside_the_blind_label(tmp_path):
    person = Person("dark gray", "", "", "", "", "r", "gray", "linen", "", "")
    log, done = session(tmp_path, person)
    events = log.events()
    assert done == 1 and stages(log) == ["blind", "revised"]
    assert events[0].slots["color"] == ["dark gray"] and events[0].slots["material"] == []
    assert events[1].slots == TEACHER[7]
    assert all(e.time and e.version == "v1" for e in events)


def test_unsure_is_recorded_without_revealing(tmp_path):
    person = Person("?")
    log, done = session(tmp_path, person)
    assert done == 1 and stages(log) == ["unsure"] and "Gemini" not in person.text()


def test_words_outside_the_query_are_asked_again(tmp_path):
    person = Person("purple", "dark gray", "linen", "", "", "", "")
    log, _ = session(tmp_path, person)
    assert "not in the query: purple" in person.text()
    assert log.events()[0].slots["color"] == ["dark gray"]


def test_a_word_cannot_be_in_two_slots(tmp_path):
    person = Person("linen", "linen", "", "", "", "", "")
    log, _ = session(tmp_path, person)
    assert "already in another slot" in person.text()
    assert log.events()[0].slots["material"] == []


def test_redo_during_and_after_labelling_stores_only_the_final_label(tmp_path):
    person = Person("gray", "!", "gray", "", "", "", "!", "dark gray", "", "", "", "", "")
    log, _ = session(tmp_path, person)
    assert stages(log) == ["blind", "kept"]
    assert log.events()[0].slots["color"] == ["dark gray"]


def test_quit_before_saving_stores_nothing(tmp_path):
    log, done = session(tmp_path, Person("dark gray", "q"))
    assert done == 0 and log.events() == []


def test_quit_at_the_reveal_resumes_there_without_a_second_blind_label(tmp_path):
    log, done = session(tmp_path, Person("dark gray", "linen", "", "", "", "q"))
    assert done == 0 and stages(log) == ["blind"]
    person = Person("")
    log, done = session(tmp_path, person)
    assert done == 1 and stages(log) == ["blind", "kept"]
    assert person.text().count("color    >") == 0


def test_finished_queries_are_not_asked_again(tmp_path):
    session(tmp_path, Person("dark gray", "linen", "", "", "", ""))
    log, done = session(tmp_path, Person())
    assert done == 1 and stages(log) == ["blind", "kept"]


def test_labels_under_different_rules_are_not_mixed(tmp_path):
    session(tmp_path, Person("?"))
    with pytest.raises(ValueError, match="other rules"):
        session(tmp_path, Person(), version="v2")


def test_log_holds_ids_and_no_query_text(tmp_path):
    log, _ = session(tmp_path, Person("", "", "", "", "", ""))
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


def event(query_id, stage, color=None):
    slots = None if stage == "unsure" else {s: [] for s in SLOTS} | {"color": color or []}
    return {"query_id": query_id, "stage": stage, "slots": slots, "time": "t", "version": "v1"}


def test_summary_scores_the_blind_label_and_counts_the_rest(tmp_path):
    rows = [
        event(1, "blind", ["red"]),
        event(1, "kept", ["red"]),
        event(2, "blind", ["blue"]),
        event(2, "revised", ["Navy"]),
        event(3, "unsure"),
    ]
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    empty = {s: [] for s in SLOTS}
    teacher = {1: {**empty, "color": ["Red"]}, 2: {**empty, "color": ["navy"]}, 3: empty}
    out = summary(VerificationLog(path), teacher)
    assert out == {
        "blind": 2,
        "unsure": 1,
        "agree_all_slots": 1,
        "blind_agreement": 0.5,
        "revised": 1,
        "revised_to_teacher": 1,
    }
