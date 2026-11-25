"""Slot verification by hand: guided labelling, practice, and agreement with the teacher.

Labelling never sees the teacher: nothing here takes its answers except the summary.
"""

import json
import random
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shelfrank.qu.annotator import PASS_MARK, SLOT_GUIDE, Attempt, PracticeItem, practice_id
from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.slot_vocab import Term, room_slots, room_spans
from shelfrank.qu.teacher import contains_span
from shelfrank.text.normalize import tokenize

Slots = dict[str, list[str]]
Ask = Callable[[str], str]
Say = Callable[[str], None]

QUIT, UNSURE, REDO = "q", "?", "!"
HELP = (
    f"Type the words from the query, separated by commas. Enter = nothing. "
    f"{UNSURE} = unsure, skip this query. {REDO} = redo this query. {QUIT} = save and quit."
)


@dataclass(frozen=True)
class Event:
    """One recorded decision. Query text is not stored, only its id."""

    query_id: int
    stage: str
    slots: Slots | None
    time: str
    version: str


class Quit(Exception):
    """The annotator asked to stop."""


class Redo(Exception):
    """The annotator asked to start the query again."""


class Unsure(Exception):
    """The annotator declined to label the query."""


class VerificationLog:
    """Append-only file of decisions, so that a label can never be overwritten."""

    def __init__(self, path: Path) -> None:
        """Use ``path``, created on the first event."""
        self.path = path

    def events(self) -> list[Event]:
        """Read all decisions."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Event(**json.loads(line)) for line in lines if line.strip()]

    def append(self, event: Event) -> None:
        """Write one decision immediately."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def finished(self) -> set[int]:
        """Queries with a label or an "unsure"."""
        return {event.query_id for event in self.events()}


def order(query_ids: Sequence[int], seed: int) -> list[int]:
    """Shuffle once and deterministically, so that a resumed session continues the same order."""
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    return ids


def read_spans(answer: str) -> list[str]:
    """Split an answer into spans."""
    return list(dict.fromkeys(part.strip() for part in answer.split(",") if part.strip()))


def guide_lines(slot: str, styles: Collection[str] = ()) -> str:
    """What is shown above a slot prompt: the rule, two examples that fit and two that do not."""
    guide = SLOT_GUIDE[slot]
    lines = [f"  {slot.upper()}: {guide.rule}"]
    lines += [f"    yes: {text}  ->  {span}" for text, span in guide.yes]
    lines += [f"    no : {text}  ->  {reason}" for text, reason in guide.no]
    if slot == "style" and styles:
        names = ", ".join(sorted(styles))
        lines.append(
            textwrap.fill(
                f"catalogue styles ({len(styles)}): {names}",
                96,
                initial_indent="    ",
                subsequent_indent="      ",
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def span_words(span: str) -> set[str]:
    """Words of a span, ignoring case and surrounding punctuation."""
    return {w.strip(".,;:!?()\"'") for w in span.casefold().split()}


def problems(query: str, slots: Slots, spans: Sequence[str]) -> list[str]:
    """Reasons to refuse an answer. These are hard blocks, with no way past them."""
    out = [f"not in the query, word for word: {s}" for s in spans if not contains_span(query, s)]
    for slot, others in slots.items():
        used = set().union(*map(span_words, others), set())
        taken = sorted(used & set().union(*map(span_words, spans), set()))
        if taken:
            out.append(f"already in {slot}: {', '.join(taken)}. A word belongs to one slot.")
    return out


def unknown_styles(spans: Sequence[str], known: Collection[Term]) -> list[str]:
    """Style answers that are not a catalogue style or one of its spellings."""
    return [s for s in spans if tuple(tokenize(s)) not in known]


def ask_slots(
    query: str,
    ask: Ask,
    say: Say,
    style_names: Collection[str] = (),
    style_terms: Collection[Term] = (),
) -> Slots:
    """Ask for each slot with its guide, and accept only words of the query, each in one slot."""
    slots: Slots = {}
    for slot in SLOTS:
        say(guide_lines(slot, style_names))
        while slot not in slots:
            say(f'  "{query}"')
            answer = ask(f"  {slot:15s}> ").strip()
            if answer == QUIT:
                raise Quit
            if answer == REDO:
                raise Redo
            if answer == UNSURE:
                raise Unsure
            spans = read_spans(answer)
            found = problems(query, slots, spans)
            for problem in found:
                say(f"    BLOCKED: {problem}")
            if found:
                continue
            odd = unknown_styles(spans, style_terms) if slot == "style" and style_terms else []
            if odd:
                say(f"    WARNING: not a catalogue style: {', '.join(odd)}")
                if ask("    keep it? y = keep, Enter = type again > ").strip().lower() != "y":
                    continue
            slots[slot] = spans
    return slots


def show(slots: Slots) -> str:
    """Render slots on one line."""
    filled = [f"{slot}={', '.join(slots[slot])}" for slot in SLOTS if slots.get(slot)]
    return "; ".join(filled) or "(nothing)"


@dataclass(frozen=True)
class Session:
    """Everything one labelling run needs, so that the steps below share it."""

    log: VerificationLog
    version: str
    ask: Ask
    say: Say
    style_names: Collection[str] = ()
    style_terms: Collection[Term] = ()

    def record(self, query_id: int, stage: str, slots: Slots | None) -> None:
        """Store one decision at once."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.log.append(Event(query_id, stage, slots, now, self.version))

    def slots(self, query: str) -> Slots:
        """Ask for all slots of a query."""
        return ask_slots(query, self.ask, self.say, self.style_names, self.style_terms)


def label_query(session: Session, query_id: int, query: str) -> None:
    """Take a label, or "unsure", and store it."""
    while True:
        session.say(f'\n"{query}"')
        try:
            slots = session.slots(query)
        except Redo:
            continue
        except Unsure:
            session.record(query_id, "unsure", None)
            return
        session.say(f"  yours : {show(slots)}")
        choice = session.ask(f"  save? Enter = yes, {REDO} = redo, {QUIT} = quit > ").strip()
        if choice == QUIT:
            raise Quit
        if choice != REDO:
            session.record(query_id, "label", slots)
            return


def run_session(session: Session, queries: Mapping[int, str], seed: int) -> int:
    """Go through the unfinished queries; return how many are finished."""
    events = session.log.events()
    versions = {e.version for e in events}
    if versions - {session.version}:
        raise ValueError(f"the log holds labels under other rules: {sorted(versions)}")
    done = session.log.finished()
    todo = [q for q in order(list(queries), seed) if q not in done]
    session.say(f"{len(queries) - len(todo)} of {len(queries)} done. {HELP}")
    try:
        for query_id in todo:
            label_query(session, query_id, queries[query_id])
    except (Quit, EOFError, KeyboardInterrupt):
        # Every finished step is already on disk, so stopping at any point loses only this query.
        session.say("\nSaved. Run the same command to continue.")
    return len(session.log.finished())


def run_practice(
    items: Sequence[PracticeItem],
    version: str,
    ask: Ask,
    say: Say,
    style_names: Collection[str] = (),
    style_terms: Collection[Term] = (),
) -> Attempt | None:
    """Label every practice query with feedback. None when stopped early."""
    say(f"{len(items)} invented queries. {PASS_MARK} must be right on every slot. {HELP}")
    correct = 0
    for number, item in enumerate(items, 1):
        answer: Slots | None = None
        while answer is None:
            say(f'\n{number}/{len(items)}  "{item.query}"')
            try:
                answer = ask_slots(item.query, ask, say, style_names, style_terms)
            except Redo:
                continue
            except Unsure:
                break
            except (Quit, EOFError, KeyboardInterrupt):
                say("\nStopped. Practice starts again next time.")
                return None
        right = answer is not None and same_slots(answer, item.answer)
        correct += right
        say(f"  {'RIGHT' if right else 'WRONG'}")
        say(f"  yours : {show(answer or {})}")
        say(f"  answer: {show(item.answer)}")
        say(f"  why   : {item.why}")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    return Attempt(correct, len(items), version, practice_id(), now)


def _spans(slots: Slots, slot: str) -> set[str]:
    return {" ".join(s.casefold().split()) for s in slots.get(slot, [])}


def same_slots(a: Slots, b: Slots) -> bool:
    """Whether two labels agree on every slot, ignoring case, spacing and order."""
    return all(_spans(a, slot) == _spans(b, slot) for slot in SLOTS)


def scored_labels(log: VerificationLog, excluded: Collection[int] = ()) -> dict[int, Slots]:
    """First label per query, without the excluded queries."""
    labels: dict[int, Slots] = {}
    for event in log.events():
        if event.stage == "label" and event.query_id not in excluded:
            labels.setdefault(event.query_id, event.slots or {})
    return labels


def summary(
    log: VerificationLog, teacher: Mapping[int, Slots], excluded: Collection[int] = ()
) -> dict[str, Any]:
    """Agreement with the teacher overall and per slot, with the n it rests on.

    Per slot, agreement alone would look high because most slots are empty, so the F1 of the
    teacher's spans is given with its support.
    """
    events = log.events()
    labels = scored_labels(log, excluded)
    scored = sorted(q for q in labels if q in teacher)
    per_slot = {}
    for slot in SLOTS:
        pairs = [(_spans(labels[q], slot), _spans(teacher[q], slot)) for q in scored]
        hits = sum(len(mine & theirs) for mine, theirs in pairs)
        mine_n, theirs_n = sum(len(m) for m, _ in pairs), sum(len(t) for _, t in pairs)
        per_slot[slot] = {
            "agreement": sum(m == t for m, t in pairs) / len(pairs) if pairs else float("nan"),
            "support": mine_n,
            "teacher_spans": theirs_n,
            "f1": 2 * hits / (mine_n + theirs_n) if mine_n + theirs_n else float("nan"),
        }
    agree = sum(same_slots(labels[q], teacher[q]) for q in scored)
    return {
        "n": len(scored),
        "excluded": sorted(excluded),
        "unsure": sum(e.stage == "unsure" for e in events),
        "agree_all_slots": agree,
        "agreement": agree / len(scored) if scored else float("nan"),
        "per_slot": per_slot,
    }


def room_check(
    log: VerificationLog,
    queries: Mapping[int, str],
    phrases: set[tuple[str, ...]],
    excluded: Collection[int] = (),
) -> dict[str, Any]:
    """Compare room labels with the room rule, on queries that contain a room word.

    The thresholds of the rule were chosen while looking at seven examples, so it needs labels made
    without it. Only Round B gives them.
    """
    disagree = []
    cases = 0
    for query_id, slots in scored_labels(log, excluded).items():
        if not room_spans(queries[query_id]):
            continue
        cases += 1
        mine = sorted(" ".join(tokenize(s)) for s in slots.get("room", []))
        if mine != sorted(room_slots(queries[query_id], phrases)):
            disagree.append(query_id)
    return {"cases": cases, "disagreements": len(disagree), "query_ids": sorted(disagree)}
