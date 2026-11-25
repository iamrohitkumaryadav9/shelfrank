"""Blind slot verification: label first, see the teacher afterwards, revise if needed."""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.teacher import contains_span

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
    """Append-only file of decisions, so that a blind label can never be overwritten."""

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

    def stages(self) -> dict[int, set[str]]:
        """Stages recorded per query."""
        out: dict[int, set[str]] = {}
        for event in self.events():
            out.setdefault(event.query_id, set()).add(event.stage)
        return out


def order(query_ids: Sequence[int], seed: int) -> list[int]:
    """Shuffle once and deterministically, so that a resumed session continues the same order."""
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    return ids


def read_spans(answer: str) -> list[str]:
    """Split an answer into spans."""
    return list(dict.fromkeys(part.strip() for part in answer.split(",") if part.strip()))


def ask_slots(query: str, ask: Ask, say: Say, allow_unsure: bool = True) -> Slots:
    """Ask for each slot and accept only words of the query, each in one slot."""
    slots: Slots = {}
    for slot in SLOTS:
        while True:
            answer = ask(f"  {slot:9s}> ").strip()
            if answer == QUIT:
                raise Quit
            if answer == REDO:
                raise Redo
            if answer == UNSURE and allow_unsure:
                raise Unsure
            spans = read_spans(answer)
            absent = [s for s in spans if not contains_span(query, s)]
            used = [s for s in spans if any(s in other for other in slots.values())]
            if not absent and not used:
                slots[slot] = spans
                break
            if absent:
                say(f"    not in the query: {', '.join(absent)}")
            if used:
                say(f"    already in another slot: {', '.join(used)}")
    return slots


def show(slots: Slots) -> str:
    """Render slots on one line."""
    filled = [f"{slot}={', '.join(slots[slot])}" for slot in SLOTS if slots.get(slot)]
    return "; ".join(filled) or "(nothing)"


def verify_query(
    query_id: int,
    query: str,
    teacher: Callable[[int], Slots],
    log: VerificationLog,
    done: set[str],
    version: str,
    ask: Ask,
    say: Say,
) -> None:
    """Take a blind label, store it, and only then look the teacher's answer up."""

    def record(stage: str, slots: Slots | None) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        log.append(Event(query_id, stage, slots, now, version))

    blind: Slots | None = None
    if "blind" not in done:
        while blind is None:
            say(f'\n"{query}"')
            try:
                blind = ask_slots(query, ask, say)
            except Redo:
                continue
            except Unsure:
                record("unsure", None)
                return
            say(f"  yours : {show(blind)}")
            if ask("  save? Enter = yes, ! = redo > ").strip() == REDO:
                blind = None
        record("blind", blind)
    else:
        blind = next(e.slots for e in log.events() if e.query_id == query_id and e.stage == "blind")
        say(f'\n"{query}"\n  yours : {show(blind or {})}')

    say(f"  Gemini: {show(teacher(query_id))}")
    choice = ask("  keep yours? Enter = keep, r = revise > ").strip().lower()
    if choice == QUIT:
        raise Quit
    if choice == "r":
        while True:
            try:
                record("revised", ask_slots(query, ask, say, allow_unsure=False))
                return
            except Redo:
                continue
    record("kept", blind)


def run_session(
    queries: Mapping[int, str],
    teacher: Callable[[int], Slots],
    log: VerificationLog,
    version: str,
    seed: int,
    ask: Ask,
    say: Say,
) -> int:
    """Go through the unfinished queries; return how many are finished."""
    finished = {"kept", "revised", "unsure"}
    stages = log.stages()
    versions = {e.version for e in log.events()}
    if versions - {version}:
        raise ValueError(f"the log holds labels under other rules: {sorted(versions)}")
    todo = [q for q in order(list(queries), seed) if not stages.get(q, set()) & finished]
    say(f"{len(queries) - len(todo)} of {len(queries)} done. {HELP}")
    try:
        for query_id in todo:
            done = stages.get(query_id, set())
            verify_query(query_id, queries[query_id], teacher, log, done, version, ask, say)
    except Quit:
        say("\nSaved. Run the same command to continue.")
    return sum(bool(s & finished) for s in log.stages().values())


def summary(log: VerificationLog, teacher: Mapping[int, Slots]) -> dict[str, float | int]:
    """Blind agreement with the teacher, revisions and unsure counts.

    Agreement uses the blind label. A label changed after seeing the answer is not independent.
    """
    events = log.events()
    blind = {e.query_id: e.slots for e in events if e.stage == "blind"}
    unsure = {e.query_id for e in events if e.stage == "unsure"}
    revised = {e.query_id: e.slots for e in events if e.stage == "revised"}

    def same(a: Slots | None, b: Slots | None) -> bool:
        if a is None or b is None:
            return False
        return all(
            {s.casefold() for s in a.get(slot, [])} == {s.casefold() for s in b.get(slot, [])}
            for slot in SLOTS
        )

    scored = [q for q in blind if q in teacher]
    agree = sum(same(blind[q], teacher[q]) for q in scored)
    return {
        "blind": len(blind),
        "unsure": len(unsure),
        "agree_all_slots": agree,
        "blind_agreement": agree / len(scored) if scored else float("nan"),
        "revised": len(revised),
        "revised_to_teacher": sum(same(s, teacher.get(q)) for q, s in revised.items()),
    }
