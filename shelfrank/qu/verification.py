"""Blind slot verification: label first, see the teacher afterwards, revise if needed."""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.slot_vocab import room_slots, room_spans
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
            answer = ask(f"  {slot:15s}> ").strip()
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
    except (Quit, EOFError, KeyboardInterrupt):
        # Every finished step is already on disk, so stopping at any point loses only this query.
        say("\nSaved. Run the same command to continue.")
    return sum(bool(s & finished) for s in log.stages().values())


def _spans(slots: Slots, slot: str) -> set[str]:
    return {" ".join(s.casefold().split()) for s in slots.get(slot, [])}


def same_slots(a: Slots, b: Slots) -> bool:
    """Whether two labels agree on every slot, ignoring case, spacing and order."""
    return all(_spans(a, slot) == _spans(b, slot) for slot in SLOTS)


def summary(log: VerificationLog, teacher: Mapping[int, Slots]) -> dict[str, Any]:
    """Blind agreement with the teacher overall and per slot, revisions and unsure counts.

    Agreement uses the blind label. A label changed after seeing the answer is not independent.
    Per slot, agreement alone would look high because most slots are empty, so the F1 of the
    teacher's spans against the blind ones is given with its support.
    """
    events = log.events()
    blind = {e.query_id: e.slots or {} for e in events if e.stage == "blind"}
    revised = {e.query_id: e.slots or {} for e in events if e.stage == "revised"}
    scored = sorted(q for q in blind if q in teacher)
    per_slot = {}
    for slot in SLOTS:
        pairs = [(_spans(blind[q], slot), _spans(teacher[q], slot)) for q in scored]
        hits = sum(len(mine & theirs) for mine, theirs in pairs)
        mine_n, theirs_n = sum(len(m) for m, _ in pairs), sum(len(t) for _, t in pairs)
        per_slot[slot] = {
            "agreement": sum(m == t for m, t in pairs) / len(pairs) if pairs else float("nan"),
            "support": mine_n,
            "teacher_spans": theirs_n,
            "f1": 2 * hits / (mine_n + theirs_n) if mine_n + theirs_n else float("nan"),
        }
    agree = sum(same_slots(blind[q], teacher[q]) for q in scored)
    return {
        "blind": len(blind),
        "unsure": sum(e.stage == "unsure" for e in events),
        "agree_all_slots": agree,
        "blind_agreement": agree / len(scored) if scored else float("nan"),
        "revised": len(revised),
        "revised_to_teacher": sum(
            same_slots(s, teacher[q]) for q, s in revised.items() if q in teacher
        ),
        "per_slot": per_slot,
    }


def room_check(
    log: VerificationLog, queries: Mapping[int, str], phrases: set[tuple[str, ...]]
) -> dict[str, Any]:
    """Compare blind room labels with the room rule, on queries that contain a room word.

    This is the independent test of the rule, whose thresholds were chosen while looking at seven
    examples.
    """
    disagree = []
    cases = 0
    for event in log.events():
        if event.stage != "blind" or not room_spans(queries[event.query_id]):
            continue
        cases += 1
        mine = sorted(" ".join(tokenize(s)) for s in (event.slots or {}).get("room", []))
        if mine != sorted(room_slots(queries[event.query_id], phrases)):
            disagree.append(event.query_id)
    return {"cases": cases, "disagreements": len(disagree), "query_ids": sorted(disagree)}
