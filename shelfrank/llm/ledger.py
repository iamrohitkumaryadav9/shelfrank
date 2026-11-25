"""Append-only record of every LLM call and what it cost."""

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


class LedgerError(RuntimeError):
    """The ledger cannot be trusted, so no call may be made."""


@dataclass(frozen=True)
class Entry:
    """One line. A call has a "reserved" line and, normally, a later line that replaces it."""

    call_id: str
    time: str
    purpose: str
    batch: str
    model: str
    status: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_inr: float
    note: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """Spend so far, recomputed from the file each time so that no state can drift from it."""

    def __init__(self, path: Path) -> None:
        """Use ``path``; it is created on the first reservation."""
        self.path = path

    def entries(self) -> list[Entry]:
        """Read every line. A line that does not parse stops everything.

        Skipping it could hide spend, which is the one thing a ledger must not do.
        """
        if not self.path.exists():
            return []
        out = []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            try:
                entry = Entry(**json.loads(line))
            except (ValueError, TypeError) as error:
                raise LedgerError(f"{self.path.name} line {number} is unreadable") from error
            if not entry.cost_inr >= 0:
                raise LedgerError(f"{self.path.name} line {number} has a negative cost")
            out.append(entry)
        return out

    def calls(self) -> dict[str, Entry]:
        """Latest line of each call, which is the settled one when there is one."""
        return {entry.call_id: entry for entry in self.entries()}

    def spent(self, purpose: str | None = None, batch: str | None = None) -> float:
        """Rupees spent or still reserved."""
        return sum(
            e.cost_inr
            for e in self.calls().values()
            if (purpose is None or e.purpose == purpose) and (batch is None or e.batch == batch)
        )

    def _append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(entry)) + "\n")
            handle.flush()
            # Forced to disk: the reservation must survive a crash during the call it covers.
            os.fsync(handle.fileno())

    def reserve(self, purpose: str, batch: str, model: str, worst_inr: float) -> str:
        """Record the worst case before a call and return the call id."""
        call_id = uuid.uuid4().hex
        self._append(Entry(call_id, _now(), purpose, batch, model, "reserved", 0, 0, 0, worst_inr))
        return call_id

    def settle(
        self,
        call_id: str,
        status: str,
        cost_inr: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        note: str = "",
    ) -> None:
        """Replace a reservation by what the call cost."""
        held = self.calls().get(call_id)
        if held is None or held.status != "reserved":
            raise LedgerError("only an open reservation can be settled")
        self._append(
            Entry(
                call_id,
                _now(),
                held.purpose,
                held.batch,
                held.model,
                status,
                input_tokens,
                output_tokens,
                thinking_tokens,
                cost_inr,
                note,
            )
        )

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Allow one process at a time.

        Two processes could each see room under the cap and cross it together.
        """
        path = self.path.with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            message = (
                f"{path.name} exists: another run is active, or one crashed. Check, then delete it"
            )
            raise LedgerError(message) from error
        try:
            yield
        finally:
            os.close(handle)
            path.unlink()
