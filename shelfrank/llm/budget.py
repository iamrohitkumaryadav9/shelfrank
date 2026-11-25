"""Refuse calls that could exceed a spending limit."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shelfrank.llm.ledger import Ledger

# The owner's all-in ceiling. A constant, so that editing a configuration file cannot raise it.
ABSOLUTE_LIMIT_INR = 500.0
# A new batch may spend this much on its 10-item pilot before it has an approval.
PILOT_LIMIT_INR = 2.0
PILOT_ITEMS = 10


class BudgetError(RuntimeError):
    """A limit would be crossed, so the call was not made."""


@dataclass(frozen=True)
class Approval:
    """A recorded yes for one batch, with the most it may spend."""

    batch: str
    max_inr: float
    date: str


@dataclass(frozen=True)
class BufferApproval:
    """A recorded yes to go beyond the planned caps for one purpose."""

    purpose: str
    extra_inr: float
    date: str


class Budget:
    """Every limit, checked against the ledger plus the worst case of the next call."""

    def __init__(
        self,
        ledger: Ledger,
        caps: Mapping[str, float],
        approvals: Sequence[Approval] = (),
        buffer: Sequence[BufferApproval] = (),
    ) -> None:
        """Reject configurations that contradict the absolute limit or each other."""
        if caps["total"] > ABSOLUTE_LIMIT_INR:
            raise BudgetError("the planned total is above the absolute limit")
        names = [approval.batch for approval in approvals]
        if len(names) != len(set(names)):
            raise BudgetError("a batch has more than one approval")
        if any(item.extra_inr < 0 for item in buffer):
            raise BudgetError("a buffer approval must add money")
        self.ledger = ledger
        self.caps = dict(caps)
        self.approvals = {approval.batch: approval for approval in approvals}
        self.buffer = list(buffer)

    def _extra(self, purpose: str | None = None) -> float:
        return sum(b.extra_inr for b in self.buffer if purpose is None or b.purpose == purpose)

    def limits(self, purpose: str, batch: str, pilot: bool) -> list[tuple[str, float, float]]:
        """Return (name, spent so far, limit) for every limit that applies to a call."""
        if purpose not in self.caps or purpose == "total":
            raise BudgetError(f"no cap is configured for {purpose!r}")
        spent = self.ledger.spent
        approval = self.approvals.get(batch)
        if approval is None and not pilot:
            raise BudgetError(f"batch {batch!r} has no recorded approval")
        batch_limit = approval.max_inr if approval else PILOT_LIMIT_INR
        return [
            ("absolute limit", spent(), ABSOLUTE_LIMIT_INR),
            ("planned total", spent(), self.caps["total"] + self._extra()),
            (f"{purpose} cap", spent(purpose), self.caps[purpose] + self._extra(purpose)),
            (f"batch {batch}", spent(batch=batch), batch_limit),
        ]

    def check(self, worst_inr: float, purpose: str, batch: str, pilot: bool = False) -> None:
        """Raise unless the worst case fits under every limit.

        The worst case and not the expected cost: a check on the expectation would let the last
        call before a limit cross it.
        """
        if not worst_inr >= 0:
            raise BudgetError("a worst case must be a non-negative number")
        for name, spent, limit in self.limits(purpose, batch, pilot):
            if spent + worst_inr > limit:
                raise BudgetError(
                    f"{name}: Rs {spent:.2f} spent + Rs {worst_inr:.2f} worst case "
                    f"would cross Rs {limit:.2f}"
                )

    def summary(self) -> str:
        """Line for gate reports."""
        return f"Rs {self.ledger.spent():.2f} of {ABSOLUTE_LIMIT_INR:.0f}"


def budget_from_config(ledger: Ledger, config: Mapping[str, Any]) -> Budget:
    """Build from configs/llm.yaml."""
    return Budget(
        ledger,
        config["caps_inr"],
        [Approval(a["batch"], a["max_inr"], str(a["date"])) for a in config["approvals"]],
        [
            BufferApproval(b["purpose"], b["extra_inr"], str(b["date"]))
            for b in config["buffer_approvals"]
        ],
    )
