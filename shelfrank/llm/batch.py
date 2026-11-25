"""Run items through the client in chunks, pilot first."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from shelfrank.llm.budget import PILOT_ITEMS, BudgetError
from shelfrank.llm.cache import Request
from shelfrank.llm.client import LLMClient, Validator

BuildRequest = Callable[[Sequence[str]], Request]
BuildValidator = Callable[[Sequence[str]], Validator]
Parse = Callable[[str, Sequence[str]], list]


@dataclass(frozen=True)
class Projection:
    """Measured pilot cost, scaled to a full batch."""

    pilot_items: int
    pilot_inr: float
    items: int
    remaining_calls: int
    projected_inr: float

    @property
    def inr_per_item(self) -> float:
        """Cost per item as measured, including instructions."""
        return self.pilot_inr / self.pilot_items


def chunks(items: Sequence[str], size: int) -> list[list[str]]:
    """Split in order."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_batch(
    client: LLMClient,
    name: str,
    purpose: str,
    items: Sequence[str],
    chunk_size: int,
    build: BuildRequest,
    validator: BuildValidator,
    parse: Parse,
    pilot: bool = False,
) -> list:
    """Return one parsed result per item, in order.

    A pilot takes the first items only and refuses more, so it cannot become the whole batch.
    """
    if pilot and len(items) > PILOT_ITEMS:
        raise BudgetError(f"a pilot is limited to {PILOT_ITEMS} items")
    if not pilot and name not in client.budget.approvals and client.transport is not None:
        raise BudgetError(f"batch {name!r} has no recorded approval")
    results: list = []
    for group in chunks(items, chunk_size):
        response = client.generate(build(group), purpose, name, validator(group), pilot)
        parsed = parse(response.text, group)
        if len(parsed) != len(group):
            raise ValueError(f"{len(parsed)} results for {len(group)} items")
        results.extend(parsed)
    return results


def project(
    client: LLMClient, name: str, items: int, chunk_size: int, pilot_items: int
) -> Projection:
    """Scale the pilot to the whole batch.

    Per call and not per item, rounding up: each call pays for the instructions again, and a
    full chunk is larger than the pilot, so output is scaled by items within the call.
    """
    spent = client.budget.ledger.spent(batch=name)
    calls = math.ceil(items / chunk_size)
    per_call = spent * max(1.0, chunk_size / pilot_items)
    return Projection(pilot_items, spent, items, calls, per_call * calls)
