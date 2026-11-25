"""The standing approval of 2026-11-26: when a batch may follow its pilot without a separate yes."""

from shelfrank.llm.budget import ABSOLUTE_LIMIT_INR

TOLERANCE = 0.20
TOTAL_LIMIT_INR = 300.0


def reasons_to_ask(
    projected_per_item: float,
    pilot_per_item: float,
    batch_projection: float,
    phase_spent: float,
    phase_cap: float,
    total_spent: float,
) -> list[str]:
    """Empty when the batch may run. Anything else means: stop and ask the owner."""
    out = []
    if abs(pilot_per_item - projected_per_item) > TOLERANCE * projected_per_item:
        out.append(
            f"the pilot cost Rs {pilot_per_item:.4f} per item, "
            f"projected Rs {projected_per_item:.4f}"
        )
    if phase_spent + batch_projection > phase_cap:
        out.append(f"the phase would reach Rs {phase_spent + batch_projection:.2f}")
    if total_spent + batch_projection > min(TOTAL_LIMIT_INR, ABSOLUTE_LIMIT_INR):
        out.append(f"the project would reach Rs {total_spent + batch_projection:.2f}")
    return out
