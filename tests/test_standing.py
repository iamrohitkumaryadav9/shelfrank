import pytest

from shelfrank.llm.standing import reasons_to_ask

OK = {
    "projected_per_item": 0.010,
    "pilot_per_item": 0.011,
    "batch_projection": 20.0,
    "phase_spent": 15.0,
    "phase_cap": 100.0,
    "total_spent": 16.0,
}


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({}, None),
        ({"pilot_per_item": 0.012}, None),
        ({"pilot_per_item": 0.0121}, "per item"),
        ({"pilot_per_item": 0.0079}, "per item"),
        ({"phase_spent": 81.0}, "phase"),
        ({"total_spent": 281.0}, "project"),
        ({"total_spent": 280.0}, None),
    ],
)
def test_standing_approval(change, problem):
    reasons = reasons_to_ask(**(OK | change))
    assert (reasons == []) if problem is None else (len(reasons) == 1 and problem in reasons[0])
