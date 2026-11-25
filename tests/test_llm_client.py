import json
from dataclasses import replace

import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.llm import budget as budget_module
from shelfrank.llm.batch import project, run_batch
from shelfrank.llm.budget import (
    ABSOLUTE_LIMIT_INR,
    Approval,
    Budget,
    BudgetError,
    BufferApproval,
    budget_from_config,
)
from shelfrank.llm.cache import CacheMissError, Request, Response, ResponseCache
from shelfrank.llm.client import (
    InvalidResponseError,
    LLMClient,
    RawResponse,
    TransportError,
    api_key,
    worst_case_tokens,
)
from shelfrank.llm.ledger import Ledger, LedgerError
from shelfrank.llm.setup import make_client

# Rs 1 per 1,000 input tokens and Rs 10 per 1,000 output tokens.
CONFIG = {
    "model": "m",
    "models": {"m": {"input": 10.0, "output": 100.0, "thinking_level": "minimal"}},
    "currency": {"inr_per_usd": 100.0, "safety_factor": 1.0},
}
CAPS = {"labelling": 100, "rewriting": 200, "total": 300}
APPROVED = (Approval("b", 90, "d"),)
REQUEST = Request("m", "label this", {}, "minimal", max_output_tokens=100)
OK = RawResponse('{"a": 1}', "m-001", "STOP", 20, 10, 5)


class FakeTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes) or [OK]
        self.calls = 0

    def send(self, request):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(tmp_path, *outcomes, approvals=APPROVED, caps=CAPS, buffer=(), attempts=2):
    ledger = Ledger(tmp_path / "spend.jsonl")
    transport = FakeTransport(*outcomes)
    client = LLMClient(
        CONFIG,
        ResponseCache(tmp_path / "cache"),
        Budget(ledger, caps, approvals, buffer),
        transport,
        attempts,
    )
    return client, transport, ledger


def call(client, request=REQUEST, purpose="labelling", batch="b", validate=json.loads, pilot=False):
    return client.generate(request, purpose, batch, validate, pilot)


def spend(ledger, amount, purpose="labelling", batch="b"):
    call_id = ledger.reserve(purpose, batch, "m", amount)
    ledger.settle(call_id, "ok", amount)


# Ledger


def test_reservation_counts_until_settled(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    call_id = ledger.reserve("labelling", "b", "m", 5.0)
    assert ledger.spent() == 5.0
    ledger.settle(call_id, "ok", 0.5, 1, 2, 3)
    assert ledger.spent() == 0.5 and len(ledger.entries()) == 2


def test_spend_survives_a_new_process(tmp_path):
    spend(Ledger(tmp_path / "l.jsonl"), 3.0)
    assert Ledger(tmp_path / "l.jsonl").spent() == 3.0


def test_spend_filters_by_purpose_and_batch(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 1.0, "labelling", "x")
    spend(ledger, 2.0, "rewriting", "y")
    assert (ledger.spent("labelling"), ledger.spent(batch="y"), ledger.spent()) == (1.0, 2.0, 3.0)


def test_settling_twice_or_unknown_is_refused(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    call_id = ledger.reserve("labelling", "b", "m", 1.0)
    ledger.settle(call_id, "ok", 1.0)
    for bad in (call_id, "unknown"):
        with pytest.raises(LedgerError, match="open reservation"):
            ledger.settle(bad, "ok", 0.0)


@pytest.mark.parametrize("line", ["not json", "{}", '{"call_id": "x"}'])
def test_unreadable_ledger_stops_everything(tmp_path, line):
    client, transport, ledger = make(tmp_path)
    ledger.path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 1"):
        call(client)
    assert transport.calls == 0


def test_negative_or_nan_cost_is_refused(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    for value in (-1.0, float("nan")):
        ledger.path.unlink(missing_ok=True)
        ledger.reserve("labelling", "b", "m", value)
        with pytest.raises(LedgerError, match="negative"):
            ledger.spent()


def test_lock_excludes_a_second_run_and_is_released(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    with ledger.lock(), pytest.raises(LedgerError, match="another run"), ledger.lock():
        pass
    with ledger.lock():
        pass


# Budget


def test_absolute_limit_is_500_and_lives_in_code():
    assert ABSOLUTE_LIMIT_INR == 500.0


def test_absolute_limit_refuses_whatever_the_other_caps_say(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    budget = Budget(
        ledger,
        {"labelling": 10_000, "total": 500},
        [Approval("b", 10_000, "d")],
        [BufferApproval("labelling", 10_000, "d")],
    )
    spend(ledger, 499.0)
    budget.check(1.0, "labelling", "b")
    with pytest.raises(BudgetError, match="absolute limit"):
        budget.check(1.01, "labelling", "b")


def test_absolute_limit_blocks_the_call_itself(tmp_path):
    client, transport, ledger = make(
        tmp_path,
        approvals=[Approval("b", 10_000, "d")],
        buffer=[BufferApproval("labelling", 10_000, "d")],
    )
    spend(ledger, 499.99)
    with pytest.raises(BudgetError, match="absolute limit"):
        call(client)
    assert transport.calls == 0 and ledger.spent() == 499.99


def test_absolute_limit_counts_every_purpose_and_failures(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 250.0, "rewriting", "r")
    ledger.reserve("labelling", "x", "m", 249.5)
    budget = Budget(
        ledger, CAPS, [Approval("b", 900, "d")], [BufferApproval("labelling", 900, "d")]
    )
    with pytest.raises(BudgetError, match="absolute limit"):
        budget.check(0.51, "labelling", "b")


def test_config_cannot_plan_above_the_absolute_limit(tmp_path):
    with pytest.raises(BudgetError, match="above the absolute"):
        Budget(Ledger(tmp_path / "l"), {"total": 501})


def test_purpose_cap(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 99.0, batch="other")
    budget = Budget(ledger, CAPS, [Approval("b", 90, "d")])
    with pytest.raises(BudgetError, match="labelling cap"):
        budget.check(1.5, "labelling", "b")


def test_planned_total_needs_a_recorded_buffer_approval(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 200.0, "rewriting", "r")
    spend(ledger, 99.0, "labelling", "x")
    caps = {**CAPS, "labelling": 300}
    with pytest.raises(BudgetError, match="planned total"):
        Budget(ledger, caps, [Approval("b", 90, "d")]).check(2.0, "labelling", "b")
    extended = Budget(ledger, caps, APPROVED, [BufferApproval("labelling", 5, "d")])
    extended.check(2.0, "labelling", "b")


def test_buffer_for_one_purpose_does_not_raise_another(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 99.5)
    budget = Budget(ledger, CAPS, [Approval("b", 999, "d")], [BufferApproval("rewriting", 50, "d")])
    with pytest.raises(BudgetError, match="labelling cap"):
        budget.check(1.0, "labelling", "b")


def test_unapproved_batch_is_refused_outside_a_pilot(tmp_path):
    client, transport, _ = make(tmp_path, approvals=())
    with pytest.raises(BudgetError, match="no recorded approval"):
        call(client)
    assert transport.calls == 0


def test_pilot_spend_is_limited(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    budget = Budget(ledger, CAPS)
    budget.check(1.9, "labelling", "new", pilot=True)
    spend(ledger, 1.9, batch="new")
    with pytest.raises(BudgetError, match="batch new"):
        budget.check(0.2, "labelling", "new", pilot=True)


def test_approval_limits_the_batch(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 9.5)
    with pytest.raises(BudgetError, match="batch b"):
        Budget(ledger, CAPS, [Approval("b", 10, "d")]).check(0.6, "labelling", "b")


@pytest.mark.parametrize("purpose", ["unknown", "total"])
def test_purposes_without_a_cap_are_refused(tmp_path, purpose):
    with pytest.raises(BudgetError, match="no cap"):
        Budget(Ledger(tmp_path / "l"), CAPS).check(0.1, purpose, "b", pilot=True)


@pytest.mark.parametrize("worst", [-1.0, float("nan")])
def test_nonsense_worst_case_is_refused(tmp_path, worst):
    with pytest.raises(BudgetError, match="non-negative"):
        Budget(Ledger(tmp_path / "l"), CAPS).check(worst, "labelling", "b", pilot=True)


def test_duplicate_approvals_and_negative_buffers_are_refused(tmp_path):
    with pytest.raises(BudgetError, match="more than one"):
        Budget(Ledger(tmp_path / "l"), CAPS, [Approval("b", 1, "d"), Approval("b", 9, "d")])
    with pytest.raises(BudgetError, match="must add"):
        Budget(Ledger(tmp_path / "l"), CAPS, [], [BufferApproval("labelling", -5, "d")])


def test_summary_is_against_the_absolute_limit(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    spend(ledger, 12.345)
    assert Budget(ledger, CAPS).summary() == "Rs 12.35 of 500"


# Client


def test_call_is_charged_with_thinking_tokens(tmp_path):
    client, _, ledger = make(tmp_path)
    reply = call(client)
    assert reply.thinking_tokens == 5
    entry = next(iter(ledger.calls().values()))
    assert entry.cost_inr == pytest.approx(20 / 1000 + (10 + 5) / 100)
    assert (entry.status, entry.thinking_tokens) == ("ok", 5)


def test_second_call_is_cached_and_free(tmp_path):
    client, transport, ledger = make(tmp_path)
    first = call(client)
    before = ledger.spent()
    assert call(client) == first
    assert transport.calls == 1 and ledger.spent() == before


def test_cache_replays_offline_without_ledger_or_budget(tmp_path):
    client, _, ledger = make(tmp_path)
    first = call(client)
    ledger.path.unlink()
    offline = LLMClient(CONFIG, client.cache, Budget(ledger, CAPS), transport=None)
    assert call(offline, batch="unapproved") == first


def test_offline_miss_is_an_error(tmp_path):
    client, *_ = make(tmp_path)
    client.transport = None
    with pytest.raises(CacheMissError):
        call(client)


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "other"},
        {"model": "m2"},
        {"schema": {"type": "object"}},
        {"thinking_level": "low"},
        {"max_output_tokens": 99},
        {"temperature": 0.5},
        {"seed": 1},
    ],
)
def test_any_change_is_a_different_cache_key(change):
    assert replace(REQUEST, **change).key() != REQUEST.key()


def test_cache_files_hold_no_prompt_text(tmp_path):
    client, *_ = make(tmp_path)
    call(client, replace(REQUEST, prompt="turquoise unicorn lamp", schema={"x": "zebra"}))
    stored = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "cache").rglob("*.json"))
    assert "unicorn" not in stored and "zebra" not in stored and "prompt_sha256" in stored


def test_worst_case_bounds_input_by_bytes_and_output_by_the_limit():
    worst_in, worst_out = worst_case_tokens(replace(REQUEST, prompt="é" * 10))
    assert worst_in >= 20 and worst_out == 100


def test_reservation_is_written_before_the_request_is_sent(tmp_path):
    client, transport, ledger = make(tmp_path)

    class Spy(FakeTransport):
        def send(self, request):
            entries = ledger.entries()
            assert [e.status for e in entries] == ["reserved"] and entries[0].cost_inr > 0
            return OK

    client.transport = Spy()
    call(client)


def test_crash_during_a_call_leaves_the_worst_case_counted(tmp_path):
    client, _, ledger = make(tmp_path)

    class Killed(FakeTransport):
        def send(self, request):
            raise KeyboardInterrupt

    client.transport = Killed()
    with pytest.raises(KeyboardInterrupt):
        call(client)
    worst_in, worst_out = worst_case_tokens(REQUEST)
    assert ledger.spent() == pytest.approx(worst_in / 1000 + worst_out / 100)


def test_rejected_call_is_still_counted(tmp_path):
    error = TransportError("HTTP 400", rejected=True, retryable=False)
    client, transport, ledger = make(tmp_path, error)
    with pytest.raises(TransportError):
        call(client)
    entry = next(iter(ledger.calls().values()))
    assert transport.calls == 1 and entry.status == "failed" and entry.cost_inr > 0


def test_timeout_is_counted_at_the_worst_case(tmp_path):
    error = TransportError("timeout", rejected=False, retryable=False)
    client, _, ledger = make(tmp_path, error)
    with pytest.raises(TransportError):
        call(client)
    worst_in, worst_out = worst_case_tokens(REQUEST)
    assert ledger.spent() == pytest.approx(worst_in / 1000 + worst_out / 100)


def test_every_retry_has_its_own_ledger_call(tmp_path):
    error = TransportError("HTTP 503", rejected=True, retryable=True)
    client, transport, ledger = make(tmp_path, error, OK)
    call(client)
    statuses = [e.status for e in ledger.calls().values()]
    assert transport.calls == 2 and statuses == ["failed", "ok"]
    assert ledger.spent() > sum(e.cost_inr for e in ledger.calls().values() if e.status == "ok")


def test_retries_stop_at_the_limit(tmp_path):
    error = TransportError("HTTP 503", rejected=True, retryable=True)
    client, transport, ledger = make(tmp_path, error, attempts=2)
    with pytest.raises(TransportError):
        call(client)
    assert transport.calls == 2 and len(ledger.calls()) == 2


def test_a_retry_that_would_cross_a_limit_is_not_sent(tmp_path):
    error = TransportError("HTTP 503", rejected=True, retryable=True)
    worst_in, worst_out = worst_case_tokens(REQUEST)
    worst = worst_in / 1000 + worst_out / 100
    client, transport, _ = make(tmp_path, error, approvals=[Approval("b", worst + 0.01, "d")])
    with pytest.raises(BudgetError):
        call(client)
    assert transport.calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        replace(OK, text="not json"),
        replace(OK, finish_reason="MAX_TOKENS"),
        replace(OK, model_version="other-model"),
    ],
)
def test_unusable_responses_are_charged_but_not_cached(tmp_path, bad):
    client, transport, ledger = make(tmp_path, bad)
    with pytest.raises(InvalidResponseError):
        call(client)
    entry = next(iter(ledger.calls().values()))
    assert entry.status == "invalid" and entry.cost_inr > 0
    assert len(client.cache) == 0 and transport.calls == 1


def test_missing_usage_is_charged_at_the_worst_case(tmp_path):
    client, _, ledger = make(tmp_path, replace(OK, input_tokens=None))
    with pytest.raises(InvalidResponseError, match="usage"):
        call(client)
    worst_in, worst_out = worst_case_tokens(REQUEST)
    assert ledger.spent() == pytest.approx(worst_in / 1000 + worst_out / 100)


def test_cost_above_the_worst_case_is_recorded_and_raised(tmp_path):
    client, _, ledger = make(tmp_path, replace(OK, thinking_tokens=5000))
    with pytest.raises(BudgetError, match="exceeded the worst case"):
        call(client)
    assert ledger.spent() == pytest.approx(20 / 1000 + 5010 / 100)


def test_missing_ledger_with_a_filled_cache_blocks_new_calls(tmp_path):
    client, transport, ledger = make(tmp_path)
    call(client)
    ledger.path.unlink()
    with pytest.raises(LedgerError, match="missing"):
        call(client, replace(REQUEST, prompt="new"))
    assert transport.calls == 1


def test_lock_is_released_after_an_error(tmp_path):
    error = TransportError("HTTP 400", rejected=True, retryable=False)
    client, *_ = make(tmp_path, error, OK)
    with pytest.raises(TransportError):
        call(client)
    assert call(client).text == OK.text


def test_interrupted_cache_write_is_not_an_answer(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put(REQUEST, Response("x", "m", 1, 1, 0))
    path = next(tmp_path.rglob("*.json"))
    path.rename(path.with_suffix(".tmp"))
    assert cache.get(REQUEST) is None


# Key handling


def test_key_comes_from_the_environment_first(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    assert api_key(tmp_path / ".env") == "from-env"


def test_missing_key_has_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="replay without it"):
        api_key(tmp_path / ".env")


def test_default_client_is_offline():
    assert make_client().transport is None


# Batches


def build(group):
    return replace(REQUEST, prompt="|".join(group))


def parse(text, group):
    return [json.loads(text)["a"]] * len(group)


def batch(client, items, pilot=False, name="b", size=4):
    return run_batch(
        client, name, "labelling", items, size, build, lambda _: json.loads, parse, pilot
    )


def test_batch_returns_one_result_per_item(tmp_path):
    client, transport, _ = make(tmp_path)
    assert batch(client, [str(i) for i in range(10)]) == [1] * 10
    assert transport.calls == 3


def test_interrupted_batch_resumes_without_paying_twice(tmp_path):
    error = TransportError("HTTP 400", rejected=True, retryable=False)
    client, transport, _ = make(tmp_path, OK, error, OK)
    with pytest.raises(TransportError):
        batch(client, [str(i) for i in range(8)])
    batch(client, [str(i) for i in range(8)])
    assert transport.calls == 3


def test_pilot_runs_without_approval_but_only_on_ten_items(tmp_path):
    client, transport, _ = make(tmp_path, approvals=())
    assert len(batch(client, ["x"] * 10, pilot=True, size=10, name="new")) == 10
    with pytest.raises(BudgetError, match="limited to 10"):
        batch(client, ["x"] * 11, pilot=True, name="new")
    with pytest.raises(BudgetError, match="no recorded approval"):
        batch(client, ["x"], name="new")
    assert transport.calls == 1


def test_wrong_number_of_results_is_an_error(tmp_path):
    client, *_ = make(tmp_path)
    with pytest.raises(ValueError, match="results for"):
        run_batch(
            client, "b", "labelling", ["x", "y"], 4, build, lambda _: json.loads, lambda *_: [1]
        )


def test_projection_scales_the_measured_pilot(tmp_path):
    client, _, ledger = make(tmp_path, approvals=())
    spend(ledger, 0.5, batch="new")
    out = project(client, "new", items=2000, chunk_size=20, pilot_items=10)
    assert out.inr_per_item == pytest.approx(0.05)
    assert out.projected_inr == pytest.approx(0.5 * 2 * 100)


def test_projection_of_a_chunk_no_larger_than_the_pilot_is_not_scaled_down(tmp_path):
    client, _, ledger = make(tmp_path, approvals=())
    spend(ledger, 0.5, batch="new")
    assert project(client, "new", 100, 5, 10).projected_inr == pytest.approx(0.5 * 20)


# Committed state


def test_committed_config_builds_a_valid_budget(tmp_path):
    config = load_config("llm")
    built = budget_from_config(Ledger(tmp_path / "l"), config)
    assert built.caps["labelling"] == 100 and built.caps["rewriting"] == 200
    assert config["caps_inr"]["total"] == 300 and config["buffer_approvals"] == []
    assert budget_module.ABSOLUTE_LIMIT_INR == 500


def test_committed_spend_is_within_every_limit():
    config = load_config("llm")
    ledger = Ledger(REPO_ROOT / "results/llm_spend.jsonl")
    extra = sum(b["extra_inr"] for b in config["buffer_approvals"])
    assert ledger.spent() <= min(ABSOLUTE_LIMIT_INR, config["caps_inr"]["total"] + extra)
    for purpose in ("labelling", "rewriting", "smoke"):
        assert ledger.spent(purpose) <= config["caps_inr"][purpose] + extra
    for approval in config["approvals"]:
        assert ledger.spent(batch=approval["batch"]) <= approval["max_inr"]


def test_no_reservation_is_left_open_in_the_committed_ledger():
    calls = Ledger(REPO_ROOT / "results/llm_spend.jsonl").calls().values()
    assert not [c for c in calls if c.status == "reserved"]
