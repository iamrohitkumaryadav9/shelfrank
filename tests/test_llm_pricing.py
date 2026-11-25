import pytest

from shelfrank.config import load_config
from shelfrank.llm.pricing import Batch, cost_inr, project

CONFIG = {
    "model": "m",
    "models": {"m": {"input": 1.0, "output": 10.0}},
    "currency": {"inr_per_usd": 100.0, "safety_factor": 1.5},
}


def test_cost_by_hand():
    assert cost_inr(CONFIG, 1_000_000, 0) == pytest.approx(150)
    assert cost_inr(CONFIG, 0, 100_000) == pytest.approx(150)


def test_unknown_model_is_an_error_and_not_free():
    with pytest.raises(KeyError, match="no price"):
        cost_inr(CONFIG, 1, 1, "other")


def test_calls_round_up_per_repeat():
    assert Batch("b", 41, 0, 0, 0, queries_per_call=20).calls == 3
    assert Batch("b", 41, 0, 0, 0, queries_per_call=20, repeats=2).calls == 6


def test_instructions_are_paid_per_call_and_thinking_as_output():
    batch = Batch("b", 40, 1000, 10, 5, queries_per_call=20, thinking_per_call=100)
    assert batch.input_tokens == 2 * 1000 + 40 * 10
    assert batch.output_tokens == 40 * 5 + 2 * 100


def test_projection_uses_the_batch_tokens():
    batch = Batch("b", 20, 1000, 0, 0, thinking_per_call=0)
    assert project(CONFIG, batch) == pytest.approx(cost_inr(CONFIG, 1000, 0))


def test_committed_config_is_complete():
    config = load_config("llm")
    assert config["model"] in config["models"]
    assert config["checked"] and all(u.startswith("https://") for u in config["sources"].values())
    assert config["currency"]["safety_factor"] >= 1
    caps = config["caps_inr"]
    assert caps == {"smoke": 1, "labelling": 100, "rewriting": 200, "total": 300}


def test_planned_labelling_stays_under_the_cap():
    from scripts.project_cost import PLAN

    config = load_config("llm")
    assert sum(project(config, b) for b in PLAN) < config["caps_inr"]["labelling"]
