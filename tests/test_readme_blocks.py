import importlib.util
import json
import re

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.readme_blocks import MAPPING, spend
from shelfrank.readme_tables import replace_block

README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("mk", REPO_ROOT / "scripts/make_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_generated_block_is_up_to_date(script):
    text = README.read_text(encoding="utf-8")
    for name, body in script.readme_blocks().items():
        assert replace_block(text, name, body) == text, name


def test_every_number_outside_generated_blocks_is_a_known_constant():
    """Hand-typed numbers may describe the setup. Results live only in generated blocks."""
    text = README.read_text(encoding="utf-8")
    prose = re.sub(r"<!-- begin:(\w+) -->.*?<!-- end:\1 -->", "", text, flags=re.DOTALL)
    links = r"```.*?```|\(https?://[^)]*\)|\([\w/.#-]+\)|`[^`]*`|Apache-2.0"
    prose = re.sub(links, "", prose, flags=re.DOTALL)
    found = {n.rstrip(",") for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", prose)}
    allowed = {
        "42,994",
        "480",
        "233,448",
        "384",
        "2022",
        "2,000",
        "5",
        "0",
        "1",
        "4",
        "10",
        "2",
        "100",
        "10,000",
        "20",
        "6",
    }
    assert found <= allowed, found - allowed


def test_spend_counts_the_last_state_of_each_call():
    rows = [
        {"call_id": "a", "purpose": "labelling", "cost_inr": 5.0},
        {"call_id": "a", "purpose": "labelling", "cost_inr": 1.0},
        {"call_id": "b", "purpose": "rewriting", "cost_inr": 2.0},
    ]
    text = spend(rows)
    assert "**Rs 3.00**" in text and "labelling Rs 1.00" in text and "rewriting Rs 2.00" in text


def test_spend_in_the_readme_equals_the_budget_code():
    from shelfrank.llm.ledger import Ledger

    ledger = Ledger(REPO_ROOT / "results/llm_spend.jsonl")
    assert f"**Rs {ledger.spent():.2f}**" in README.read_text(encoding="utf-8")


def test_internal_links_exist():
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", text):
        if not target.startswith("http"):
            assert (REPO_ROOT / target).exists(), target


def test_anchors_of_the_mapping_exist():
    def anchors(path):
        heads = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in heads}

    for _, _, link in MAPPING:
        target, _, anchor = link.partition("#")
        path = REPO_ROOT / (target or "README.md")
        assert path.exists()
        if anchor:
            assert anchor in anchors(path), link


def test_silver_is_stated_wherever_slot_f1_is_shown():
    text = README.read_text(encoding="utf-8")
    assert text.count("vs LLM silver labels; human verification pending") >= 2


def test_results_page_is_up_to_date(script):
    if not (REPO_ROOT / "results/leak_checks.json").exists():
        pytest.skip("no checks yet")
    from shelfrank.eval.results_page import render

    load = script._load
    notes = json.loads((REPO_ROOT / "docs/failure_diagnoses.json").read_text(encoding="utf-8"))
    page = render(
        load("final_run"),
        load("leak_checks"),
        load("failure_cases"),
        notes,
        load("similar_primary"),
        load("suggest"),
        (REPO_ROOT / "results/rewrite_primary.json").exists(),
    )
    assert (REPO_ROOT / "docs/RESULTS.md").read_text(encoding="utf-8") == page


def test_every_failure_case_has_a_diagnosis(script):
    notes = json.loads((REPO_ROOT / "docs/failure_diagnoses.json").read_text(encoding="utf-8"))
    cases = script._load("failure_cases")["cases"]
    assert len(cases) == 20 and {str(c["query_id"]) for c in cases} == set(notes)
