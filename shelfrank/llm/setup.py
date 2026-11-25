"""Assemble the client from the repository's configuration."""

from pathlib import Path
from typing import Any

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.llm.budget import budget_from_config
from shelfrank.llm.cache import Request, ResponseCache
from shelfrank.llm.client import GeminiTransport, LLMClient, api_key
from shelfrank.llm.ledger import Ledger

LEDGER = Path("results/llm_spend.jsonl")
CACHE = Path("data/llm_cache")


def make_client(live: bool = False, max_attempts: int = 2) -> LLMClient:
    """Build a client. Offline unless ``live`` is asked for, so a test or rerun cannot spend."""
    config = load_config("llm")
    budget = budget_from_config(Ledger(REPO_ROOT / LEDGER), config)
    transport = GeminiTransport(api_key(REPO_ROOT / ".env")) if live else None
    return LLMClient(config, ResponseCache(REPO_ROOT / CACHE), budget, transport, max_attempts)


def make_request(
    config: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    model: str | None = None,
    rules_version: str = "",
) -> Request:
    """Build a request with the thinking level recorded for the model."""
    name = model or config["model"]
    level = config["models"][name]["thinking_level"]
    return Request(name, prompt, schema, level, max_output_tokens, rules_version=rules_version)
