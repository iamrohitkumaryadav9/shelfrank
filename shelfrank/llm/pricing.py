"""Convert tokens to rupees, pessimistically."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Batch:
    """A planned set of calls, described by token assumptions."""

    name: str
    queries: int
    prompt_tokens: int
    tokens_per_query: int
    output_per_query: int
    queries_per_call: int = 20
    thinking_per_call: int = 200
    repeats: int = 1

    @property
    def calls(self) -> int:
        """Calls needed, rounding the last partial one up."""
        return -(-self.queries // self.queries_per_call) * self.repeats

    @property
    def input_tokens(self) -> int:
        """Instructions are paid once per call, query text once per query."""
        return self.calls * self.prompt_tokens + self.repeats * self.queries * self.tokens_per_query

    @property
    def output_tokens(self) -> int:
        """Visible output plus thinking, which is billed at the output price."""
        visible = self.repeats * self.queries * self.output_per_query
        return visible + self.calls * self.thinking_per_call


def cost_inr(
    config: Mapping[str, Any], input_tokens: int, output_tokens: int, model: str | None = None
) -> float:
    """Cost including the safety factor. An unknown model is an error, never a free call."""
    name = model or config["model"]
    if name not in config["models"]:
        raise KeyError(f"no price recorded for {name}")
    price = config["models"][name]
    usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return usd * config["currency"]["inr_per_usd"] * config["currency"]["safety_factor"]


def project(config: Mapping[str, Any], batch: Batch, model: str | None = None) -> float:
    """Projected cost of a batch."""
    return cost_inr(config, batch.input_tokens, batch.output_tokens, model)
