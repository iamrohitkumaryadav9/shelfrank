"""The only code path that may call an LLM: cache, then budget, then reserve, then call."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from shelfrank.llm.budget import Budget, BudgetError
from shelfrank.llm.cache import CacheMissError, Request, Response, ResponseCache
from shelfrank.llm.ledger import LedgerError
from shelfrank.llm.pricing import cost_inr

Validator = Callable[[str], object]


class TransportError(Exception):
    """A call that returned no usable response.

    ``rejected`` means the server answered with an error, so no output was generated. Otherwise (a
    timeout, a dropped connection) the output may have been generated and billed.
    """

    def __init__(self, message: str, rejected: bool, retryable: bool) -> None:
        """Keep what the ledger and the retry loop need."""
        super().__init__(message)
        self.rejected = rejected
        self.retryable = retryable


class InvalidResponseError(RuntimeError):
    """A paid response that failed validation."""


@dataclass(frozen=True)
class RawResponse:
    """What a transport returns. Usage is None when the server did not report it."""

    text: str
    model_version: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None


class Transport(Protocol):
    """Sends exactly one request. It must not retry: a hidden retry is spend the ledger misses."""

    def send(self, request: Request) -> RawResponse:
        """Send once."""
        ...


def worst_case_tokens(request: Request) -> tuple[int, int]:
    """Upper bounds on input and output tokens.

    A token is at least one byte, so the byte count bounds the input. ``max_output_tokens`` bounds
    the output and includes thinking. The margin covers message framing.
    """
    schema_bytes = len(repr(request.schema).encode("utf-8"))
    return len(request.prompt.encode("utf-8")) + schema_bytes + 64, request.max_output_tokens


class LLMClient:
    """Calls a model within the budget and remembers the answers."""

    def __init__(
        self,
        config: Mapping[str, Any],
        cache: ResponseCache,
        budget: Budget,
        transport: Transport | None = None,
        max_attempts: int = 2,
    ) -> None:
        """Without a transport, only cached answers are available."""
        self.config = config
        self.cache = cache
        self.budget = budget
        self.transport = transport
        self.max_attempts = max_attempts

    def generate(
        self,
        request: Request,
        purpose: str,
        batch: str,
        validate: Validator,
        pilot: bool = False,
    ) -> Response:
        """Return a cached or new response that passes ``validate``."""
        cached = self.cache.get(request)
        if cached is not None:
            validate(cached.text)
            return cached
        if self.transport is None:
            raise CacheMissError("response is not cached and the client is offline")
        if len(self.cache) and not self.budget.ledger.path.exists():
            raise LedgerError("responses are cached but the ledger is missing; restore it first")
        with self.budget.ledger.lock():
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return self._call(request, purpose, batch, validate, pilot)
                except TransportError as error:
                    if not error.retryable or attempt == self.max_attempts:
                        raise
        raise AssertionError("unreachable")

    def _cost(self, request: Request, input_tokens: int, output_tokens: int) -> float:
        return cost_inr(self.config, input_tokens, output_tokens, request.model)

    def _call(
        self, request: Request, purpose: str, batch: str, validate: Validator, pilot: bool
    ) -> Response:
        """Make one attempt. Every path through here leaves a line in the ledger."""
        assert self.transport is not None
        ledger = self.budget.ledger
        worst_in, worst_out = worst_case_tokens(request)
        worst = self._cost(request, worst_in, worst_out)
        self.budget.check(worst, purpose, batch, pilot)
        call_id = ledger.reserve(purpose, batch, request.model, worst)
        try:
            raw = self.transport.send(request)
        except TransportError as error:
            # Documented as free when rejected, but counted as if the input were billed.
            cost = self._cost(request, worst_in, 0) if error.rejected else worst
            ledger.settle(call_id, "failed", cost, note=str(error)[:200])
            raise
        except Exception as error:
            ledger.settle(call_id, "failed", worst, note=type(error).__name__)
            raise

        if raw.input_tokens is None or raw.output_tokens is None:
            ledger.settle(call_id, "no usage reported", worst)
            raise InvalidResponseError("the response reported no token usage")
        thinking = raw.thinking_tokens or 0
        actual = self._cost(request, raw.input_tokens, raw.output_tokens + thinking)
        usage = (raw.input_tokens, raw.output_tokens, thinking)
        problem = self._problem(request, raw, validate)
        if problem:
            ledger.settle(call_id, "invalid", actual, *usage, note=problem[:200])
            raise InvalidResponseError(problem)
        ledger.settle(call_id, "ok", actual, *usage)
        response = Response(raw.text, raw.model_version, *usage)
        self.cache.put(request, response)
        if actual > worst:
            raise BudgetError(f"actual cost Rs {actual:.4f} exceeded the worst case Rs {worst:.4f}")
        return response

    @staticmethod
    def _problem(request: Request, raw: RawResponse, validate: Validator) -> str:
        """Say why a response cannot be used, or return an empty string."""
        if not raw.model_version.startswith(request.model):
            return f"asked for {request.model}, answered by {raw.model_version}"
        if raw.finish_reason != "STOP":
            return f"finish reason {raw.finish_reason}"
        try:
            validate(raw.text)
        except Exception as error:
            return f"validation failed: {type(error).__name__}"
        return ""


def api_key(env_path: Path) -> str:
    """Read the key from the environment or ``.env`` without putting it anywhere else."""
    if "GOOGLE_API_KEY" not in os.environ and env_path.exists():
        from dotenv import dotenv_values

        value = dotenv_values(env_path).get("GOOGLE_API_KEY")
    else:
        value = os.environ.get("GOOGLE_API_KEY")
    if not value:
        raise RuntimeError("GOOGLE_API_KEY is not set; cached results still replay without it")
    return value


class GeminiTransport:
    """Google's SDK with retries off and the lowest thinking level the model lists."""

    def __init__(self, key: str, timeout_seconds: int = 120) -> None:
        """Import lazily, so that offline use needs no SDK."""
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _config(self, request: Request) -> Any:  # noqa: ANN401
        types = self._types
        thinking = None
        if request.thinking_level != "off":
            level = types.ThinkingLevel[request.thinking_level.upper()]
            thinking = types.ThinkingConfig(thinking_level=level)
        return types.GenerateContentConfig(
            temperature=request.temperature,
            seed=request.seed,
            max_output_tokens=request.max_output_tokens,
            candidate_count=1,
            response_mime_type="application/json",
            response_json_schema=request.schema,
            thinking_config=thinking,
        )

    def send(self, request: Request) -> RawResponse:
        """Send once and translate errors without repeating the request or the key."""
        from google.genai import errors

        try:
            reply = self._client.models.generate_content(
                model=request.model, contents=request.prompt, config=self._config(request)
            )
        except errors.APIError as error:
            code = int(getattr(error, "code", 0) or 0)
            message = f"HTTP {code} {getattr(error, 'status', '')}"
            retryable = code in (429, 500, 503)
            # A gateway timeout can arrive after the model has generated output.
            raise TransportError(message, rejected=code != 504, retryable=retryable) from None
        except Exception as error:
            raise TransportError(type(error).__name__, rejected=False, retryable=False) from None
        usage = reply.usage_metadata
        candidate = reply.candidates[0] if reply.candidates else None
        finish = candidate.finish_reason.name if candidate and candidate.finish_reason else "NONE"
        return RawResponse(
            text=reply.text or "",
            model_version=reply.model_version or "",
            finish_reason=finish,
            input_tokens=usage.prompt_token_count if usage else None,
            # A response can be all thinking, in which case no candidate count is reported.
            output_tokens=(usage.candidates_token_count or 0) if usage else None,
            thinking_tokens=usage.thoughts_token_count if usage else None,
        )
