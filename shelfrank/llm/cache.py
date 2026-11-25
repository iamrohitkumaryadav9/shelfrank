"""Committed cache of LLM responses, so that every result replays offline at no cost."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Request:
    """Everything that determines a response."""

    model: str
    prompt: str
    schema: dict[str, Any]
    thinking_level: str
    max_output_tokens: int
    temperature: float = 0.0
    seed: int = 20260
    rules_version: str = ""

    def key(self) -> str:
        """Hash of all fields, so any change to a prompt or setting is a different entry.

        An empty version is left out, which keeps the keys of calls that are not labels.
        """
        fields = {k: v for k, v in asdict(self).items() if k != "rules_version" or v}
        body = json.dumps(fields, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Response:
    """A validated answer and its usage."""

    text: str
    model_version: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int


class CacheMissError(KeyError):
    """The response is not cached and calls are not allowed."""


class ResponseCache:
    """One file per response.

    The prompt is stored only as a hash. Prompts contain query text, and cached evaluation prompts
    would otherwise put test queries into the repository.
    """

    def __init__(self, directory: Path) -> None:
        """Use ``directory``, created on the first write."""
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: Request) -> Response | None:
        """Return the stored response, or None."""
        path = self._path(request.key())
        if not path.exists():
            return None
        return Response(**json.loads(path.read_text(encoding="utf-8"))["response"])

    def put(self, request: Request, response: Response) -> None:
        """Store atomically, so an interrupted write is never read as an answer."""
        path = self._path(request.key())
        path.parent.mkdir(parents=True, exist_ok=True)
        settings = {k: v for k, v in asdict(request).items() if k not in ("prompt", "schema")}
        payload = {
            "settings": settings,
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "response": asdict(response),
        }
        partial = path.with_suffix(".tmp")
        body = json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
        partial.write_text(body, encoding="utf-8", newline="\n")
        partial.replace(path)

    def __len__(self) -> int:
        """Number of stored responses."""
        return sum(1 for _ in self.directory.glob("*/*.json"))
