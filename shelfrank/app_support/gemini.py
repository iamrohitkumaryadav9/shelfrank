"""One live rewrite for the demo, with the visitor's own key.

Plain HTTPS from the standard library, so the demo needs no SDK. The key is sent in a header, never
in the URL, and is not stored, logged or repeated in an error.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from shelfrank.config import load_config
from shelfrank.rewrite.prompt import MAX_OUTPUT, SCHEMA, parse, rewrite_prompt

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
Send = Callable[[urllib.request.Request], bytes]


def _send(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=20) as reply:
        return bytes(reply.read())


def live_rewrites(query: str, key: str, send: Send = _send) -> tuple[list[str], str]:
    """Return (rewrites, problem). The problem never contains the key."""
    model = load_config("llm")["model"]
    body: dict[str, Any] = {
        "contents": [{"parts": [{"text": rewrite_prompt(query)}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": MAX_OUTPUT,
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key.strip()},
    )
    try:
        reply = json.loads(send(request))
        text = reply["candidates"][0]["content"]["parts"][0]["text"]
        return parse(text).texts(query), ""
    except urllib.error.HTTPError as error:
        return [], f"Gemini answered with HTTP {error.code}."
    except Exception as error:
        return [], f"The rewrite failed ({type(error).__name__})."
