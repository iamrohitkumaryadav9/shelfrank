import io
import json
import urllib.error

import pytest

from shelfrank.app_support.gemini import live_rewrites

KEY = "secret-key-value"
ANSWER = {"product_type": "settee", "attributes": ["green"], "synonyms": []}


def reply(text):
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()


def test_rewrites_are_parsed_and_the_key_travels_in_a_header_only():
    seen = {}

    def send(request):
        seen["url"], seen["headers"], seen["data"] = request.full_url, request.headers, request.data
        return reply(json.dumps(ANSWER))

    assert live_rewrites("grn settee", KEY, send) == (["green settee"], "")
    assert KEY not in seen["url"] and KEY.encode() not in seen["data"]
    assert KEY in seen["headers"].values()


@pytest.mark.parametrize(
    "send",
    [
        lambda r: reply("not json"),
        lambda r: b"{}",
        lambda r: (_ for _ in ()).throw(OSError(f"failed with {KEY}")),
        lambda r: (_ for _ in ()).throw(
            urllib.error.HTTPError(f"https://x/?k={KEY}", 403, KEY, {}, io.BytesIO(KEY.encode()))
        ),
    ],
)
def test_problems_are_reported_without_the_key(send):
    found, problem = live_rewrites("x", KEY, send)
    assert found == [] and problem and KEY not in problem
