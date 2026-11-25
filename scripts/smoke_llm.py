"""One tiny call per model, to learn whether the key works. No retries."""

import json
import sys

from shelfrank.llm.budget import BudgetError
from shelfrank.llm.client import InvalidResponseError, TransportError
from shelfrank.llm.setup import make_client, make_request

MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
PROMPT = "Return the product type in this text as JSON: 'a wooden chair'."
SCHEMA = {
    "type": "object",
    "properties": {"product_type": {"type": "string"}},
    "required": ["product_type"],
}


def check(text: str) -> None:
    """Require the one field."""
    assert json.loads(text)["product_type"]


def main() -> int:
    """Call each model once and report usage."""
    client = make_client(live=True, max_attempts=1)
    for model in MODELS:
        request = make_request(client.config, PROMPT, SCHEMA, 256, model)
        try:
            reply = client.generate(request, "smoke", "smoke", check)
        except (TransportError, InvalidResponseError, BudgetError) as error:
            print(f"{model}: {type(error).__name__}: {error}")
            continue
        print(
            f"{model}: answered by {reply.model_version}, tokens in {reply.input_tokens}, "
            f"out {reply.output_tokens}, thinking {reply.thinking_tokens}, text {reply.text!r}"
        )
    print(client.budget.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
