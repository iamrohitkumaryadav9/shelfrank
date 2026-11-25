"""The precomputed serving path must rank exactly as the plain one, on every WANDS query.

Writes results/fast_path.json. Compares product ids and scores as floats, without tolerance.
"""

import json

from shelfrank.app_support.encoder import QueryEncoder, fetch_model
from shelfrank.app_support.engine import LTR, Engine
from shelfrank.config import REPO_ROOT
from shelfrank.ltr.pipeline import load_split


def main() -> None:
    """Rank all 480 queries both ways."""
    folder = REPO_ROOT / "app_data"
    encoder = QueryEncoder(fetch_model(REPO_ROOT / "artifacts/onnx"), folder / "tokenizer.json")
    fast, plain = Engine(folder, encoder), Engine(folder, encoder, precomputed=False)
    _, texts, _ = load_split("all")
    different = []
    for text in texts:
        a = fast.search(text, 100, LTR, rewrites=[])
        b = plain.search(text, 100, LTR, rewrites=[])
        if [(h.product.product_id, h.score) for h in a.hits] != [
            (h.product.product_id, h.score) for h in b.hits
        ]:
            different.append(text)
    body = {"queries": len(texts), "depth": 100, "identical": len(texts) - len(different)}
    path = REPO_ROOT / "results/fast_path.json"
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(body)
    if different:
        raise SystemExit(f"{len(different)} queries differ")


if __name__ == "__main__":
    main()
