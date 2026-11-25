"""Encode the catalogue with a configured model. About 35 minutes on a 4-core CPU, resumable."""

import sys

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_products
from shelfrank.dense.encoder import (
    encode_cached,
    load_sentence_transformer,
    model_identity,
    product_texts,
)


def main() -> None:
    """Encode with the model named on the command line."""
    name = sys.argv[1] if len(sys.argv) > 1 else "bge_small"
    cfg = load_config("retrieval")["dense"][name]
    products = load_products(REPO_ROOT / load_config("data")["wands"]["raw_dir"])
    identity = model_identity(cfg, REPO_ROOT)
    vectors = encode_cached(
        product_texts(products),
        REPO_ROOT / "artifacts" / "embeddings" / name,
        identity,
        load_sentence_transformer(**identity),
        log=lambda message: print(message, flush=True),
    )
    print(f"{name}: {vectors.shape}, norms {abs((vectors**2).sum(1) - 1).max():.1e} from 1")


if __name__ == "__main__":
    main()
