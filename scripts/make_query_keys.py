"""Write hashed token keys of all WANDS queries, used by the leakage tests in CI."""

import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.leakage import key_hashes


def main() -> None:
    """Hash and save."""
    cfg = load_config("data")["wands"]
    queries = load_queries(REPO_ROOT / cfg["raw_dir"])["query"]
    payload = {
        "wands_commit": cfg["commit"],
        "hashes": sorted(set().union(*map(key_hashes, queries))),
    }
    path = REPO_ROOT / "data/splits/wands_query_keys.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.name}: {len(payload['hashes'])} keys")


if __name__ == "__main__":
    main()
