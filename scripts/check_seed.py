"""Check the files that came back from Colab: hashes, counts, overlap and WANDS leakage."""

import hashlib
import json
import sys

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.leakage import exact_matches, load_key_hashes, near_matches
from shelfrank.text.near_duplicates import leakage_keys

FOLDER = REPO_ROOT / "data" / "esci"


def main() -> int:
    """Print one line per check; return 1 if any fails."""
    manifest_path = FOLDER / "esci_seed_manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}; save the three Colab files into data/esci")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_config("esci")["seed"]
    results: dict[str, bool] = {"settings match configs/esci.yaml": manifest["settings"] == cfg}
    lines: dict[str, list[str]] = {}
    for name, info in manifest["files"].items():
        # Normalised, because a browser or editor may have changed the line endings.
        body = (FOLDER / name).read_bytes().replace(b"\r\n", b"\n")
        lines[name] = body.decode("utf-8").splitlines()
        results[f"{name} hash"] = hashlib.sha256(body).hexdigest() == info["sha256"]
    seed, dev = lines["esci_seed_queries.txt"], lines["esci_prompt_dev.txt"]
    results["seed has the configured size"] = len(seed) == cfg["n_seed"]
    results["prompt-dev has the configured size"] = len(dev) == cfg["n_dev"]
    results["seed and prompt-dev do not overlap"] = not (
        {k for q in seed for k in leakage_keys(q)} & {k for q in dev for k in leakage_keys(q)}
    )
    hashes = load_key_hashes(REPO_ROOT / "data/splits/wands_query_keys.json")
    results["no WANDS query (exact tokens)"] = not exact_matches(seed + dev, hashes)
    raw = REPO_ROOT / load_config("data")["wands"]["raw_dir"]
    if (raw / "query.csv").exists():
        wands = load_queries(raw)["query"]
        found = near_matches(seed + dev, wands, cfg["near_duplicate_jaccard"])
        results["no near-duplicate of a WANDS query"] = not found
    for name, passed in results.items():
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
