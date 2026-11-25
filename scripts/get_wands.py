"""Download the pinned WANDS files into data/raw/wands and verify their hashes."""

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.download import ensure_file


def main() -> None:
    """Fetch every configured file, skipping those already present and correct."""
    cfg = load_config("data")["wands"]
    for name, spec in cfg["files"].items():
        url = cfg["raw_base"].format(commit=cfg["commit"], name=name)
        fetched = ensure_file(url, REPO_ROOT / cfg["raw_dir"] / name, spec["sha256"])
        print(f"{'downloaded' if fetched else 'verified  '} {name} ({spec['bytes']:,} bytes)")


if __name__ == "__main__":
    main()
