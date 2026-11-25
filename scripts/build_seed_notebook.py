"""Write notebooks/esci_seed_queries.ipynb from the package source and configs/esci.yaml."""

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.qu.seed_notebook import build, dumps


def main() -> None:
    """Build and save the notebook."""
    cfg = load_config("esci")
    path = REPO_ROOT / "notebooks" / "esci_seed_queries.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        dumps(build(REPO_ROOT, cfg["seed"], cfg["sources"])), encoding="utf-8", newline="\n"
    )
    print(f"wrote {path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
