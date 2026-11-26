"""Download the serving files of app_data from the GitHub release and verify their hashes.

python scripts/fetch_assets.py
"""

from shelfrank.app_support.assets import ensure_assets
from shelfrank.config import REPO_ROOT


def main() -> None:
    """Fetch what is missing."""
    names = ensure_assets(REPO_ROOT / "app_data")
    print(f"downloaded {len(names)} files" if names else "every file is present and verified")


if __name__ == "__main__":
    main()
