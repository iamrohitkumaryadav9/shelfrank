"""Fill the generated README blocks from committed results."""

import json

from shelfrank.config import REPO_ROOT
from shelfrank.readme_tables import limitations, replace_block


def main() -> None:
    """Rewrite each block in place."""
    audit = json.loads((REPO_ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    readme = REPO_ROOT / "README.md"
    text = replace_block(readme.read_text(encoding="utf-8"), "limitations", limitations(audit))
    readme.write_text(text, encoding="utf-8", newline="\n")
    print("updated README.md")


if __name__ == "__main__":
    main()
