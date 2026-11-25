"""Write data/class_terms.json from the catalogue."""

import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_products
from shelfrank.qu.class_terms import build_class_terms


def main() -> None:
    """Build and save the terms."""
    cfg = load_config("data")
    products = load_products(REPO_ROOT / cfg["wands"]["raw_dir"])
    taxonomy = json.loads((REPO_ROOT / cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    terms = build_class_terms(products, taxonomy["classes"])
    path = REPO_ROOT / "data" / "class_terms.json"
    path.write_text(
        json.dumps(terms, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    empty = [label for label, items in terms.items() if not items]
    print(f"{len(terms)} classes, {sum(map(len, terms.values()))} terms, {len(empty)} without any")
    print("without terms:", empty)


if __name__ == "__main__":
    main()
