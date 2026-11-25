"""Write notebooks/finetune_bge_esci.ipynb. Run it after committing the modules it embeds."""

import subprocess

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.dense.finetune_notebook import EMBEDDED, build
from shelfrank.qu.seed_notebook import dumps


def main() -> None:
    """Build with the last commit that touched the embedded code."""
    command = ["git", "status", "--porcelain", "--", *EMBEDDED, "configs/esci.yaml"]
    dirty = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise SystemExit("commit the embedded modules first:\n" + dirty.stdout)
    command = ["git", "log", "-1", "--format=%H", "--", *EMBEDDED, "configs/esci.yaml"]
    commit = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    model = load_config("retrieval")["dense"]["bge_small"]
    notebook = build(REPO_ROOT, load_config("esci"), model, commit.stdout.strip())
    path = REPO_ROOT / "notebooks" / "finetune_bge_esci.ipynb"
    path.write_text(dumps(notebook), encoding="utf-8", newline="\n")
    print(f"wrote {path.name}: {path.stat().st_size:,} bytes, commit {commit.stdout.strip()[:7]}")


if __name__ == "__main__":
    main()
