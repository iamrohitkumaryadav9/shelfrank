"""Write app_data/assets.json and upload the files to the data release. Run after build_app_data.py.

python scripts/publish_assets.py            write the manifest only
python scripts/publish_assets.py --upload   also create the release and upload (needs gh)
"""

import argparse
import json
import subprocess

from shelfrank.config import REPO_ROOT
from shelfrank.data.download import sha256_of

FOLDER = REPO_ROOT / "app_data"
KEPT = {"meta.json", "rewrites.json", "suggest_queries.txt", "assets.json"}
REPOSITORY, RELEASE = "iamrohitkumaryadav9/shelfrank", "v1.0-data"
NOTE = REPO_ROOT / "docs/LICENSE-DATA.md"


def main() -> None:
    """Hash, write, upload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    files = sorted(
        p
        for p in FOLDER.iterdir()
        if p.is_file()
        and p.name not in KEPT
        and p.name != "catalog.duckdb"
        and p.suffix != ".partial"
    )
    body = {
        "repository": REPOSITORY,
        "release": RELEASE,
        "license": "WANDS is MIT licensed (Copyright (c) 2021 ecir2022); the files are derived from it. See LICENSE-DATA.md in the release.",
        "files": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size} for p in files},
    }
    (FOLDER / "assets.json").write_text(
        json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    sums = "\n".join(f"{v['sha256']}  {k}" for k, v in body["files"].items()) + "\n"
    (REPO_ROOT / "artifacts").mkdir(exist_ok=True)
    checksums = REPO_ROOT / "artifacts/SHA256SUMS"
    checksums.write_text(sums, encoding="utf-8", newline="\n")
    print(f"{len(files)} files, {sum(v['bytes'] for v in body['files'].values()) / 1e6:.1f} MB")
    if args.upload:
        notes = "Serving files for the ShelfRank demo and API, derived from WANDS (MIT). Hashes: SHA256SUMS and app_data/assets.json."
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                RELEASE,
                "--repo",
                REPOSITORY,
                "--title",
                "Data for the demo and API",
                "--notes",
                notes,
            ],
            check=True,
        )
        uploads = [str(p) for p in files] + [str(checksums), str(NOTE)]
        subprocess.run(
            ["gh", "release", "upload", RELEASE, "--repo", REPOSITORY, *uploads], check=True
        )


if __name__ == "__main__":
    main()
