"""Large serving files live on a GitHub release, not in Git. Fetch, verify and keep them.

app_data/assets.json lists each file with its SHA-256. A file is downloaded only when it is missing
or its hash differs, and a wrong download is refused. With a GITHUB_TOKEN in the environment the
files can also be read from a private repository.
"""

import json
import os
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from shelfrank.data.download import CHUNK, ensure_file, sha256_of

Fetch = Callable[[str, Path], None]
API = "https://api.github.com/repos/{repository}/releases/tags/{release}"


def manifest(folder: Path) -> dict[str, Any]:
    """The committed list of files."""
    return json.loads((folder / "assets.json").read_text(encoding="utf-8"))


def public_url(body: Mapping[str, Any], name: str) -> str:
    """Download address of a release file of a public repository."""
    return f"https://github.com/{body['repository']}/releases/download/{body['release']}/{name}"


def _stream(request: urllib.request.Request, target: Path) -> None:
    with urllib.request.urlopen(request, timeout=300) as response, target.open("wb") as out:
        while block := response.read(CHUNK):
            out.write(block)


def token_fetch(body: Mapping[str, Any], token: str) -> Fetch:
    """Read release files through the API, which also works for a private repository."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "shelfrank-assets/1.0"}
    request = urllib.request.Request(API.format(**body), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    by_name = {a["name"]: a["url"] for a in release["assets"]}

    def fetch(url: str, target: Path) -> None:
        name = url.rsplit("/", 1)[-1]
        asset = urllib.request.Request(
            by_name[name], headers=headers | {"Accept": "application/octet-stream"}
        )
        _stream(asset, target)

    return fetch


def plain_fetch(url: str, target: Path) -> None:
    """Public download, following GitHub's redirect to its file storage."""
    _stream(urllib.request.Request(url, headers={"User-Agent": "shelfrank-assets/1.0"}), target)


def missing(folder: Path) -> list[str]:
    """Files that are absent or differ from the manifest."""
    files = manifest(folder)["files"]
    return [
        name
        for name, info in files.items()
        if not (folder / name).exists() or sha256_of(folder / name) != info["sha256"]
    ]


def ensure_assets(
    folder: Path, fetch: Fetch | None = None, log: Callable[[str], None] = print
) -> list[str]:
    """Make every listed file present and correct. Returns the names downloaded.

    A folder without a manifest was built locally and has nothing to fetch.
    """
    if not (folder / "assets.json").exists():
        return []
    body = manifest(folder)
    todo = missing(folder)
    if not todo:
        return []
    if fetch is None:
        token = os.environ.get("GITHUB_TOKEN")
        fetch = token_fetch(body, token) if token else plain_fetch
    for name in todo:
        log(f"downloading {name} ({body['files'][name]['bytes'] / 1e6:.0f} MB)")
        ensure_file(public_url(body, name), folder / name, body["files"][name]["sha256"], fetch)
    return todo
