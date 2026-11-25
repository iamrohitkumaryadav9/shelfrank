"""Fetch pinned files and refuse anything whose content differs from the recorded hash."""

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

CHUNK = 1 << 20
Opener = Callable[[str, Path], None]


class ChecksumError(RuntimeError):
    """The downloaded bytes are not the bytes the results were computed from."""


def sha256_of(path: Path) -> str:
    """Hash in chunks so the 90 MB product file never has to fit in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def http_fetch(url: str, target: Path) -> None:
    """Stream ``url`` to ``target``."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS download: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        while block := response.read(CHUNK):
            out.write(block)


def ensure_file(url: str, target: Path, sha256: str, fetch: Opener = http_fetch) -> bool:
    """Make ``target`` exist with the expected hash; return True if a download happened.

    The download goes to a temporary name first, so an interrupted run cannot leave a truncated
    file that a later run would mistake for the real one.
    """
    if target.exists() and sha256_of(target) == sha256:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        fetch(url, partial)
        actual = sha256_of(partial)
        if actual != sha256:
            raise ChecksumError(f"{target.name}: expected {sha256}, got {actual}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return True
