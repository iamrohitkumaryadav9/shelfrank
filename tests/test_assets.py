import hashlib
import json

import pytest

from shelfrank.app_support import assets
from shelfrank.config import REPO_ROOT
from shelfrank.data.download import ChecksumError

BODY = b"product data"


def folder_with(tmp_path, content=BODY):
    sha = hashlib.sha256(content).hexdigest()
    body = {
        "repository": "o/r",
        "release": "v1",
        "license": "MIT",
        "files": {"a.bin": {"sha256": sha, "bytes": len(content)}},
    }
    (tmp_path / "assets.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_missing_files_are_downloaded_from_the_release_and_verified(tmp_path):
    seen = []

    def fetch(url, target):
        seen.append(url)
        target.write_bytes(BODY)

    folder = folder_with(tmp_path)
    assert assets.ensure_assets(folder, fetch, lambda m: None) == ["a.bin"]
    assert seen == ["https://github.com/o/r/releases/download/v1/a.bin"]
    assert assets.ensure_assets(folder, lambda u, t: pytest.fail("fetched again"), print) == []


def test_a_wrong_download_is_refused_and_not_kept(tmp_path):
    folder = folder_with(tmp_path)
    with pytest.raises(ChecksumError):
        assets.ensure_assets(folder, lambda u, t: t.write_bytes(b"other"), lambda m: None)
    assert not (folder / "a.bin").exists() and not list(folder.glob("*.part"))


def test_a_changed_local_file_is_replaced(tmp_path):
    folder = folder_with(tmp_path)
    (folder / "a.bin").write_bytes(b"stale")
    assert assets.missing(folder) == ["a.bin"]
    assets.ensure_assets(folder, lambda u, t: t.write_bytes(BODY), lambda m: None)
    assert (folder / "a.bin").read_bytes() == BODY


def test_a_token_selects_the_api_and_is_never_in_a_url(tmp_path, monkeypatch):
    calls = []

    class Reply:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            out, self.data = self.data, b""
            return out

    def urlopen(request, timeout):
        calls.append((request.full_url, dict(request.header_items())))
        if "releases/tags" in request.full_url:
            return Reply(
                json.dumps(
                    {"assets": [{"name": "a.bin", "url": "https://api.github.com/assets/1"}]}
                ).encode()
            )
        return Reply(BODY)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    assets.ensure_assets(folder_with(tmp_path), log=lambda m: None)
    assert all("secret-token" not in url for url, _ in calls)
    assert calls[-1][0] == "https://api.github.com/assets/1"
    assert calls[-1][1]["Accept"] == "application/octet-stream"


def test_committed_manifest_lists_every_large_file_and_git_holds_none():
    import subprocess

    body = assets.manifest(REPO_ROOT / "app_data")
    assert body["release"] == "v1.0-data" and "MIT" in body["license"]
    assert {"catalog.duckdb.gz", "vectors.npy", "rankers.json", "class_student.npz"} <= set(
        body["files"]
    )
    tracked = subprocess.run(
        ["git", "ls-files", "app_data"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.split()
    assert sorted(tracked) == [
        "app_data/assets.json",
        "app_data/meta.json",
        "app_data/rewrites.json",
        "app_data/suggest_queries.txt",
    ]


def test_data_license_note_reproduces_the_mit_notice():
    text = (REPO_ROOT / "docs/LICENSE-DATA.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2021 ecir2022" in text and "shall be included in all" in text


def test_a_locally_built_folder_needs_no_manifest(tmp_path):
    assert assets.ensure_assets(tmp_path, lambda u, t: pytest.fail("fetched"), print) == []
