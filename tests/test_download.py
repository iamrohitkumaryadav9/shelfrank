import hashlib

import pytest

from shelfrank.data.download import ChecksumError, ensure_file, http_fetch, sha256_of

BODY = b"hello wands"
DIGEST = hashlib.sha256(BODY).hexdigest()


class FakeFetch:
    def __init__(self, body=BODY, fail=False):
        self.body, self.fail, self.calls = body, fail, 0

    def __call__(self, url, target):
        self.calls += 1
        target.write_bytes(self.body)
        if self.fail:
            raise OSError("connection reset")


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(BODY)
    assert sha256_of(path) == DIGEST


def test_downloads_into_a_new_directory(tmp_path):
    target = tmp_path / "a" / "b" / "f.csv"
    assert ensure_file("https://x", target, DIGEST, FakeFetch()) is True
    assert target.read_bytes() == BODY


def test_skips_a_verified_file(tmp_path):
    target = tmp_path / "f.csv"
    target.write_bytes(BODY)
    fetch = FakeFetch()
    assert ensure_file("https://x", target, DIGEST, fetch) is False
    assert fetch.calls == 0


def test_replaces_a_corrupted_file(tmp_path):
    target = tmp_path / "f.csv"
    target.write_bytes(b"truncat")
    assert ensure_file("https://x", target, DIGEST, FakeFetch()) is True
    assert target.read_bytes() == BODY


def test_wrong_content_is_rejected_and_leaves_nothing(tmp_path):
    target = tmp_path / "f.csv"
    with pytest.raises(ChecksumError):
        ensure_file("https://x", target, DIGEST, FakeFetch(body=b"tampered"))
    assert list(tmp_path.iterdir()) == []


def test_bad_download_does_not_destroy_the_existing_file(tmp_path):
    target = tmp_path / "f.csv"
    target.write_bytes(b"old copy")
    with pytest.raises(ChecksumError):
        ensure_file("https://x", target, DIGEST, FakeFetch(body=b"tampered"))
    assert target.read_bytes() == b"old copy"


def test_interrupted_download_leaves_no_partial_file(tmp_path):
    with pytest.raises(OSError, match="connection reset"):
        ensure_file("https://x", tmp_path / "f.csv", DIGEST, FakeFetch(fail=True))
    assert list(tmp_path.iterdir()) == []


def test_plain_http_is_refused(tmp_path):
    with pytest.raises(ValueError, match="non-HTTPS"):
        http_fetch("http://example.com/x", tmp_path / "x")


def test_config_pins_commit_and_hashes(data_cfg):
    wands = data_cfg["wands"]
    assert len(wands["commit"]) == 40
    assert "{commit}" in wands["raw_base"]
    for spec in wands["files"].values():
        assert len(spec["sha256"]) == 64 and int(spec["sha256"], 16) >= 0
