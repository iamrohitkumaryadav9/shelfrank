import subprocess
import sys

import pytest

from scripts import reproduce as rp
from shelfrank.config import REPO_ROOT


def run_guarded(tmp_path, code):
    script = tmp_path / "s.py"
    script.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", rp.GUARD, str(script)], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_network_is_off_inside_a_step(tmp_path):
    code = "import socket\nsocket.create_connection(('192.0.2.1', 80), timeout=2)"
    done = run_guarded(tmp_path, code)
    assert done.returncode != 0 and "network access is off" in done.stderr


def test_the_key_is_not_visible_inside_a_step(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    done = run_guarded(tmp_path, "import os\nassert 'GOOGLE_API_KEY' not in os.environ")
    assert done.returncode == 0, done.stderr


def test_offline_flag_is_required(monkeypatch):
    monkeypatch.setattr("sys.argv", ["reproduce.py"])
    with pytest.raises(SystemExit):
        rp.main()


def test_no_step_can_go_live():
    assert not any("--live" in step.command for step in rp.STEPS)
    for step in rp.STEPS:
        assert (REPO_ROOT / step.command[0]).exists()


def test_fixture_steps_need_no_download():
    steps = [s for s in rp.STEPS if s.in_fixture]
    assert len(steps) >= 5 and not any(s.needs for s in steps)


def test_changed_output_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "out.json").write_text("1", encoding="utf-8")
    (tmp_path / "s.py").write_text("open('out.json', 'w').write('2')", encoding="utf-8")
    assert rp.run_step(rp.Step("x", ("s.py",), ("out.json",))) == "CHANGED: out.json"


def test_failed_step_restores_its_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "out.json").write_text("1", encoding="utf-8")
    code = "open('out.json', 'w').write('2')\nraise ValueError('no')"
    (tmp_path / "s.py").write_text(code, encoding="utf-8")
    assert rp.run_step(rp.Step("x", ("s.py",), ("out.json",))).startswith("FAILED: ValueError")
    assert (tmp_path / "out.json").read_text(encoding="utf-8") == "1"


def test_line_endings_are_not_content(tmp_path):
    (tmp_path / "a").write_bytes(b"x\r\ny")
    (tmp_path / "b").write_bytes(b"x\ny")
    assert rp.digest(tmp_path / "a") == rp.digest(tmp_path / "b")
