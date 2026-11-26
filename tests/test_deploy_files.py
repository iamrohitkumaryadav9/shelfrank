import re

import pytest

from shelfrank.config import REPO_ROOT

HEAVY = ("torch", "transformers", "sentence-transformers", "scikit-learn", "google-genai")


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-api.txt"])
def test_deployed_requirements_are_light_and_pinned_like_the_project(name):
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    lines = [x for x in text.splitlines() if x and not x.startswith("#")]
    packages = {re.split(r"[=<>]", x)[0] for x in lines}
    assert not packages & set(HEAVY)
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in lines:
        assert f'"{line}"' in project, f"{line} differs from pyproject.toml"


def test_image_has_no_secret_and_runs_without_root():
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER runner" in docker and "API_KEY" not in docker and "torch" in docker.splitlines()[0]
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ".env" in ignored and ".git" in ignored


def test_no_env_file_is_tracked():
    import subprocess

    done = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    names = done.stdout.split()
    assert not [n for n in names if n == ".env" or n.endswith("secrets.toml")]


def test_base_dependencies_are_exactly_the_demo_set():
    """Community Cloud installs uv.lock without extras, so the base set must be the app."""
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    wanted = {x for x in text.splitlines() if x and not x.startswith("#")}
    assert set(project["dependencies"]) == wanted
    assert "app" not in project["optional-dependencies"]
    heavy = {re.split(r"[=<>]", x)[0] for x in project["dependencies"]} & set(HEAVY)
    assert not heavy


def test_ci_installs_the_way_community_cloud_does():
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked\n" in ci and "scripts/smoke_app.py" in ci
