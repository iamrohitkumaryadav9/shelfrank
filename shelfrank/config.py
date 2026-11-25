"""Locate the repository and read its YAML configuration."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(name: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Read ``configs/<name>.yaml`` with the safe loader, because configs are data, not code."""
    path = root / "configs" / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded
