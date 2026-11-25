"""The selected class student, trained once and cached."""

import json
from pathlib import Path

import pandas as pd

from shelfrank.config import REPO_ROOT
from shelfrank.qu.class_data import arm_examples, seed_parts
from shelfrank.qu.class_student import LinearClassifier

MAX_FEATURES = 60_000


def class_model(
    products: pd.DataFrame, taxonomy: list[str], root: Path = REPO_ROOT
) -> LinearClassifier:
    """The student selected in results/class_dev.json."""
    import joblib

    results = json.loads((root / "results/class_dev.json").read_text(encoding="utf-8"))
    chosen = results["best"][results["selected"]]
    arm, tail, share = chosen["arm"], chosen["tail"], chosen["query_share"]
    low, high = chosen["ngrams"]
    name = f"class_{arm}_s{share}_tail{tail}_n{low}{high}_a{chosen['alpha']:.0e}.joblib"
    path = root / "artifacts" / name
    if path.exists():
        return joblib.load(path)
    model = LinearClassifier((low, high), chosen["alpha"], MAX_FEATURES)
    parts = seed_parts(root)
    queries = parts[0] if parts else None
    model.fit(arm_examples(arm, tail, share, products, taxonomy, queries))
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    return model
