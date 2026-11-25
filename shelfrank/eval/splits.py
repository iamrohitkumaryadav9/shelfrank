"""Assign queries to cross-validation folds once, then only ever read the committed file."""

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

RARE = "__rare_or_missing__"


@dataclass(frozen=True)
class Folds:
    """A frozen fold assignment."""

    assignment: dict[int, int]
    n_folds: int
    dev_fold: int

    def queries(self, *folds: int) -> list[int]:
        """Query ids in the given folds, sorted."""
        wanted = set(folds)
        return sorted(q for q, fold in self.assignment.items() if fold in wanted)

    @property
    def dev(self) -> list[int]:
        """Queries that exploration and tuning may look at."""
        return self.queries(self.dev_fold)

    @property
    def primary(self) -> list[int]:
        """Queries the headline result is computed on."""
        return self.queries(*(f for f in range(self.n_folds) if f != self.dev_fold))


def strata(classes: Mapping[int, str | None], n_folds: int) -> dict[int, str]:
    """Stratum per query. Classes too small to appear in every fold are pooled."""
    counts = Counter(name for name in classes.values() if name)
    return {
        query_id: name if name and counts[name] >= n_folds else RARE
        for query_id, name in classes.items()
    }


def assign_folds(
    classes: Mapping[int, str | None],
    groups: Iterable[list[int]],
    n_folds: int,
    seed: int,
) -> dict[int, int]:
    """Spread each stratum over the folds, keep groups whole, keep fold sizes equal.

    Library stratifiers cannot guarantee equal fold sizes when nearly all strata are smaller than
    the fold count, which is the case here. Groups are placed first because singletons can always
    fill whatever capacity remains.
    """
    if not 1 < n_folds <= len(classes):
        raise ValueError("n_folds must be between 2 and the number of queries")
    stratum = strata(classes, n_folds)
    grouped = [sorted(group) for group in groups]
    in_group = [query_id for group in grouped for query_id in group]
    if len(in_group) != len(set(in_group)) or not set(in_group) <= set(classes):
        raise ValueError("groups must be disjoint and contain only known queries")

    rng = random.Random(seed)
    singles = [[query_id] for query_id in sorted(set(classes) - set(in_group))]
    rng.shuffle(singles)
    # Larger strata first, so balance is settled while capacity is still plentiful.
    size = Counter(stratum.values())
    singles.sort(key=lambda unit: (stratum[unit[0]] == RARE, -size[stratum[unit[0]]]))
    units = sorted(grouped, key=lambda unit: (-len(unit), unit)) + singles

    capacity = -(-len(classes) // n_folds)
    totals = [0] * n_folds
    per_stratum: dict[str, list[int]] = defaultdict(lambda: [0] * n_folds)
    assignment: dict[int, int] = {}
    for unit in units:
        counts = per_stratum[stratum[unit[0]]]
        open_folds = [f for f in range(n_folds) if totals[f] + len(unit) <= capacity]
        if not open_folds:
            raise ValueError(f"group of {len(unit)} queries does not fit in any fold")
        fold = min(open_folds, key=lambda f: (counts[f], totals[f], rng.random()))
        for query_id in unit:
            assignment[query_id] = fold
            per_stratum[stratum[query_id]][fold] += 1
            totals[fold] += 1
    return dict(sorted(assignment.items()))


def save_folds(folds: Folds, path: Path, provenance: Mapping[str, object]) -> None:
    """Write the assignment with how it was made. Query text is left out on purpose."""
    payload = {
        "n_folds": folds.n_folds,
        "dev_fold": folds.dev_fold,
        "provenance": dict(provenance),
        "assignment": {str(q): f for q, f in sorted(folds.assignment.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")


def load_folds(path: Path) -> Folds:
    """Read a committed assignment and check that it is internally consistent."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignment = {int(q): int(f) for q, f in payload["assignment"].items()}
    n_folds, dev_fold = int(payload["n_folds"]), int(payload["dev_fold"])
    if set(assignment.values()) != set(range(n_folds)) or dev_fold not in range(n_folds):
        raise ValueError(f"{path} is not a valid {n_folds}-fold assignment")
    return Folds(assignment=assignment, n_folds=n_folds, dev_fold=dev_fold)
