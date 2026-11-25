# ShelfRank

Query understanding and learning-to-rank for home-goods product search, evaluated on
[WANDS](https://github.com/wayfair/WANDS), Wayfair's public relevance benchmark (ECIR 2022).

**Status: Phase 0, data audit. No ranking results exist yet.**

| Document | What it covers |
| --- | --- |
| [docs/data_audit.md](docs/data_audit.md) | What is in WANDS and what it means for evaluation (generated) |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Splits, metrics, label rules and what has been looked at so far |
| [docs/DATA.md](docs/DATA.md) | Source, commit, license, published baseline |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Pinned versions and why |

## Reproduce

```powershell
uv sync --extra dev
uv run python scripts/get_wands.py      # about 96 MB, verified by SHA-256
uv run python scripts/data_audit.py     # writes results/data_audit.json and docs/data_audit.md
uv run pytest
```

Individual project by Rohit Kumar, M.Tech CSE, IIIT Delhi.
