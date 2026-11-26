# Environment

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). `uv.lock` fixes every transitive version.

```powershell
uv sync                                          # the demo only: what Streamlit Community Cloud installs
uv sync --extra dev                              # tests and data audit
uv sync --extra dev --extra bench --extra api     # everything
uv run python scripts/check_env.py                # does each compiled dependency load?
```

## Why these versions

The development laptop has Windows Smart App Control enforcing. It rejects compiled files without an
established reputation, which affects some recent wheels. Observed in this project:

| Result | Package |
| --- | --- |
| Rejected | scipy from an unpinned install (`scipy.linalg`), on every attempt |
| Rejected | the `sklearn.cluster` extension in scikit-learn 1.5.1, which breaks `import shap` |
| Rejected | the launcher `streamlit.exe`; `python -m streamlit` works |
| Rejected once, then accepted | pandas 3.1.1, Python's own `pyexpat` |
| Accepted | every version pinned in `pyproject.toml` |

Pins are therefore older releases that load. The policy is left on. Consequences:

- **SHAP values come from LightGBM's built-in TreeSHAP** (`pred_contrib=True`), which needs no extra
  package.
- **Metric cross-checks use ranx.** `pytrec_eval` has no Windows wheel and needs a C++ compiler.
- Pins do not change during the project without running `scripts/check_env.py`.
