# Gate report: Phase 9, README and final audit

Checked on 2026-11-26 at the commit that adds this file.

| Check | Result |
| --- | --- |
| No secrets in history | gitleaks 8.21.2 (container) over 44 commits: no leaks. A second scan of all 2,224 objects and the release files for the exact Gemini key, Hugging Face token and GitHub token, and for key and token patterns: 0 matches. `.env` was never committed, only `.env.example` with an empty value |
| README numbers trace to results | Numbers sit in generated blocks; `tests/test_readme_blocks.py` fails on any hand-typed number outside them |
| Leakage | `tests/test_leakage.py` passes. A wider scan of every tracked file for primary queries of 3+ words finds them only where the protocol allows: post-freeze rewrite caches, the failure analysis and serving data. Seed and suggestion files hold longer ESCI queries that contain a WANDS phrase (for example "king size bed foam topper"), which the exact and near-duplicate checks allow |
| Tests and CI | 1,060 tests; ruff clean; CI runs `reproduce.py --offline --fixture` |
| Reproduce | `reproduce.py --offline`: 20 steps, 0 problems, zero API calls |
| App without a key | `tests/test_streamlit_app.py` starts the page with no key and searches |
| LLM spend | Rs 46.30 of 500 (ledger, reservations replaced by results) |
| Licenses | Code MIT (`LICENSE`); WANDS MIT, ESCI Apache-2.0 (`docs/DATA.md`, `docs/LICENSE-DATA.md`); base encoder MIT |
| Files over 1 MB | `results/final_run.json` 1.66 MB, `results/final_rankings.json` 1.06 MB; nothing larger in history |
| Hugging Face | `iamrohitkumar9/bge-small-en-v1.5-esci` is public and not gated; all 11 model files match the local artifact by SHA-256 (`results/hf_model.json`); the card's text equals `docs/hf_model_card.md` apart from CRLF line endings |
