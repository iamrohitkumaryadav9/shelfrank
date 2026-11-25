# Gate report: Phase 8, demo and production slice

- Streamlit demo, FastAPI service and Docker image share one engine, which uses the evaluation's feature code. Agreement and latency: README and
  `results/latency.json`.
- The image was built and run locally: healthy, non-root, `/search` and `/similar` answer. CI does not build it, because it downloads a 133 MB model.
- `app_data` is committed. The database is gzipped, because GitHub refuses files above 100 MB.
- Cached rewrites and the trigger are empty until Phase 5 is approved.
