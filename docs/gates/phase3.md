# Gate report: Phase 3, fine-tuned encoder

- **Artifact checks:** [phase3_checks.md](../phase3_checks.md). **Dev comparison and serving format:** [dense_dev.md](../dense_dev.md).
- **Outcome:** no clear difference from off-the-shelf as a dense retriever, so by the rule written before measuring G2 keeps
  off-the-shelf. int8 fails its tolerance, so the demo uses fp32 ONNX.
- **Not logged:** wall time, resume history and per-query held-out ESCI scores of the run. The notebook now logs them.
- No new non-dev exposure.
