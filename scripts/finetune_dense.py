"""Fine-tune the dense retriever on ESCI. Every stage can be repeated, and a repeat continues.

The notebook notebooks/finetune_bge_esci.ipynb calls these stages on a Colab GPU. ESCI is never
downloaded on the development laptop.
"""

import csv
import hashlib
import json
import platform
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shelfrank.data.download import ensure_file
from shelfrank.dense.esci_data import TrainingData, make_batches, make_triplets, prepare
from shelfrank.dense.finetune import (
    Checkpoints,
    RunLog,
    TrainConfig,
    encode,
    held_out_scores,
    train,
)
from shelfrank.dense.onnx_export import agreement, export_fp32, onnx_encode, quantize

Log = Callable[[str], None]
MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)
# Invented, so that vectors made on the GPU machine can be compared with vectors made elsewhere.
REFERENCE_TEXTS = (
    "tangerine ukulele case with zipper",
    "two person kayak paddle set",
    "left handed fountain pen purple ink",
    "stainless steel bicycle bell",
    "gluten free sourdough starter kit",
    "waterproof hiking gaiters size medium",
    "usb c to ethernet adapter gigabit",
    "wooden chess pieces weighted no board",
    "toddler rain boots with dinosaurs",
    "replacement strap for a diving watch",
    "noise cancelling earbuds for running",
    "telescopic fishing rod saltwater",
    "dog harness reflective extra small",
    "refillable deodorant cedar scent",
    "mechanical keyboard with quiet switches",
    "Ukulele . Acme . tangerine . Concert size with a padded carrying bag",
    "Kayak paddle . Acme . Two pieces that lock together, 220 cm, fibreglass",
    "Fountain pen . Acme . purple . Fine nib for left handed writers, two cartridges",
    "Bicycle bell . Acme . silver . Fits handlebars from 22 to 32 mm",
    "Keyboard . Acme . black . 87 keys, hot swappable, quiet linear switches",
)


@dataclass(frozen=True)
class Paths:
    """Where things live. ``work`` is fast and lost with the session; ``keep`` survives it."""

    work: Path
    keep: Path

    @property
    def prepared(self) -> Path:
        """Triplet data, kept so that a new session skips the 1 GB product file."""
        return self.keep / "training_data.json"

    @property
    def checkpoints(self) -> Path:
        """Training state."""
        return self.keep / "checkpoints"

    @property
    def results(self) -> Path:
        """Measurements so far."""
        return self.keep / "finetune_results.json"

    @property
    def run_log(self) -> Path:
        """Sessions, resumes and every step's loss."""
        return self.keep / "run_log.json"

    @property
    def output(self) -> Path:
        """The finished package."""
        return self.keep / "package"


def read_results(paths: Paths) -> dict[str, Any]:
    """Measurements from earlier stages."""
    if not paths.results.exists():
        return {}
    return json.loads(paths.results.read_text(encoding="utf-8"))


def write_results(paths: Paths, **values: Any) -> None:  # noqa: ANN401
    """Add measurements."""
    body = json.dumps(read_results(paths) | values, indent=1)
    paths.results.write_text(body + "\n", encoding="utf-8")


def stage_data(
    paths: Paths, sources: Mapping[str, Any], settings: Mapping[str, Any], log: Log = print
) -> TrainingData:
    """Download, filter and save the training data, or load what an earlier run saved."""
    paths.keep.mkdir(parents=True, exist_ok=True)
    if paths.prepared.exists():
        log("training data: loaded from the kept folder")
        return TrainingData.load(paths.prepared)
    files = {name: paths.work / f"{name}.file" for name in sources}
    for name, spec in sources.items():
        files[name].parent.mkdir(parents=True, exist_ok=True)
        ensure_file(spec["url"], files[name], spec["sha256"])
        log(f"downloaded {name}")
    # WANDS queries are read by code only, to remove look-alikes. Never printed or saved.
    with files["wands_queries"].open(encoding="utf-8", newline="") as handle:
        protected = [row["query"] for row in csv.DictReader(handle, delimiter="\t")]
    data = prepare(files["examples"], files["products"], protected, settings)
    data.save(paths.prepared)
    write_results(paths, data=data.report, settings=dict(settings))
    log(json.dumps(data.report, indent=1))
    return data


def load_base(directory: Path) -> tuple[Any, Any]:
    """Base model and tokenizer from a local folder."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModel.from_pretrained(directory, add_pooling_layer=False)
    return model, tokenizer


def download_base(work: Path, model: str, revision: str) -> Path:
    """Fetch the pinned base model."""
    from huggingface_hub import snapshot_download

    target = work / "base"
    allow = [*MODEL_FILES, "model.safetensors"]
    snapshot_download(model, revision=revision, local_dir=target, allow_patterns=allow)
    return target


def data_fingerprint(data: TrainingData) -> str:
    """Hash of the training data, so checkpoints never continue on other data."""
    body = json.dumps(asdict(data), sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def stage_train(
    paths: Paths,
    data: TrainingData,
    base: Path,
    config: TrainConfig,
    query_prefix: str,
    device: str,
    log: Log = print,
) -> dict[str, Any]:
    """Measure the base model once, then train or continue training."""
    model, tokenizer = load_base(base)
    if "held_out_ndcg_before" not in read_results(paths):
        scores = held_out_scores(model, tokenizer, data, config, query_prefix, device)
        before = sum(scores.values()) / len(scores)
        write_results(paths, held_out_ndcg_before=before, held_out_before_per_query=scores)
        log(f"held-out ESCI nDCG of the base model: {before:.4f}")
    batches = make_batches(
        make_triplets(data.train, data.passages, config.seed), config.batch_size, config.seed
    )
    if not batches:
        raise ValueError("not enough queries for one batch")
    hard = [t.hard for batch in batches for t in batch]
    # The time limit and the logging interval do not change the result, so they may change.
    identity = {
        **{k: v for k, v in asdict(config).items() if k not in ("max_minutes", "log_every")},
        "data": data_fingerprint(data),
        "query_prefix": query_prefix,
    }
    checkpoints = Checkpoints(paths.checkpoints, identity)
    record = RunLog(paths.run_log)
    out = train(
        model, tokenizer, data, batches, config, checkpoints, query_prefix, device, log, record
    )
    write_results(
        paths,
        train=out,
        config=asdict(config),
        triplets=len(hard),
        hard_negative_share=sum(hard) / len(hard),
    )
    return out


def trained_model(paths: Paths, base: Path) -> tuple[Any, Any]:
    """Base architecture with the newest trained weights."""
    results = read_results(paths)
    if not results.get("train", {}).get("finished"):
        raise RuntimeError("training has not finished: run the training cell again")
    model, tokenizer = load_base(base)
    newest = None
    for note in paths.checkpoints.glob("slot_*.json"):
        info = json.loads(note.read_text(encoding="utf-8"))
        if info["complete"] and (newest is None or info["step"] > newest[0]):
            newest = (info["step"], note.with_suffix(".pt"))
    assert newest is not None
    state = torch.load(newest[1], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model, tokenizer


def stage_evaluate(
    paths: Paths,
    data: TrainingData,
    base: Path,
    config: TrainConfig,
    query_prefix: str,
    device: str,
    log: Log = print,
) -> None:
    """Held-out ESCI nDCG of the trained model."""
    model, tokenizer = trained_model(paths, base)
    scores = held_out_scores(model, tokenizer, data, config, query_prefix, device)
    after = sum(scores.values()) / len(scores)
    write_results(paths, held_out_ndcg_after=after, held_out_after_per_query=scores)
    log(f"held-out ESCI nDCG: {read_results(paths)['held_out_ndcg_before']:.4f} -> {after:.4f}")


def save_model(model: Any, base: Path, target: Path) -> None:  # noqa: ANN401
    """Write weights with the base model's own configuration files.

    Copying them keeps the folder readable by the older library versions used locally.
    """
    from safetensors.torch import save_file

    target.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (base / name).exists():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, target / name)
    weights = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(target / "model.safetensors"), metadata={"format": "pt"})


def export_int8(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    target: Path,
    work: Path,
    texts: list[str],
    limit: int,
) -> dict[str, Any]:
    """Write model_int8.onnx and measure how far it is from PyTorch."""
    fp32 = work / f"{target.name}_fp32.onnx"
    export_fp32(model, tokenizer, fp32)
    out = target / "model_int8.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize(fp32, out)
    reference = encode(model, tokenizer, texts, limit, "cpu").numpy()
    started = time.perf_counter()
    vectors = onnx_encode(out, tokenizer, texts, limit)
    return {
        "megabytes": out.stat().st_size / 1e6,
        "cosine_to_pytorch": agreement(reference, vectors),
        "ms_per_text_colab_cpu": 1000 * (time.perf_counter() - started) / len(texts),
    }


def stage_package(
    paths: Paths,
    data: TrainingData,
    base: Path,
    config: TrainConfig,
    query_prefix: str,
    provenance: Mapping[str, Any],
    log: Log = print,
) -> Path:
    """Weights, int8 files and model card in one zip. A failed export does not lose the weights."""
    model, tokenizer = trained_model(paths, base)
    target = paths.output / "bge_small_esci"
    save_model(model, base, target)
    sample = [query_prefix + data.queries[q] for q in sorted(data.held_out)[:100]]
    original, _ = load_base(base)
    exports: dict[str, Any] = {}
    jobs = (("bge_small_esci", model), ("bge_small_base", original))
    for name, network in jobs:
        try:
            exports[name] = export_int8(
                network, tokenizer, paths.output / name, paths.work, sample, config.max_query_tokens
            )
        except Exception as error:  # an export problem must not block the download
            exports[name] = {"error": f"{type(error).__name__}: {error}"}
        log(f"{name}: {exports[name]}")
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    write_results(paths, onnx=exports, provenance=dict(provenance), versions=versions)
    if paths.run_log.exists():
        write_results(paths, run=json.loads(paths.run_log.read_text(encoding="utf-8")))
    save_reference(model, tokenizer, config.max_passage_tokens, target)
    results = read_results(paths)
    (target / "training.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (target / "README.md").write_text(model_card(results), encoding="utf-8")
    archive = shutil.make_archive(str(paths.keep / "bge_small_esci"), "zip", paths.output)
    log(f"{archive}  {Path(archive).stat().st_size / 1e6:.0f} MB")
    return Path(archive)


def save_reference(model: Any, tokenizer: Any, limit: int, target: Path) -> None:  # noqa: ANN401
    """Vectors of fixed texts, to check that another machine reproduces this model."""
    vectors = encode(model, tokenizer, REFERENCE_TEXTS, limit, "cpu")
    body = {"texts": list(REFERENCE_TEXTS), "max_tokens": limit, "vectors": vectors.tolist()}
    path = target / "reference_embeddings.json"
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")


def run_lines(results: Mapping[str, Any]) -> str:
    """Wall time and resumes, or "not logged". Never an estimate."""
    run = results.get("run")
    if not run:
        return "Wall time: not logged. Resume history: not logged."
    minutes = ", ".join(f"{s['seconds'] / 60:.1f}" for s in run["sessions"])
    resumes = "; ".join(r["message"] for r in run["resumes"]) or "none"
    return f"Training minutes per session: {minutes}. Resumes: {resumes}."


def held_out_lines(results: Mapping[str, Any]) -> str:
    """The in-domain check, with a paired interval only when per-query scores were kept."""
    before, after = results["held_out_ndcg_before"], results.get("held_out_ndcg_after")
    text = "not measured" if after is None else f"{before:.4f} before, {after:.4f} after"
    rows = [results.get(f"held_out_{x}_per_query") for x in ("before", "after")]
    if rows[0] and rows[1]:
        difference = np.array([rows[1][q] - rows[0][q] for q in sorted(rows[0])])
        pick = np.random.default_rng(20260).integers(0, len(difference), (10_000, len(difference)))
        low, high = np.percentile(difference[pick].mean(axis=1), [2.5, 97.5])
        return f"{text}. Paired difference {difference.mean():+.4f} [{low:+.4f}, {high:+.4f}]."
    return f"{text}. Per-query scores were not saved, so there is no confidence interval."


def model_card(results: Mapping[str, Any]) -> str:
    """Plain description of what was trained, on what, and what was measured."""
    data = results["data"]
    config = "\n".join(f"| {k} | {v} |" for k, v in results["config"].items())
    return f"""# bge-small-en-v1.5 fine-tuned on ESCI (US, small version)

Base: {results["provenance"]["base_model"]} at {results["provenance"]["base_revision"]}.
Code: commit {results["provenance"]["git_commit"]}.

Trained on {results["triplets"]:,} triplets from {data["train_queries"]:,} ESCI training queries
(Exact as positive, Irrelevant of the same query as hard negative for
{results["hard_negative_share"]:.0%} of them). Queries that resemble a WANDS query were removed.
No WANDS data was used.

Held-out ESCI nDCG over judged products, {data["held_out_queries"]:,} queries set aside, of which those
without an Exact product are skipped: {held_out_lines(results)}
This is an in-domain check. It says nothing about WANDS, which is measured separately.

Steps: {results["train"]["step"]:,} of {results["train"]["total"]:,}. {run_lines(results)}

| Setting | Value |
| --- | --- |
{config}

ESCI is licensed Apache-2.0 by Amazon. bge-small-en-v1.5 is licensed MIT by BAAI.
"""
