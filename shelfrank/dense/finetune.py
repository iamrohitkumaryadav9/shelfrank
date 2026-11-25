"""Fine-tune a bi-encoder with in-batch and hard negatives, resumable from checkpoints.

A plain PyTorch loop and not a trainer library: the loss is ten lines, resuming is exact because
every step sets its own random seed, and it does not depend on which library versions Colab ships.
"""

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from shelfrank.dense.esci_data import ESCI_GAINS, Judged, TrainingData, Triplet

Log = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    """Every training setting, written to the model card."""

    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_share: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scale: float = 20.0
    max_query_tokens: int = 32
    max_passage_tokens: int = 128
    checkpoint_every: int = 500
    log_every: int = 50
    max_minutes: float = 100.0
    seed: int = 20260


def embed(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Unit vectors from the first token, which is how bge models pool."""
    hidden = model(**batch).last_hidden_state[:, 0]
    return nn.functional.normalize(hidden.float(), dim=-1)


def contrastive_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    blocked: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Multiple-negatives ranking loss over in-batch positives and all hard negatives.

    ``blocked`` marks candidates known to be Exact for a query other than its own positive. Scoring
    them as negatives would teach the model something false.
    """
    scores = scale * queries @ torch.cat([positives, negatives]).T
    scores = scores.masked_fill(blocked, float("-inf"))
    return nn.functional.cross_entropy(scores, torch.arange(len(queries), device=scores.device))


def blocked_candidates(batch: Sequence[Triplet], judged: Judged) -> torch.Tensor:
    """True where a candidate is Exact for the row's query and is not the row's own positive."""
    candidates = [t.positive for t in batch] + [t.negative for t in batch]
    out = torch.zeros((len(batch), len(candidates)), dtype=torch.bool)
    for i, triplet in enumerate(batch):
        exact = judged[triplet.query]
        for j, product in enumerate(candidates):
            out[i, j] = j != i and exact.get(product) == "E"
    return out


def learning_rate_factor(step: int, total: int, warmup_share: float) -> float:
    """Linear warm-up, then linear decay to 0."""
    warmup = max(1, round(total * warmup_share))
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


class Checkpoints:
    """Two alternating slots, so a crash while writing never destroys the only copy.

    Files are overwritten and not deleted, because deleted files on Google Drive go to the bin and
    still count against the quota.
    """

    def __init__(self, directory: Path, identity: Mapping[str, Any]) -> None:
        """Use ``directory``; ``identity`` describes the run the checkpoints belong to."""
        self.directory = directory
        self.identity = dict(identity)
        directory.mkdir(parents=True, exist_ok=True)

    def _slots(self) -> list[tuple[int, Path]]:
        found = []
        for name in ("a", "b"):
            note = self.directory / f"slot_{name}.json"
            if note.exists():
                try:
                    info = json.loads(note.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if info.get("complete"):
                    if info["identity"] != self.identity:
                        raise ValueError(
                            f"{self.directory} holds a run with other settings or data. "
                            "Use another folder, or delete it to start again."
                        )
                    found.append((int(info["step"]), self.directory / f"slot_{name}.pt"))
        return sorted(found)

    def save(self, step: int, state: dict[str, Any]) -> None:
        """Write the older slot and mark it complete last."""
        slots = self._slots()
        used = [path.stem[-1] for _, path in slots]
        name = next((n for n in ("a", "b") if n not in used), used[0] if used else "a")
        note = self.directory / f"slot_{name}.json"
        body = {"step": step, "identity": self.identity}
        note.write_text(json.dumps(body | {"complete": False}), encoding="utf-8")
        torch.save(state, self.directory / f"slot_{name}.pt")
        note.write_text(json.dumps(body | {"complete": True}), encoding="utf-8")

    def latest(self) -> dict[str, Any] | None:
        """Newest complete checkpoint."""
        slots = self._slots()
        if not slots:
            return None
        return torch.load(slots[-1][1], map_location="cpu", weights_only=True)


def train(
    model: nn.Module,
    tokenizer: Any,  # noqa: ANN401 - transformers tokenizers share no base type worth importing
    data: TrainingData,
    batches: Sequence[Sequence[Triplet]],
    config: TrainConfig,
    checkpoints: Checkpoints,
    query_prefix: str,
    device: str,
    log: Log = print,
) -> dict[str, Any]:
    """Run until the epoch ends or the time limit is reached. Call again to continue."""
    started = time.monotonic()
    total = len(batches)
    model.to(device)
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
    half = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=half)
    step, history = 0, []
    state = checkpoints.latest()
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, history = int(state["step"]), list(state["history"])
        log(f"resumed at step {step} of {total}")

    def save() -> None:
        checkpoints.save(
            step,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "history": history,
            },
        )

    def tokens(texts: list[str], limit: int) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=limit, return_tensors="pt"
        )
        return {k: v.to(device) for k, v in encoded.items()}

    model.train()
    running: list[float] = []
    first = step
    while step < total:
        if (time.monotonic() - started) / 60 >= config.max_minutes:
            save()
            log(f"time limit reached at step {step} of {total}; run the cell again to continue")
            break
        batch = batches[step]
        # Seeding each step makes a resumed run identical to an uninterrupted one.
        torch.manual_seed(config.seed + step)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * learning_rate_factor(
                step, total, config.warmup_share
            )
        queries = tokens(
            [query_prefix + data.queries[t.query] for t in batch], config.max_query_tokens
        )
        passages = tokens(
            [data.passages[t.positive] for t in batch] + [data.passages[t.negative] for t in batch],
            config.max_passage_tokens,
        )
        with torch.autocast(device, enabled=half):
            q, p = embed(model, queries), embed(model, passages)
        loss = contrastive_loss(
            q,
            p[: len(batch)],
            p[len(batch) :],
            blocked_candidates(batch, data.train).to(device),
            config.scale,
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        running.append(float(loss.detach()))
        if step % config.log_every == 0 or step == total:
            history.append([step, sum(running) / len(running)])
            minutes = (time.monotonic() - started) / 60
            left = minutes / max(1, step - first) * (total - step)
            log(
                f"step {step}/{total}  loss {history[-1][1]:.4f}  "
                f"{minutes:.0f} min, about {left:.0f} min left"
            )
            running = []
        if step % config.checkpoint_every == 0 or step == total:
            save()
    return {"step": step, "total": total, "finished": step >= total, "history": history}


@torch.no_grad()
def encode(
    model: nn.Module,
    tokenizer: Any,  # noqa: ANN401
    texts: Sequence[str],
    limit: int,
    device: str,
    batch_size: int = 128,
) -> torch.Tensor:
    """Unit vectors on the CPU, in input order."""
    model.to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[torch.Tensor | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in chunk],
            padding=True,
            truncation=True,
            max_length=limit,
            return_tensors="pt",
        )
        vectors = embed(model, {k: v.to(device) for k, v in encoded.items()}).cpu()
        for i, vector in zip(chunk, vectors, strict=True):
            out[i] = vector
    return torch.stack([v for v in out if v is not None])


def ndcg(gains: Sequence[float]) -> float:
    """Full-list nDCG of gains in ranked order."""
    best = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)))
    got = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    return got / best if best else 0.0


def held_out_ndcg(
    model: nn.Module,
    tokenizer: Any,  # noqa: ANN401
    data: TrainingData,
    config: TrainConfig,
    query_prefix: str,
    device: str,
) -> float:
    """Rank the judged products of each held-out ESCI query. Queries with no Exact are skipped."""
    ids = sorted(q for q, v in data.held_out.items() if "E" in v.values())
    products = sorted({p for q in ids for p in data.held_out[q]})
    row = {p: i for i, p in enumerate(products)}
    texts = [query_prefix + data.queries[q] for q in ids]
    queries = encode(model, tokenizer, texts, config.max_query_tokens, device)
    passages = [data.passages[p] for p in products]
    vectors = encode(model, tokenizer, passages, config.max_passage_tokens, device)
    scores = []
    for q, vector in zip(ids, queries, strict=True):
        judged = sorted(data.held_out[q])
        values = vectors[[row[p] for p in judged]] @ vector
        ranked = sorted(zip(values.tolist(), judged, strict=True), key=lambda x: (-x[0], x[1]))
        scores.append(ndcg([ESCI_GAINS[data.held_out[q][p]] for _, p in ranked]))
    return sum(scores) / len(scores)
